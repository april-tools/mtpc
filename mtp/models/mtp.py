import os
import torch

from torch import Tensor
from copy import deepcopy

from .lm import LM
from .mtp_head import MultiTokenHead
from .circuits import CircuitCP
from .circuit_layers import TorchBatchedCategoricalLayer, TorchBatchedSumLayer
from .loss import compute_full_kl, compute_binary_approx_kl, compute_cross_entropy


class MultiTokenLM(torch.nn.Module):
    """A MultiTokenLM comprises three parts:

    1. A LM encoder, which can be the encoder (i.e. arch without lm_head)
    of any pretrained LLM. The encoder provides contextual embeddings for
    tokens.

    2. A mt_head, which expands the contextual embeddings into parameters
    for the (circuit) output layer.

    3. A circuit which models the output tokens and encodes their dependencies.

    init_from_lm_head: whether to initialise the unembedding matrix from the LM head.

    beta: float, weighting for tradeing off KL loss vs CE loss.
        0 means CE only, 1 means KL only, values in between trade-off.
    gamma: float, discount factor for each successive token in the multi-token
        block. 1 weighs all tokens the same.
    kl_type: string, the type of KL to use, can be forward or reverse.
    """

    def __init__(
        self,
        lm: LM,
        mt_head: MultiTokenHead,
        circuit: CircuitCP,
        init_from_lm_head: bool = True,
        beta: float = .9,
        gamma: float = 1.,
        kl_type: str = 'forward',
        kl_algorithm: str = 'binary_approx'
    ):
        super().__init__()
        self.lm = lm
        self.mt_head = mt_head
        self.circuit = circuit

        self.init_from_lm_head = init_from_lm_head

        # Below are the params for weighting the kl and ce losses.
        # Keep these globally to avoid shooting ourselves in the foot
        # by computing train and validation with different hyperparams
        assert 0 <= beta <= 1, 'Expected 0 <= beta <= 1, got: %.2f' % beta
        assert 0 < gamma <= 1, 'Expected 0 <= gamma <= 1, got: %.2f' % gamma
        assert kl_type in ('forward', 'reverse'), 'Unknown kl_type: %s' % kl_type
        assert kl_algorithm in ('full', 'binary_approx'), 'Unknown kl_algorithm: %s' % kl_algorithm
        self.beta = beta
        self.gamma = gamma
        self.kl_type = kl_type
        self.kl_algorithm = kl_algorithm

        # Keep track of what we need to compute
        self.compute_ce, self.compute_kl = self.beta < 1, self.beta > 0

        if self.compute_kl:
            # NOTE: We compute teacher_log_probs in a no_grad block.
            assert self.lm.freeze is True, 'Unfreezing LM with KL loss is not currently supported'
            assert self.lm.encoder_only is False, 'We need the LM head to compute KL'
        else:
            # We need encoder_only is false during generation - due to speculative decoding
            if not (os.environ.get('MODE', None) == 'generate'):
                assert self.lm.encoder_only is True, 'We do not need the LM head since we are not computing KL'

        # Retrieve the circuit layers to parameterize
        layers = list(self.circuit.circuit.topological_ordering())
        self._cat_layer: TorchBatchedCategoricalLayer = layers[self.circuit.cat_layer_idx]
        assert isinstance(self._cat_layer, TorchBatchedCategoricalLayer)
        self._sum_layer = layers[self.circuit.sum_layer_idx]
        assert isinstance(self._sum_layer, TorchBatchedSumLayer)

        if self.init_from_lm_head:
            self.mt_head.set_unembedding_weights(lm.lm_head_weights)
        del lm.lm_head_weights

    # def forward(
    #     self,
    #     xx: Tensor,
    #     yy: Tensor,
    #     return_log_probs: bool = False,
    #     return_stp_loss: bool = False
    # ) -> dict:
    #     # Compute the loss, i.e., the multi-token average negated log-likelihood
    #
    #     # xx: (B, S, D)
    #     xx = self.lm.encoder(xx)['last_hidden_state']
    #
    #     # At training time, we want to learn to predict the next H tokens
    #     # xx: (B, S', D), where S' = S - H + 1
    #     xx = xx[:, : xx.shape[1] - self.mt_head.n_token + 1]
    #
    #     # Parameterize the circuit
    #     self.parameterize_circuit(xx)
    #
    #     # Compute sliding windows indices
    #     # from yy: (B, S) to yy: (B, S', H)
    #     # where S' = S - H + 1
    #     yy = yy.unfold(dimension=1, size=self.mt_head.n_token, step=1)
    #     # Note that we unsqueeze a channel dimension, as required by cirkit
    #     # yy: (B, S', H) -> (B * S', 1, H)
    #     yy = yy.reshape(-1, 1, yy.shape[2])
    #
    #     # Compute the conditional log-likelihoods
    #     # yy: (B * S', 1, H)
    #     # log_probs: (B * S', 1, 1)
    #     log_probs = self.circuit(yy)
    #
    #     # The loss is the negated average conditional log-likelihood
    #     mtp_loss = -log_probs.mean()
    #     if not return_log_probs:
    #         log_probs = None
    #
    #     if return_stp_loss:
    #         # Compute also the single token loss, if needed
    #         stp_loss = self.compute_next_token_loss(yy)
    #     else:
    #         stp_loss = None
    #
    #     return dict(log_probs=log_probs, loss=mtp_loss, mtp_loss=mtp_loss, stp_loss=stp_loss)

    def forward(
            self,
            xx: torch.Tensor,               # (B, S) input ids
            yy: torch.Tensor,               # (B, S) target ids
            return_log_probs: bool = False) -> dict:
        r"""
        Reference: https://arxiv.org/abs/2410.17765 , Eq 14.

        Forward pass that mixes KL-distillation from a teacher model
        and cross-entropy with ground-truth targets.

        The total loss for each predicted token k = 1..H is:
            L_k = β * KL( p^c_k || p^d_k ) + (1 - β) * CE( p^d_k, x_{k} )
        possibly multiplied by a discount factor gamma^(k),
        and summed over all tokens.

        Args:
        xx: shape (B, S), the input token indices
        yy: shape (B, S), the target token indices (offset by one)

          xx :      | t1 | t2 | t3 | t4 | t5 | t6 |
          yy :           | t2 | t3 | t4 | t5 | t6 | t7 |

        Returns:
        A dictionary with keys:
            'loss': the combined loss used for training
            'kl_loss_at_h': the kl loss for token h (for h in H)
            'ce_loss_at_h': the cross entropy loss for token h (for h in H)
            'log_probs': the log probs from the draft model, if
                return_log_probs is True.
        """
        H = self.mt_head.n_token
        # B = xx.shape[0]
        S = xx.shape[1]
        # R = self.mt_head.n_component
        # V = self.mt_head.vocab_size

        # 1) Encode the inputs with the underlying LM (backbone).
        #    shape -> (B, S, D)
        xx = self.lm.encoder(xx)['last_hidden_state']

        # 2) Compute teacher log probs. We do this before truncating xx.
        #  teacher_log_probs: shape (B * S', H, V)
        if self.compute_kl:
            with torch.no_grad():
                # shape: (B, S, V)
                logits = self.lm.head(xx)
                # shape: B, S, V
                teacher_log_probs = torch.log_softmax(logits, axis=-1)
                if self.kl_algorithm == 'binary_approx':
                    # We only need the log probs for the target category
                    # shape: B, S, 1
                    teacher_log_probs = torch.gather(teacher_log_probs,
                                                     dim=-1,
                                                     index=yy.unsqueeze(-1))
                # Make teacher_log_probs windowed for kl with circuit logprobs
                # TODO: Since we are using a for loop in the KL computation
                # shape: B, S', V, H
                teacher_log_probs = teacher_log_probs.unfold(dimension=1, size=H, step=1)
                # shape: H, B, S', V
                teacher_log_probs = teacher_log_probs.permute(3, 0, 1, 2)
                # shape: H, B * S', V
                teacher_log_probs = teacher_log_probs.flatten(1, 2)
                # If V=1 because of binary approx, remove the dim
                teacher_log_probs = teacher_log_probs.squeeze(-1)
        else:
            teacher_log_probs = None

        # 3) Truncate the activations
        # For multi-token training, for each position, t, we predict the next
        # H tokens in one forward pass. This means we run out of future tokens
        # at position S' = S - H + 1. E.g., for H=3, the prediction windows:
        # xx :      | t1 |
        # yy :           | t2 | t3 | t4 |
        #                      ...
        #                      ...
        # xx :      | t1 | t2 | t3 | t4 |
        # yy :                          | t5 | t6 | t7 |
        #
        # So S' = S - H + 1 = 6 - 3 + 1 = 4
        # xx: (B, S', D), where S' = S - H + 1
        history_idx = S - H + 1
        xx = xx[:, : history_idx]

        # 4) Parameterize the circuit with our NN activations
        self.parameterize_circuit(xx)

        # 5) Make target idxs, yy, windowed
        # from yy: (B, S) to yy: (B, S', H)
        yy = yy.unfold(dimension=1, size=H, step=1)
        # We also unsqueeze a channel dimension, as required by cirkit
        # yy: (B, S', H) -> (B * S', 1, H)
        yy = yy.reshape(-1, 1, H)

        # 6) Compute draft log probs with the circuit
        if self.compute_kl and self.kl_algorithm == 'full':
            # shape: H, B * S', V   Compute conditional distributions for circuit
            log_probs = self.circuit.autoregressive_conditionals(yy=yy, with_logits=True)
        else:
            # shape: H, B * S'  We do not need to expand logits
            log_probs = self.circuit.autoregressive_conditionals(yy=yy, with_logits=False)

        # 7) Compute CE loss per token and, optionally, KL loss
        losses = self.compute_per_token_losses(
            yy,
            draft_log_probs=log_probs,
            teacher_log_probs=teacher_log_probs
        )

        # 8) Weigh the losses and optionally discount
        sum_combined_loss = 0
        loss_for_log = dict()
        for k in range(H):

            kl_loss = losses['kl_loss'][k] if self.compute_kl else 0.
            ce_loss = losses['ce_loss'][k] if self.compute_ce else 0.

            # L_k = β * KL( p^c_k || p^d_k ) + (1 - β) * CE( p^d_k, x_{k} )
            combined_loss = self.beta * kl_loss + (1.0 - self.beta) * ce_loss

            # Possibly discount by gamma^k (no discount if gamma = 1.)
            sum_combined_loss += (self.gamma ** k) * combined_loss

            # Compute losses for logging / these are detached outside
            if self.compute_kl:
                if self.kl_algorithm == 'full':
                    loss_for_log['kl_loss_at_%d' % (k+1)] = kl_loss
                elif self.kl_algorithm == 'binary_approx':
                    loss_for_log['kl_loss_ba_at_%d' % (k+1)] = kl_loss

            if self.compute_ce:
                loss_for_log['ce_loss_at_%d' % (k+1)] = ce_loss

        # We the loss to stay on same scale for more tokens
        # and for change of gamma - gamma should only scale relatively
        sum_combined_loss = sum_combined_loss / sum([self.gamma ** k
                                                     for k in range(H)])

        outputs = {'loss': sum_combined_loss}
        if self.compute_kl or self.compute_ce:
            outputs.update(loss_for_log)
        if return_log_probs:
            # TODO: standardize format of returned log_probs
            outputs['log_probs'] = log_probs
        return outputs

    def parameterize_circuit(self, xx: Tensor, generate: bool = False):
        # TODO: Make this a parameterise function on the circuit

        # Free previous tensors before we produce new ones
        # this is important, since cat_layer probs is a large tensor
        self._cat_layer.log_probs = None
        self._sum_layer.weight = None

        # Obtain dictionary of circuit parameters
        circuit_params = self.mt_head(xx, generate=generate)

        # cat_logits: (H, B, S', R, V)
        cat_log_probs = circuit_params["cat_log_probs"]
        # sum_weight: (B, S', 1, R)
        sum_weight = circuit_params["sum_weight"]

        # cat_log_probs: (H, B * S', R, V)
        cat_log_probs = cat_log_probs.view(cat_log_probs.shape[0], -1, cat_log_probs.shape[3], cat_log_probs.shape[4])
        # sum_weight: (1, B * S', 1, R)
        sum_weight = sum_weight.view(1, -1, 1, sum_weight.shape[3])

        # Set the parameters of the circuit
        self._cat_layer.log_probs = cat_log_probs
        self._sum_layer.weight = sum_weight

    @torch._dynamo.disable
    def compute_next_token_loss(self, yy: Tensor) -> Tensor:
        # We keep track of next token prediction loss too, in order to discern
        # how good the model would be for just next token prediction
        #
        # yy: (B * S', 1, H)
        # log_probs: (B * S')
        log_probs = self.circuit.univariate_marginal_at_k(k=0, yy=yy, with_logits=False)
        stp_loss = -log_probs.mean()
        # scalar
        return stp_loss

    @torch._dynamo.disable
    def compute_all_next_token_losses(self, yy: Tensor) -> Tensor:
        # We keep track of next token prediction loss too, in order to discern
        # how good the model would be for just next token prediction
        #
        # yy: (B * S', 1, H)
        # all_log_probs: (H, B * S')
        all_log_probs = self.circuit.autoregressive_conditionals(yy=yy, with_logits=False)
        stp_losses = -all_log_probs.mean(dim=1)
        # H dims
        return stp_losses

    @torch._dynamo.disable
    def compute_next_token_log_probs(self) -> Tensor:
        next_token_log_probs = self.circuit.univariate_marginal_at_k(k=0, with_logits=True)
        # BS, V
        return next_token_log_probs

    @torch._dynamo.disable
    def compute_all_token_log_probs(self, yy: Tensor) -> Tensor:
        all_token_log_probs = self.circuit.autoregressive_conditionals(yy=yy, with_logits=True)
        # H, BS, V
        return all_token_log_probs

    @torch._dynamo.disable
    def compute_per_token_losses(
        self,
        yy: Tensor,
        draft_log_probs: Tensor,
        teacher_log_probs: Tensor = None
    ) -> dict:
        """ Compute losses per token. If teacher_log_probs is passed as an
        argument, we compute both KL and CE losses for each token.
        Otherwise, we compute only per token CE loss.

        Args:
            yy: shape (H, BS), the target token indices
            draft_log_probs: shape (H, BS, V) or (H, BS), the log probs from the draft model
            teacher_log_probs: shape (H, BS, V) or (H, BS), the categorical distributions
                from the teacher model, windowed for easy kl computation.
        """
        if self.compute_kl:
            assert teacher_log_probs is not None, 'Expected teacher_log_probs != None'

        losses = dict()
        if self.compute_kl:
            B, S, V = teacher_log_probs.shape
            assert V == self.circuit.vocab_size, 'Circuit and teacher have different vocab size'

            if self.kl_algorithm == 'full':
                losses['kl_loss'] = compute_full_kl(log_probs, teacher_log_probs, self.kl_type)
            elif self.kl_algorithm == 'binary_approx':
                losses['kl_loss'] = compute_binary_approx_kl(log_probs, teacher_log_probs, self.kl_type)
            else:
                raise ValueError('Unknown kl_algorithm = %s' % self.kl_algorithm)

            if self.compute_ce:
                losses['ce_loss'] = compute_cross_entropy(log_probs, yy)
        else:
            losses['ce_loss'] = compute_cross_entropy(log_probs, yy=None)

        return losses

    @torch.no_grad()
    def generate(self, inputs: Tensor,
                 use_argmax: bool = False,
                 mode: str = 'mtp',
                 use_cache: bool = False,
                 past_key_values: Tensor = None,
                 past_last_hidden_states: Tensor = None) -> Tensor:
        if mode == 'mtp' and use_argmax:
            raise ValueError('Only multi-token generation by sampling is supported')
        if use_argmax and mode != 'stp':
            raise ValueError('Argmax is only supported for single token prediction')

        if use_cache:
            seen_tokens = 0
            if past_key_values is not None:
                seen_tokens = past_key_values.get_seq_length()
            outputs = self.lm.encoder(
                inputs[:, seen_tokens:],
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            if past_last_hidden_states is None:
                # xx: (B, S, D)
                xx = outputs['last_hidden_state']
            else:
                xx = torch.cat([past_last_hidden_states, outputs['last_hidden_state']], axis=1)
            past_key_values = outputs['past_key_values']
        else:
            xx = self.lm.encoder(inputs)['last_hidden_state']

        # Parameterize the circuit
        self.parameterize_circuit(xx, generate=True)

        if mode == 'mtp':
            # Sample the next tokens
            tokens, _ = self.circuit.sample(num_samples=1)
            # Remove extraneous channel dimension
            tokens = tokens.squeeze(dim=1)
        elif mode == 'stp':
            next_token_probs = torch.exp(self.compute_next_token_log_probs())
            if use_argmax:
                tokens = torch.argmax(next_token_probs, dim=1)
                tokens = tokens.unsqueeze(dim=1)
            else:
                tokens = torch.multinomial(next_token_probs, num_samples=1)
        return dict(tokens=tokens,
                    past_key_values=past_key_values,
                    past_last_hidden_states=xx)

    # TODO: Refactor to bring for-loop into function as per Edoardo's comment
    @torch.no_grad()
    def self_speculative_generate(self, seq: Tensor,
                                  use_cache: bool = False,
                                  past_key_values: Tensor = None,
                                  past_last_hidden_states: Tensor = None) -> Tensor:
        if len(seq.shape) != 2 or seq.shape[0] != 1:
            raise NotImplementedError("Multi-batch self-speculative decoding not implemented yet")
            # seq: (B, S), with B = 1 and also possibly S = 1

        # Compute the embeddings
        if use_cache:
            # NOTE: keep track of old values, needed for second lm eval
            old_past_key_values = deepcopy(past_key_values)
            seen_tokens = 0
            if past_key_values is not None:
                seen_tokens = past_key_values.get_seq_length()
            outputs = self.lm.encoder(
                seq[:, seen_tokens:],
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            # xx: (B, S, D)
            if past_last_hidden_states is None:
                xx = outputs['last_hidden_state']
            else:
                xx = torch.cat([past_last_hidden_states, outputs['last_hidden_state']], axis=1)
            past_key_values = outputs['past_key_values']
        else:
            xx = self.lm.encoder(seq)["last_hidden_state"]

        assert xx.shape[1] == seq.shape[1]
        # Set the circuit parameters, based on the last embeddings
        self.parameterize_circuit(xx, generate=True)

        # Sample the next H tokens
        # tokens: (B=1, 1, H) -> (B=1, H)
        tokens, _ = self.circuit.sample(num_samples=1)
        tokens = tokens.squeeze(dim=1)

        # Concatenate the tokens with the current sequence,
        # which gives the candidate next sequence
        # gen_seq: (B, S + H)
        gen_seq = torch.cat([seq, tokens], dim=1)

        # Compute the next-token probabilities in parallel
        if use_cache:
            # zz: (B, S + H, D) -> (B, H + 1, D)
            zz = self.lm.encoder(gen_seq[:, seen_tokens:], use_cache=use_cache, past_key_values=old_past_key_values)['last_hidden_state']
        else:
            zz = self.lm.encoder(gen_seq)['last_hidden_state']

        zz = zz[:, -tokens.shape[1] - 1 :]
        # logits: (B, H + 1, V)
        logits = self.lm.head(zz)

        # Determine the number of accepted tokens,
        # by iteratively computing conditional probabilities with the circuit
        #
        # To do so, we first compute context-conditioned marginals in parallel
        # q(x_{t+1} \mid x_{\leq t})
        # q(x_{t+1}, x_{t+2} \mid x_{\leq t})
        # ...
        # q(x_{t+1}, ..., x_{t+n} \mid x_{\leq t})
        #
        # log_marginal_probs: (H, 1, 1) -> (B=1, H, 1)
        log_marginal_probs = self.circuit.marginalizer(
            tokens.expand(size=(tokens.shape[1], -1)).unsqueeze(dim=1), integrate_vars=self.circuit._autoregressive_mar_mask
        )
        log_marginal_probs = log_marginal_probs.squeeze(dim=1).unsqueeze(dim=0)
        #
        # Sample H uniform noise values in [0,1), and take their log
        # log_noise: (B=1, H)
        log_noise = torch.log(torch.rand(size=(tokens.shape[0], tokens.shape[1])))
        #
        # Compute the number of tokens to accept
        num_accepted_tokens = 0
        for j in range(tokens.shape[1]):
            # Check whether noise > ratio of conditional univariate probabilities,
            # i.e., we should stop accepting tokens
            # In the log space, this becomes log noise > difference of some log probabilities,
            # which avoids many floating point divisions and is more numerically stable
            lm_next_token_log_probs = torch.log_softmax(logits[:, j], dim=1)
            # lm_jth_token_log_prob: (B, 1)
            lm_jth_token_log_prob = torch.gather(lm_next_token_log_probs, dim=1, index=tokens[:, [j]]).cpu()
            # Compute log conditional probabilities, conditioned on the context
            if j == 0:
                # q(x_{t+1}\mid x_{\leq t})
                # mtp_jth_token_log_prob: (B, 1)
                mtp_jth_token_log_prob = log_marginal_probs[:, 0].cpu()
            else:
                # q(x_{t+j}\mid x_{\leq t}, x_{t+1}, ..., x_{t+j-1}) = \
                #     q(x_{t+1}, ..., x_{t+j}\mid x_{\leq t}) / q(x_{t+1}, ..., x_{t+j-1}\mid x_{\leq t})
                # mtp_jth_token_log_prob: (B, 1)
                mtp_jth_token_log_prob = log_marginal_probs[:, j].cpu() - log_marginal_probs[:, j - 1].cpu()
            # Check noise > \
            #     (p(x_{t+j}\mid x_{\leq t}, x_{t+1}, ..., x_{t+j-1}) / q(x_{t+j}\mid x_{\leq t}, x_{t+1}, ..., x_{t+j-1}))
            if log_noise[:, j] > (lm_jth_token_log_prob - mtp_jth_token_log_prob):
                break
            num_accepted_tokens += 1

        if num_accepted_tokens == tokens.shape[1]:
            # We are so lucky! We accept all the H tokens
            # Let's index the probabilities to sample the H+1-th one
            lm_last_probs = torch.softmax(logits[:, -1], dim=1)
            # Sample the last token
            last_token = torch.multinomial(lm_last_probs, num_samples=1)
        else:  # num_accepted_tokens < tokens.shape[1]
            # We accepted H' < H tokens
            # Let's adjust the probabilities to sample the H'+1-th one
            # lm_last_probs: (B, V)
            lm_last_probs = torch.softmax(logits[:, num_accepted_tokens], dim=1)
            # Let j be the number of accepted tokens, then
            # max(0, p(x_{t+j+1}\mid x_{\leq t+j}) - q(x_{t+j+1}\mid x_{\leq t+j}))
            # under the consideration that
            # q(x_{t+j+1}\mid x_{\leq t+j}) = \
            #     q(x_{t+1}, ..., x_{t+j+1}\mid x_{\leq t}) / q(x_{t+1}, ..., x_{t+j}\mid x_{\leq t})
            # mtp_jp1th_tokens: (B=1, 1, H)
            mtp_jp1th_tokens = tokens.clone().unsqueeze(dim=1)
            mtp_jp1th_tokens[:, :, num_accepted_tokens] = -1
            if num_accepted_tokens + 1 == tokens.shape[1]:
                # mtp_jp1th_token_log_probs: (B * V, 1, 1)
                mtp_jp1th_token_log_probs = self.circuit(mtp_jp1th_tokens)
            else:
                # mtp_jp1th_token_log_probs: (B * V, 1, 1)
                mtp_jp1th_token_log_probs = self.circuit.marginalizer(
                    mtp_jp1th_tokens,
                    integrate_vars=self.circuit._autoregressive_mar_mask[num_accepted_tokens],
                )
            # mtp_jp1th_token_log_probs: (B * V, 1, 1) -> (B, V)
            mtp_jp1th_token_log_probs = mtp_jp1th_token_log_probs.view(tokens.shape[0], self.mt_head.vocab_size)
            # mtp_last_probs: (B, V)
            if num_accepted_tokens == 0:
                mtp_last_probs = torch.exp(mtp_jp1th_token_log_probs)
            else:
                mtp_last_probs = torch.exp(mtp_jp1th_token_log_probs - log_marginal_probs[:, num_accepted_tokens - 1])
            adj_last_probs = torch.clamp_min(lm_last_probs - mtp_last_probs, min=1e-15)
            adj_last_probs = adj_last_probs / torch.sum(adj_last_probs, dim=1, keepdim=True)
            # Sample the last token
            last_token = torch.multinomial(adj_last_probs, num_samples=1)

        # Retrieve the accepted tokens, plus the last one
        tokens = torch.cat([tokens[:, :num_accepted_tokens], last_token], dim=1)
        return dict(tokens=tokens,
                    past_key_values=past_key_values,
                    past_last_hidden_states=xx)
