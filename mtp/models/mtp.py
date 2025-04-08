import os
import torch

from torch import Tensor
from copy import deepcopy

from .lm import LM
from .mtp_head import MultiTokenHead

from .circuits import CircuitModel
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
        circuit: CircuitModel,
        mt_head_kwargs: dict,
        init_from_lm_head: bool = True,
        beta: float = 0.9,
        gamma: float = 1.0,
        kl_type: str = 'forward',
        kl_algorithm: str = 'binary_approx'
    ):
        super().__init__()
        self.lm = lm
        self.circuit = circuit
        self.mt_head = MultiTokenHead(
            self.circuit.parameters_config,
            self.circuit.vocab_size,
            n_embd=mt_head_kwargs['n_embd'],
            transformer_n_head=mt_head_kwargs.get('transformer_n_head', 1),
            tok_transformer_n_layer=mt_head_kwargs.get('tok_transformer_n_layer', 0),
            sum_transformer_n_layer=mt_head_kwargs.get('sum_transformer_n_layer', 0),
            expander_n_layer=mt_head_kwargs.get('expander_n_layer', 1),
            expander_type=mt_head_kwargs.get('expander_type', 'mlp'),
            freeze_vocab_unembedding=mt_head_kwargs.get('freeze_vocab_unembedding', False)
        )
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
            assert self.lm.freeze, 'Unfreezing LM with KL loss is not currently supported'
            assert not self.lm.encoder_only, 'We need the LM head to compute KL'
        else:
            # We need encoder_only is false during generation - due to speculative decoding
            if not (os.environ.get('MODE', None) == 'generate'):
                assert self.lm.encoder_only, 'We do not need the LM head since we are not computing KL'

        if self.init_from_lm_head:
            self.mt_head.set_unembedding_weights(lm.lm_head_weights)

    @property
    def vocab_size(self) -> int:
        return self.circuit.vocab_size

    @property
    def n_token(self) -> int:
        return self.circuit.n_token

    def forward(
        self,
        xx: torch.Tensor,
        yy: torch.Tensor,
        return_log_probs: bool = False
    ) -> dict:
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
        H = self.n_token
        # B = xx.shape[0]
        S = xx.shape[1]
        # R = self.circuit.n_component
        # V = self.vocab_size

        # 1) Encode the inputs with the underlying LM (backbone).
        #    shape -> (B, S, D)
        xxd = self.lm.encoder(xx)['last_hidden_state']

        # 2) Compute teacher log probs. We do this before truncating xx.
        #  teacher_log_probs: shape (B * S', H, V)
        if self.compute_kl:
            with torch.no_grad(), self.lm.disable_adapter_if_any():
                if self.lm.has_adapter:
                    xxv = self.lm.encoder(xx)['last_hidden_state']
                else:  # If the LM has not adaptors, then the verifier hidden features are the same of the draft features
                    xxv = xxd
                # logits: (B, S, V)
                logits = self.lm.head(xxv)

                # shape: B, S, V
                teacher_log_probs = torch.log_softmax(logits, axis=-1)

                _, S, V = teacher_log_probs.shape
                assert V == self.circuit.vocab_size, 'Circuit and teacher have different vocab size'

                if self.kl_algorithm == 'binary_approx':
                    # We only need the log probs for the target category
                    # shape: B, S, 1
                    teacher_log_probs = torch.gather(
                        teacher_log_probs, dim=-1, index=yy.unsqueeze(-1)
                    )
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
        # xxd:      | t1 |
        # yy :           | t2 | t3 | t4 |
        #                      ...
        #                      ...
        # xxd:      | t1 | t2 | t3 | t4 |
        # yy :                          | t5 | t6 | t7 |
        #
        # So S' = S - H + 1 = 6 - 3 + 1 = 4
        # xx: (B, S', D), where S' = S - H + 1
        history_idx = S - H + 1
        xxd = xxd[:, : history_idx]

        # 4) Parameterize the circuit with our NN activations
        self._parameterize_circuit(xxd)

        # 5) Make target idxs, yy, windowed
        # from yy: (B, S) to yy: (B, S', H)
        yy = yy.unfold(dimension=1, size=H, step=1)
        # yy: (B, S', H) -> (B * S', H)
        yy = yy.reshape(-1, H)

        # 6) Compute draft log probs with the circuit
        if self.compute_kl and self.kl_algorithm == 'full':
            # shape: H, B * S', V   We need the full conditional distributions
            log_probs = self.circuit.autoregressive_conditionals(yy=yy, with_logits=True)
        else:
            # shape: H, B * S'  We need conditional distributions for yy only
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
        sum_combined_loss = sum_combined_loss / sum([self.gamma ** k for k in range(H)])

        outputs = {'loss': sum_combined_loss}
        if self.compute_kl or self.compute_ce:
            outputs.update(loss_for_log)
        if return_log_probs:
            # TODO: fix below. We should not be recomputing things here
            # but we would need to standardize what log probs we return
            # currently this would differ depending on with_logits or not
            lp = self.circuit(yy)
            outputs['log_probs'] = lp
            outputs['full_log_probs'] = log_probs
        return outputs

    def _parameterize_circuit(self, xx: Tensor, generate: bool = False):
        # Obtain dictionary of circuit parameters
        circuit_params = self.mt_head(xx, generate=generate)

        # Set the parameters to the circuit
        self.circuit.parameterize(circuit_params)

    def compute_next_token_loss(self, yy: Tensor) -> Tensor:
        # We keep track of next token prediction loss too, in order to discern
        # how good the model would be for just next token prediction
        #
        # yy: (B * S', H)
        # log_probs: (B * S')
        log_probs = self.circuit.univariate_marginal_at_k(k=0, yy=yy, with_logits=False)
        stp_loss = -log_probs.mean()
        # scalar
        return stp_loss

    def compute_all_next_token_losses(self, yy: Tensor) -> Tensor:
        # We keep track of next token prediction loss too, in order to discern
        # how good the model would be for just next token prediction
        #
        # yy: (B * S', H)
        # all_log_probs: (H, B * S')
        all_log_probs = self.circuit.autoregressive_conditionals(yy=yy, with_logits=False)
        stp_losses = -all_log_probs.mean(dim=1)
        # H dims
        return stp_losses

    def compute_next_token_log_probs(self) -> Tensor:
        next_token_log_probs = self.circuit.univariate_marginal_at_k(k=0, with_logits=True)
        # BS, V
        return next_token_log_probs

    def compute_all_token_log_probs(self, yy: Tensor) -> Tensor:
        all_token_log_probs = self.circuit.autoregressive_conditionals(yy=yy, with_logits=True)
        # H, BS, V
        return all_token_log_probs

    def compute_per_token_losses(
        self,
        yy: Tensor,
        draft_log_probs: Tensor,
        teacher_log_probs: Tensor = None
    ) -> dict:
        """ Compute per token losses.

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
            if self.kl_algorithm == 'full':
                losses['kl_loss'] = compute_full_kl(draft_log_probs, teacher_log_probs, self.kl_type)
            elif self.kl_algorithm == 'binary_approx':
                losses['kl_loss'] = compute_binary_approx_kl(draft_log_probs, teacher_log_probs, self.kl_type)
            else:
                raise ValueError('Unknown kl_algorithm = %s' % self.kl_algorithm)
        if self.compute_ce:
            # If we have only computed the gold log probs
            if len(draft_log_probs.shape) == 2:
                losses['ce_loss'] = compute_cross_entropy(draft_log_probs, yy=None)
            else:
                losses['ce_loss'] = compute_cross_entropy(draft_log_probs, yy)

        return losses

    @torch.no_grad()
    def generate(
        self, inputs: Tensor,
        use_argmax: bool = False,
        mode: str = 'mtp',
        use_cache: bool = False,
        past_key_values: Tensor = None,
        past_last_hidden_states: Tensor = None
    ) -> Tensor:
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
        self._parameterize_circuit(xx, generate=True)

        if mode == 'mtp':
            # Sample the next tokens
            tokens, _ = self.circuit.sample(num_samples=1)
        elif mode == 'stp':
            next_token_probs = torch.exp(self.compute_next_token_log_probs())
            if use_argmax:
                tokens = torch.argmax(next_token_probs, dim=1)
                tokens = tokens.unsqueeze(dim=1)
            else:
                tokens = torch.multinomial(next_token_probs, num_samples=1)
        return dict(
            tokens=tokens,
            past_key_values=past_key_values,
            past_last_hidden_states=xx
        )

    # TODO: Refactor to bring for-loop into function as per Edoardo's comment
    @torch.no_grad()
    def self_speculative_generate(
        self,
        seq: Tensor,
        use_cache: bool = False,
        past_key_values: Tensor = None,
        past_last_hidden_states: Tensor = None
    ) -> Tensor:
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
        self._parameterize_circuit(xx, generate=True)

        # Sample the next H tokens
        # tokens: (B=1, H)
        tokens, _ = self.circuit.sample(num_samples=1)

        # Concatenate the tokens with the current sequence,
        # which gives the candidate next sequence
        # gen_seq: (B, S + H)
        gen_seq = torch.cat([seq, tokens], dim=1)

        # Compute the next-token probabilities in parallel
        with self.lm.disable_adapter_if_any():
            if use_cache:
                results = self.lm.encoder(
                    gen_seq[:, seen_tokens:], use_cache=use_cache, past_key_values=old_past_key_values
                )
            else:
                results = self.lm.encoder(gen_seq)
            # zz: (B, S + H, D) -> (B, H + 1, D)
            zz = results['last_hidden_state'][:, -tokens.shape[1] - 1 :]
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
            tokens.expand(size=(tokens.shape[1], -1)), integrate_vars=self.circuit._autoregressive_mar_mask
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
            # mtp_jp1th_tokens: (B=1, H)
            mtp_jp1th_tokens = tokens.clone()
            mtp_jp1th_tokens[:, num_accepted_tokens] = -1
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
            mtp_jp1th_token_log_probs = mtp_jp1th_token_log_probs.view(tokens.shape[0], self.vocab_size)
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
        return dict(
            tokens=tokens,
            past_key_values=past_key_values,
            past_last_hidden_states=xx
        )
