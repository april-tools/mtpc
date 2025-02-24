import torch

from torch import Tensor
from cirkit.utils.scope import Scope
from cirkit.backend.torch.queries import SamplingQuery, IntegrateQuery

from .lm import LM
from .mtp_head import MultiTokenHead

from .circuits import CircuitCP
from .circuit_layers import TorchBatchedCategoricalLayer, TorchBatchedSumLayer


class MultiTokenLM(torch.nn.Module):
    """A MultiTokenLM comprises three parts:

    1. A LM encoder, which can be the encoder (i.e. arch without lm_head)
    of any pretrained LLM. The encoder provides contextual embeddings for
    tokens.

    2. A mt_head, which expands the contextual embeddings into parameters
    for the (circuit) output layer.

    3. A circuit which models the output tokens and encodes their dependencies.
    """

    def __init__(self, lm: LM, mt_head: MultiTokenHead, circuit: CircuitCP, init_from_lm_head: bool = True):
        super().__init__()
        self.lm = lm
        self.mt_head = mt_head
        self.circuit = circuit
        self.init_from_lm_head = init_from_lm_head

        # Retrieve the circuit layers to parameterize
        layers = list(self.circuit.circuit.topological_ordering())
        self._cat_layer: TorchBatchedCategoricalLayer = layers[self.circuit.cat_layer_idx]
        assert isinstance(self._cat_layer, TorchBatchedCategoricalLayer)
        self._sum_layer = layers[self.circuit.sum_layer_idx]
        assert isinstance(self._sum_layer, TorchBatchedSumLayer)

        if self.init_from_lm_head:
            self.mt_head.set_unembedding_weights(lm.lm_head_weights)
        del lm.lm_head_weights

        # Initializer the sampler and the marginalizer objects
        self.sampler = SamplingQuery(self.circuit._circuit)
        self.marginalizer = IntegrateQuery(self.circuit.circuit)

        # Cache some constants used in self-speculative decoding
        mar_scopes = list(
            reversed([Scope(self.mt_head.n_token - i - 1 for i in range(t)) for t in range(self.mt_head.n_token)])
        )
        self.register_buffer(
            "_autoregressive_mar_mask", IntegrateQuery.scopes_to_mask(self.circuit.circuit, mar_scopes)
        )

    def forward(
        self,
        xx: Tensor,
        yy: Tensor,
        return_log_probs: bool = False,
        return_stp_loss: bool = False
    ) -> dict[str, Tensor]:
        # Compute the loss, i.e., the multi-token average negated log-likelihood

        # xx: (B, S, D)
        xx = self.lm.encoder(xx)['last_hidden_state']

        # At training time, we want to learn to predict the next H tokens
        # xx: (B, S', D), where S' = S - H + 1
        xx = xx[:, : xx.shape[1] - self.mt_head.n_token + 1]

        # Parameterize the circuit
        self.parameterize_circuit(xx)

        # Compute sliding windows indices
        # from yy: (B, S) to yy: (B, S', H)
        # where S' = S - H + 1
        yy = yy.unfold(dimension=1, size=self.mt_head.n_token, step=1)
        # Note that we unsqueeze a channel dimension, as required by cirkit
        # yy: (B, S', H) -> (B * S', 1, H)
        yy = yy.reshape(-1, 1, yy.shape[2])

        # Compute the conditional log-likelihoods
        # yy: (B * S', 1, H)
        # log_probs: (B * S', 1, 1)
        log_probs = self.circuit(yy)

        # The loss is the negated average conditional log-likelihood
        mtp_loss = -log_probs.mean()
        if not return_log_probs:
            log_probs = None

        if return_stp_loss:
            # Compute also the single token loss, if needed
            stp_loss = self.compute_next_token_loss(yy)
        else:
            stp_loss = None

        return dict(log_probs=log_probs, loss=mtp_loss, mtp_loss=mtp_loss, stp_loss=stp_loss)

    def forward_eq14(
            self,
            xx: torch.Tensor,               # (B, S) input ids
            yy: torch.Tensor,               # (B, S) target ids
            teacher_probs: torch.Tensor,    # teacher distribution, shape (B, S', H, V) or (H, B, S', V)
            alpha: float = 0.9,             # weight for KL-distillation
            gamma: float = 1.0,             # discount factor for each token in the multi-token block
        ) -> dict:
            r"""
            Fine-tuning forward pass that mixes KL-distillation from a teacher model
            and cross-entropy with ground-truth targets.

            The total loss for each predicted token k = 1..H is:
                L_k = alpha * KL( p^c_k || p^d_k ) + (1 - alpha)* CE( p^d_k, x_{k} )
            possibly multiplied by a discount factor gamma^(k-1),
            and summed over all tokens.

            Finally, we add the usual mixture-of-experts balancing loss,
            e.g. self._aux_loss, if desired.

            Args:
            xx: shape (B, S), the input token indices
            yy: shape (B, S), the target token indices
            teacher_probs: teacher's token distributions,
                            e.g. shape (B, S', H, V) or (H, B, S', V)
                            must align with the same sliding windows as multi-token
            alpha: float, factor for KL vs CE
            gamma: float, discount factor for each successive token in the multi-token block

            Returns:
            A dictionary with keys:
                'loss': the final scalar,
                'distill_loss': the sum of distillation terms,
                'crossent_loss': the sum of cross-entropy terms,
                'aux_loss': mixture-of-experts balancing loss,
                'mtp_loss': combined total (useful to log).
            """

            # 1) Encode the inputs with the underlying LM (backbone).
            #    shape -> (B, S, D)
            embeddings = self.lm.encoder(xx)['last_hidden_state']

            # For multi-token training, we only align up to (S - H + 1).
            # Because for each position t we attempt to predict the next H tokens
            # in one forward pass.
            # shape -> (B, S - H + 1, D)
            seq_len = embeddings.shape[1]
            needed_len = seq_len - self.mt_head.n_token + 1
            embeddings = embeddings[:, :needed_len]

            # 2) Parameterize the circuit with these embeddings
            #    (this sets self._cat_layer.log_probs and self._sum_layer.weight)
            self.parameterize_circuit(embeddings)

            # 3) We'll build the new distribution p^d_{k} for each predicted token k
            #    from the mixture-of-experts circuit.
            #    cat_log_probs has shape (H, B, S', R, V), sum_weight has shape (B, S', 1, R)
            cat_log_probs = self._cat_layer.log_probs  # shape: (H, B*S', R, V)
            sum_weight   = self._sum_layer.weight      # shape: (1, B*S', 1, R)

            # Reshape them to group (B, S') again
            # cat_log_probs -> (H, B, S', R, V)
            # sum_weight    -> (B, S', R)
            H = self.mt_head.n_token
            B_ = embeddings.shape[0]
            S_ = embeddings.shape[1]  # S' = needed_len
            R  = self.mt_head.n_component
            V  = self.mt_head.vocab_size

            cat_log_probs = cat_log_probs.view(H, B_, S_, R, V)
            sum_weight    = sum_weight.view(1, B_, S_, 1, R).squeeze(0).squeeze(3)
            # sum_weight now is (B, S', R).

            # The final distribution p^d_k is sum_{alpha} w_alpha * exp(cat_log_probs[k, ...]),
            # shape: (B, S', V).
            # We can do this for each k in a loop, or vectorize. Let's do partial vector form:
            #   (B, S', 1, R) + broadcast with cat_log_probs[k,b,s,r,v]
            # Then sum over r.
            # We'll create a list of distributions for each k, or stack them up.

            # teacher_probs is supposed to match shape (H, B, S', V) or (B, S', H, V).
            # Let's unify to (H, B, S', V) for easy indexing [k, b, s, :]
            # We'll check first which dimension is first:
            if teacher_probs.shape[0] != H:
                # then presumably teacher_probs is (B, S', H, V),
                # transpose it to (H, B, S', V)
                teacher_probs = teacher_probs.permute(2, 0, 1, 3).contiguous()

            # We also need the ground-truth next tokens for each position t
            # i.e. shape (B, S) => unfold into (B, S', H). Exactly as the normal forward does
            #   or we can do a simpler approach: each predicted token is at offset +k
            # We'll do the same approach as the normal training to match indices:
            # 'yy' shape is (B, S)
            # after unfold: (B, S', H)
            # Each [b, s, k] is the target for the k-th predicted token at position s
            # NOTE: S' = S-H+1
            y_unfold = yy.unfold(dimension=1, size=H, step=1)  # (B, S', H)

            # Distillation loss
            distill_loss = torch.zeros([], device=embeddings.device)
            # Cross-entropy
            crossent_loss = torch.zeros([], device=embeddings.device)

            for k in range(H):
                # cat_log_probs for token k: shape (B, S', R, V)
                log_probs_k = cat_log_probs[k]  # (B, S', R, V)
                # sum_weight: shape (B, S', R)
                # => p^d_k(b, s, v) = \sum_{r} sum_weight[b,s,r] * exp( log_probs_k[b,s,r,v] )
                # We'll do a stable approach with log-sum-exp in R dimension:
                #  log( p^d_k(b,s,v) ) = logsumexp( log( sum_weight ) + log_probs_k, over r )
                log_w = torch.log(sum_weight + 1e-45)  # (B, S', R)
                # broadcast to match => shape (B, S', R, V)
                log_pdraft_k = log_w.unsqueeze(-1) + log_probs_k
                # log_pdraft_k(b, s, r, v)
                # then we do logsumexp over r => shape (B, S', V)
                log_pdraft_k = torch.logsumexp(log_pdraft_k, dim=2)  # sum over r
                # p^d_k:
                pdraft_k = torch.exp(log_pdraft_k)  # shape (B, S', V)

                # teacher for this token: shape (B, S', V)
                pteacher_k = teacher_probs[k]  # (B, S', V)

                #  -- 1) KL Distillation:  KL( p^c_k || p^d_k ) = sum_{v} p^c_k log( p^c_k / p^d_k )
                # We do a safe log ratio
                kl_part = pteacher_k * (torch.log(pteacher_k + 1e-45) - log_pdraft_k)
                kl_part = torch.sum(kl_part, dim=-1)  # shape (B, S')
                #  => kl_part(b, s)
                #  -- 2) CrossEntropy with real target: -log( p^d_k(b,s, y_unfold[b,s,k]) )
                # gather the log prob:
                # shape => (B, S')
                idx_k = y_unfold[:, :, k]  # the gold token (B, S')
                ce_part = -log_pdraft_k.gather(dim=-1, index=idx_k.unsqueeze(-1)).squeeze(-1)
                #  -- combine
                # L_k(b,s) = alpha * kl_part + (1-alpha)*ce_part
                this_loss = alpha * kl_part + (1.0 - alpha) * ce_part

                # Possibly discount by gamma^k
                if gamma != 1.0:
                    this_loss = (gamma ** k) * this_loss

                # accumulate
                distill_loss += (gamma ** k) * torch.sum(kl_part)
                crossent_loss += (gamma ** k) * torch.sum(ce_part)

                # sum over all b,s
            # end for k in range(H)

            # average over total number of predicted tokens
            # total count is B*S' * H tokens if we sum them all
            denom = (B_ * S_)  # the # of positions in the sliding window
            total_loss = (
                alpha * distill_loss
                + (1.0 - alpha) * crossent_loss
            ) / (denom * 1.0)

            # 4) Add mixture-of-experts balancing loss, e.g. L_aux if desired
            # Usually we track usage of each expert alpha to ensure they are balanced
            # For example:
            #    self._aux_loss = self.balancing_loss(...)
            # We'll assume you have computed that in 'self._aux_loss', or similarly
            # If not, set self._aux_loss = 0
            aux_loss = getattr(self, '_aux_loss', 0.0)
            
            # Combine
            loss = total_loss + aux_loss

            return {
                'loss': loss,
                'distill_loss': distill_loss / denom,    # for logging
                'crossent_loss': crossent_loss / denom,  # for logging
                'aux_loss': aux_loss,                    # mixture-of-experts balancing
                'mtp_loss': loss,                        # final
            }

    def parameterize_circuit(self, xx: Tensor, generate: bool = False):
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
        # log_probs: (B * S', 1, 1)
        log_probs = self.marginalizer(yy, integrate_vars=self._autoregressive_mar_mask[0])
        stp_loss = -log_probs.mean()
        return stp_loss

    @torch._dynamo.disable
    def compute_next_token_log_probs(self) -> Tensor:
        ########## TODO: to be refactored #########
        # Next token prediction - equivalent to marginalising out future tokens
        # See https://arxiv.org/pdf/2410.17765, eq. 11
        # (1, B * S', 1, V)
        next_token_cats = torch.exp(self._cat_layer.log_probs[0, :, :, :])
        # (B * S', V)
        next_token_probs = (self._sum_layer.weight @ next_token_cats).squeeze(0, 2)
        ############################################
        return next_token_probs

    @torch.no_grad()
    def generate(self, inputs: Tensor, use_argmax: bool = False, mode: str = 'mtp') -> Tensor:
        if mode == 'mtp' and use_argmax:
            raise ValueError('Only multi-token generation by sampling is supported')
        if use_argmax and mode != 'stp':
            raise ValueError('Argmax is only supported for single token prediction')

        # Calling forward with no targets simply sets the parameters to the circuit
        # xx: (B, S, D)
        xx = self.lm.encoder(inputs)['last_hidden_state']

        # Parameterize the circuit
        self.parameterize_circuit(xx, generate=True)

        if mode == 'mtp':
            # Sample the next tokens
            tokens, _ = self.sampler(num_samples=1)
            # Remove extraneous channel dimension
            tokens = tokens.squeeze(dim=1)
        elif mode == 'stp':
            next_token_probs = self.compute_next_token_log_probs()
            if use_argmax:
                tokens = torch.argmax(next_token_probs, dim=1)
                tokens = tokens.unsqueeze(dim=1)
            else:
                tokens = torch.multinomial(next_token_probs, num_samples=1)
        return tokens

    @torch.no_grad()
    def self_speculative_generate(self, seq: Tensor) -> Tensor:
        if len(seq.shape) != 2 or seq.shape[0] != 1:
            raise NotImplementedError("Multi-batch self-speculative decoding not implemented yet")
            # seq: (B, S), with B = 1 and also possibly S = 1

        # Compute the embeddings
        # xx: (B, S, D)
        xx = self.lm.encoder(seq)['last_hidden_state']

        # Set the circuit parameters, based on the last embeddings
        self.parameterize_circuit(xx, generate=True)

        # Sample the next H tokens
        # tokens: (B=1, 1, H) -> (B=1, H)
        tokens, _ = self.sampler(num_samples=1)
        tokens = tokens.squeeze(dim=1)

        # Concatenate the tokens with the current sequence,
        # which gives the candidate next sequence
        # gen_seq: (B, S + H)
        gen_seq = torch.cat([seq, tokens], dim=1)

        # Compute the next-token probabilities in parallel
        # zz: (B, S + H, D) -> (B, H + 1, D)
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
        log_marginal_probs = self.marginalizer(
            tokens.expand(size=(tokens.shape[1], -1)).unsqueeze(dim=1), integrate_vars=self._autoregressive_mar_mask
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
            mtp_jp1th_tokens = torch.zeros(
                tokens.shape[0], self.mt_head.vocab_size, tokens.shape[1], device=tokens.device, dtype=tokens.dtype
            )
            mtp_jp1th_tokens[:, :, :num_accepted_tokens] = tokens[:, :num_accepted_tokens]
            mtp_jp1th_tokens[:, :, num_accepted_tokens] = torch.arange(
                self.mt_head.vocab_size, device=tokens.device, dtype=tokens.dtype
            ).unsqueeze(dim=0)
            mtp_jp1th_tokens = mtp_jp1th_tokens.view(
                mtp_jp1th_tokens.shape[0] * mtp_jp1th_tokens.shape[1], 1, tokens.shape[1]
            )
            if num_accepted_tokens + 1 == tokens.shape[1]:
                # mtp_jp1th_token_log_probs: (B * V, 1, 1)
                mtp_jp1th_token_log_probs = self.circuit(mtp_jp1th_tokens)
            else:
                # mtp_jp1th_token_log_probs: (B * V, 1, 1)
                mtp_jp1th_token_log_probs = self.marginalizer(
                    mtp_jp1th_tokens,
                    integrate_vars=self._autoregressive_mar_mask[num_accepted_tokens],
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
        return tokens
