import torch
import torch.nn.functional as F

from torch import Tensor, LongTensor

from transformers.cache_utils import Cache
from mtp.models.evabyte.multibyte_decoding_evabyte import multi_byte_pred_prepare_attn_mask
from mtp.utils.sampling import truncate_logprobs_top_p, truncate_probs_top_p

from mtp.models.evabyte.eva_cache import EvaStaticCacheForTriton

from .lm import LM

from mtp.models.circuits import CircuitModel
from mtp.models.loss import compute_full_kl, compute_binary_approx_kl, compute_cross_entropy, compute_valid_mask, IGNORE_TOKEN_ID


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
        beta: float = 0.0,
        gamma: float = 1.0,
        kl_type: str = 'forward',
        kl_algorithm: str = 'binary_approx'
    ):
        super().__init__()
        self.lm = lm
        self.circuit = circuit

        # Select and instantiate the MTP head
        self.mt_head_type = mt_head_kwargs.get('type', 'vanilla')
        if self.mt_head_type == 'vanilla':
            from .mtp_head import MultiTokenHead as VanillaMultiTokenHead
            mtp_head_cls = VanillaMultiTokenHead
        elif self.mt_head_type == 'evabyte':
            from .evabyte.mtp_head import MultiTokenHead as EvabyteMultiTokenHead
            mtp_head_cls = EvabyteMultiTokenHead
        else:
            raise NotImplementedError(f"Unknown multi-token head called {self.mt_head_type}")
        self.mt_head = mtp_head_cls(
            self.circuit.parameters_config,
            self.circuit.vocab_size,
            n_embd=mt_head_kwargs['n_embd'],
            transformer_n_head=mt_head_kwargs.get('transformer_n_head', 1),
            transformer_n_layer=mt_head_kwargs.get('transformer_n_layer', 0),
            expander_type=mt_head_kwargs.get('expander_type', 'linear'),
            expander_n_layer=mt_head_kwargs.get('expander_n_layer', 1),
            expander_use_skip=not init_from_lm_head,
            freeze_vocab_unembedding=mt_head_kwargs.get('freeze_vocab_unembedding', False)
        )
        self.init_from_lm_head = init_from_lm_head

        # Below are the params for weighting the kl and ce losses.
        # Keep these globally to avoid shooting ourselves in the foot
        # by computing train and validation with different hyperparams
        assert 0 <= beta <= 1, "Expected 0 <= beta <= 1, got: %.2f" % beta
        assert 0 < gamma <= 1, "Expected 0 <= gamma <= 1, got: %.2f" % gamma
        assert kl_type in ("forward", "reverse"), "Unknown kl_type: %s" % kl_type
        assert kl_algorithm in ("full", "binary_approx"), (
            "Unknown kl_algorithm: %s" % kl_algorithm
        )
        self.beta = beta
        self.gamma = gamma
        self.kl_type = kl_type
        self.kl_algorithm = kl_algorithm
        self.register_buffer('_exp_gamma_weights', torch.tensor([self.gamma ** k for k in range(self.circuit.n_token)]))
        self._exp_gamma_normalizer = torch.sum(self._exp_gamma_weights).item()

        # Keep track of what we need to compute
        self.compute_ce, self.compute_kl = self.beta < 1, self.beta > 0

        if self.compute_kl:
            # NOTE: We compute teacher_log_probs in a no_grad block.
            assert self.lm.freeze, 'Unfreezing LM with KL loss is not currently supported'
            assert not self.lm.encoder_only, 'We need the LM head to compute KL'

        if self.init_from_lm_head:
            self.mt_head.set_unembedding_weights(self.lm.lm_head_weights)
        self.lm.drop_lm_head_weights()

    @property
    def vocab_size(self) -> int:
        return self.circuit.vocab_size

    @property
    def n_token(self) -> int:
        return self.circuit.n_token

    def forward(
        self,
        input_ids: LongTensor,
        labels: LongTensor,
        attention_mask: LongTensor | None = None,
        return_log_probs: bool = False,
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
        input_ids: shape (B, S), the input token indices
        labels: shape (B, S), the target token indices (offset by one)

          input_ids :      | t1 | t2 | t3 | t4 | t5 | t6 |
          labels    :           | t2 | t3 | t4 | t5 | t6 | t7 |

        attention_mask: shape(B, S), mask specifying which positions should
            be conditioned on (true for active).

        Returns:
        A dictionary with keys:
            'loss': the combined loss used for training
            'kl_loss_at_h': the kl loss for token h (for h in H)
            'ce_loss_at_h': the cross entropy loss for token h (for h in H)
            'log_probs': the log probs from the draft model, if
                return_log_probs is True.
        """
        H = self.n_token
        B = input_ids.shape[0]
        S = input_ids.shape[1]
        # R = self.circuit.n_component
        # V = self.vocab_size

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.int32)

        # Evabyte needs special treatment since they construct two types of
        # attention mask (window and block), and cannot use vanilla huggingface
        if self.mt_head_type == 'evabyte':
            # We are not packing, so pass attention=None
            # https://github.com/OpenEvaByte/evabyte/issues/6
            enc_attention_mask = None
        else:
            enc_attention_mask = attention_mask

        # 1) Encode the inputs with the underlying LM (backbone).
        #    shape -> (B, S, D)
        xxd = self.lm.encoder(input_ids=input_ids, attention_mask=enc_attention_mask)['last_hidden_state']

        # 2) Compute teacher log probs.
        #  teacher_log_probs: shape (B * S, H, V)
        if self.compute_kl:
            with torch.no_grad(), self.lm.disable_adapter_if_any():
                if self.lm.has_adapter:
                    xxv = self.lm.encoder(input_ids=input_ids, attention_mask=enc_attention_mask)['last_hidden_state']
                else:  # If the LM has no adaptors, then the verifier hidden features are the same as the draft features
                    xxv = xxd
                # logits: (B, S, V)
                logits = self.lm.head_logits(xxv)

                # shape: B, S, V
                teacher_log_probs = torch.log_softmax(logits, dim=-1)

                _, S, V = teacher_log_probs.shape
                assert V == self.circuit.vocab_size, 'Circuit and teacher have different vocab size'

                # Pad the teacher_log_probs on the right along the sequence length
                # such that when we unfold we still get S windows to predict
                # We pad with -inf to make sure we notice if we are using invalid tokens (kl would be inf or nan)
                # shape: B, S+, V   (pad starts from last dim and moves forward in pairs)
                teacher_log_probs = F.pad(teacher_log_probs, (0, 0, 0, H - 1), mode='constant', value=-torch.inf)

                if self.kl_algorithm == "binary_approx":
                    # We only need the log probs for the target category
                    # shape: B, S+, 1
                    teacher_log_probs = torch.gather(
                        teacher_log_probs, dim=-1, index=labels.unsqueeze(-1)
                    )
                # Make teacher_log_probs windowed for kl with circuit logprobs
                # TODO: Since we are using a for loop in the KL computation
                # shape: B, S, V, H
                teacher_log_probs = teacher_log_probs.unfold(
                    dimension=1, size=H, step=1
                )
                # shape: H, B, S, V
                teacher_log_probs = teacher_log_probs.permute(3, 0, 1, 2)
                # If V=1 because of binary approx, remove the dim
                teacher_log_probs = teacher_log_probs.squeeze(-1)
        else:
            teacher_log_probs = None

        # 3) Parameterize the circuit with our NN activations
        self._parameterize_circuit(xxd, attention_mask=attention_mask)

        # 4) Pad labels on the right by H - 1  (B, S+)
        yy = F.pad(labels, (0, H - 1), mode='constant', value=IGNORE_TOKEN_ID)

        # 5) Make target labels, yy, and attention masks, windowed
        # from yy: (B, S+) to yy: (B, S, H)
        yy = yy.unfold(dimension=1, size=H, step=1)
        # yy: (B, S, H) -> (B * S, H)
        yy = yy.reshape(-1, H)

        # Also process the attention mask
        yym = attention_mask.bool()
        # Pad in the same way we padded outputs
        yym = F.pad(yym, (0, H - 1), mode='constant', value=False)
        yym = yym.unfold(dimension=1, size=H, step=1)

        # We condition on tokens with attention_mask = True
        # so we want to marginalise out those with attention_mask=False
        do_not_condition_mask = ~yym.reshape(-1, H)
        # We do not predict tokens with IGNORE_TOKEN_ID
        do_not_predict_mask = ~compute_valid_mask(yy)

        # We want to marginalise out tokens that should either not be predicted
        # or tokens that should not be conditioned on
        marg_mask = do_not_condition_mask | do_not_predict_mask

        # 6) Compute draft log probs with the circuit
        if self.compute_kl and self.kl_algorithm == "full":
            # shape: H, B, S, V   We need the full conditional distributions
            log_probs = self.circuit.autoregressive_conditionals(
                yy=yy, marg_mask=marg_mask, with_logits=True
            )
            log_probs = log_probs.view(H, B, -1, V)
        else:
            # shape: H, B, S  We need conditional distributions for yy only
            log_probs = self.circuit.autoregressive_conditionals(
                yy=yy, marg_mask=marg_mask, with_logits=False
            )
            log_probs = log_probs.view(H, B, -1)

        # 7) Compute CE loss per token and, optionally, KL loss
        # First we reshape tensors so they have the same leading dims
        yy_hbs = yy.permute(1, 0).view(H, B, -1)
        losses = self.compute_per_token_losses(
            yy_hbs, draft_log_probs=log_probs, teacher_log_probs=teacher_log_probs
        )

        # 8) Weigh the losses and optionally discount
        kl_loss = losses['kl_loss'] if self.compute_kl else 0.0
        ce_loss = losses['ce_loss'] if self.compute_ce else 0.0
        # L_k = β * KL( p^c_k || p^d_k ) + (1 - β) * CE( p^d_k, x_{k} )
        combined_loss = self.beta * kl_loss + (1.0 - self.beta) * ce_loss
        # Possibly discount by gamma^k (no discount if gamma = 1.)
        # We want the loss to stay on same scale for more tokens
        # and for change of gamma - gamma should only scale relatively
        # so divide by the exp_gamma_normalizer
        avg_combined_loss = torch.sum(combined_loss * self._exp_gamma_weights, dim=-1) / self._exp_gamma_normalizer

        # Set the losses for logging / these are detached outside
        outputs = {'loss': avg_combined_loss}
        if self.compute_kl or self.compute_ce:
            for k in range(H):
                if self.compute_kl:
                    if self.kl_algorithm == 'full':
                        outputs[f'kl_loss_at_{k+1}'] = kl_loss[k]
                    elif self.kl_algorithm == 'binary_approx':
                        outputs[f'kl_loss_ba_at_{k+1}'] = kl_loss[k]
                if self.compute_ce:
                    outputs[f'ce_loss_at_{k+1}'] = ce_loss[k]

        if return_log_probs:
            # TODO: fix below. We should not be recomputing things here
            # but we would need to standardize what log probs we return
            # currently this would differ depending on with_logits or not
            lp = self.circuit(yy)
            outputs['log_probs'] = lp.detach().cpu()
            outputs['full_log_probs'] = log_probs.detach().cpu()  # ??? What is a full log probs ???

        return outputs

    def _parameterize_circuit(
        self,
        xx: Tensor,
        use_cache: bool = False,
        attention_mask: Tensor = None,
        past_key_values: Cache = None,
        position_ids: Tensor = None,
        generate: bool = False,
        top_p: float = 1.,
    ) -> Cache:
        if top_p != 1.:
            assert generate is True

        # Obtain dictionary of circuit parameters
        outputs = self.mt_head(
            xx,
            use_cache=use_cache,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            generate=generate
        )

        # Set the parameters to the circuit
        self.circuit.parameterize({'categorical': outputs['categorical'], 'sum': outputs['sum']}, top_p=top_p)
        return outputs['past_key_values']

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
        all_log_probs = self.circuit.autoregressive_conditionals(
            yy=yy, with_logits=False
        )
        stp_losses = -all_log_probs.mean(dim=1)
        # H dims
        return stp_losses

    def compute_next_token_log_probs(self) -> Tensor:
        next_token_log_probs = self.circuit.univariate_marginal_at_k(
            k=0, with_logits=True
        )
        # BS, V
        return next_token_log_probs

    def compute_all_token_log_probs(self, yy: Tensor) -> Tensor:
        all_token_log_probs = self.circuit.autoregressive_conditionals(
            yy=yy, with_logits=True
        )
        # H, BS, V
        return all_token_log_probs

    def compute_per_token_losses(
        self, yy: Tensor, draft_log_probs: Tensor, teacher_log_probs: Tensor = None
    ) -> dict:
        """Compute per token losses.

        Args:
            yy: shape (H, B, S), the target token indices
            draft_log_probs: shape (H, B, S, V) or (H, B, S), the log probs from the draft model
            teacher_log_probs: shape (H, B, S, V) or (H, B, S), the categorical distributions
                from the teacher model, windowed for easy kl computation.
        """
        if self.compute_kl:
            assert teacher_log_probs is not None, "Expected teacher_log_probs != None"

        losses = dict()
        if self.compute_kl:
            if self.kl_algorithm == "full":
                losses["kl_loss"] = compute_full_kl(
                    draft_log_probs, teacher_log_probs, self.kl_type, valid_mask=compute_valid_mask(yy)
                )
            elif self.kl_algorithm == "binary_approx":
                losses["kl_loss"] = compute_binary_approx_kl(
                    draft_log_probs, teacher_log_probs, self.kl_type, valid_mask=compute_valid_mask(yy)
                )
            else:
                raise ValueError("Unknown kl_algorithm = %s" % self.kl_algorithm)
        if self.compute_ce:
            losses["ce_loss"] = compute_cross_entropy(draft_log_probs, yy)

        return losses

    @torch.no_grad()
    def generate(
        self,
        inputs: Tensor,
        mode: str = 'mtp',
        use_argmax: bool = False,
        use_cache: bool = False,
        attention_mask: Tensor = None,
        past_key_values: Cache = None,
        head_past_key_values: Cache = None,
        position_ids: Tensor = None,
        draft_top_p: float = 1.,
    ) -> dict:
        assert attention_mask is None
        assert position_ids is None

        if use_cache:
            if past_key_values is None:
                # LL: prepare Evabyte KV cache for multi-token prediction
                # LL: the below code is required for the first iteration, i.e., here's why check past_key_values is None here
                past_key_values = EvaStaticCacheForTriton(
                    inputs.shape[0],
                    self.lm.config.num_attention_heads,
                    self.lm.config.window_size + self.circuit.n_token,
                    self.lm.config.hidden_size // self.lm.config.num_attention_heads,
                    self.lm.config.num_hidden_layers,
                    torch.bfloat16,
                    inputs.device,
                )
                position_ids = torch.arange(0, inputs.shape[1], device=inputs.device, dtype=torch.int).unsqueeze(dim=0)
                outputs = self.lm.encoder(
                    inputs,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    multibyte_decoding=False
                )
                past_key_values = outputs['past_key_values']
                past_key_values = self.lm.lm_model._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(past_key_values)
            else:
                # LL: below I am reusing the code for multi token generation in Evabyte
                # LL: the challenge is preparing all the masks and update the multi token KV cache accordingly
                #     to the number of tokens we sample each time
                past_seen_tokens = past_key_values.get_seq_length()
                attn_mask = multi_byte_pred_prepare_attn_mask(self.lm.config, past_seen_tokens, self.circuit.n_token, device=inputs.device)
                position_ids = torch.arange(past_seen_tokens, inputs.shape[1], device=inputs.device, dtype=torch.int).unsqueeze(dim=0)
                outputs = self.lm.encoder(
                    input_ids=inputs[:, past_seen_tokens:],
                    use_cache=True,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    multibyte_decoding=True
                )
                past_key_values = outputs['past_key_values']
                past_key_values = self.lm.lm_model.multi_byte_pred_update_cache(
                    past_key_values,
                    torch.arange(self.circuit.n_token, device=inputs.device, dtype=torch.int).unsqueeze(dim=0),
                    0,
                    self.circuit.n_token,
                )
        else:
            outputs = self.lm.encoder(input_ids=inputs, use_cache=False)

        # Parameterize the circuit
        xx = outputs['last_hidden_state']
        next_head_past_key_values = self._parameterize_circuit(
            xx,
            use_cache=use_cache,
            attention_mask=attention_mask,
            past_key_values=head_past_key_values,
            position_ids=position_ids,
            generate=True,
            top_p=draft_top_p,
        )

        # Update caches for the next iteration
        if use_cache:
            head_past_key_values = next_head_past_key_values

        if mode == "mtp":
            # Sample or argmax the next tokens
            if use_argmax:
                tokens = self.circuit.argmax()
            else:
                tokens = self.circuit.sample(num_samples=1)
        elif mode == "stp":
            next_token_probs = torch.exp(self.compute_next_token_log_probs())
            if use_argmax:
                tokens = torch.argmax(next_token_probs, dim=1)
                tokens = tokens.unsqueeze(dim=1)
            else:
                tokens = torch.multinomial(next_token_probs, num_samples=1)
        else:
            assert False
        return dict(
            tokens=tokens,
            past_key_values=past_key_values,
            head_past_key_values=head_past_key_values
        )

    # TODO: Refactor to bring for-loop into function as per Edoardo's comment
    @torch.no_grad()
    def self_speculative_generate(
        self,
        inputs: Tensor,
        use_cache: bool = False,
        use_argmax: bool = False,
        attention_mask: Tensor = None,
        draft_past_key_values: Cache = None,
        verifier_past_key_values: Cache = None,
        head_past_key_values: Cache = None,
        position_ids: Tensor = None,
        past_num_tokens: int = None,
        draft_top_p: float = 1.,
        target_top_p: float = 1.,
    ) -> dict:
        if len(inputs.shape) != 2 or inputs.shape[0] != 1:
            raise NotImplementedError(
                "Multi-batch self-speculative decoding not implemented yet"
            )
            # inputs: (B, S), with B = 1 and also possibly S = 1

        # Compute the embeddings
        if use_cache:
            if draft_past_key_values is None:
                # LL: prepare Evabyte KV cache for multi-token prediction
                # LL: the below code is required for the first iteration, i.e., here's why check draft_past_key_values is None here
                draft_past_key_values = EvaStaticCacheForTriton(
                    inputs.shape[0],
                    self.lm.config.num_attention_heads,
                    self.lm.config.window_size + self.circuit.n_token,
                    self.lm.config.hidden_size // self.lm.config.num_attention_heads,
                    self.lm.config.num_hidden_layers,
                    torch.bfloat16,
                    inputs.device,
                )
                verifier_past_key_values = EvaStaticCacheForTriton(
                    inputs.shape[0],
                    self.lm.config.num_attention_heads,
                    self.lm.config.window_size + self.circuit.n_token,
                    self.lm.config.hidden_size // self.lm.config.num_attention_heads,
                    self.lm.config.num_hidden_layers,
                    torch.bfloat16,
                    inputs.device,
                )
                position_ids = torch.arange(0, inputs.shape[1], device=inputs.device, dtype=torch.int).unsqueeze(dim=0)
                outputs = self.lm.encoder(
                    inputs,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=draft_past_key_values,
                    position_ids=position_ids,
                    multibyte_decoding=False
                )
                draft_past_key_values = outputs['past_key_values']
                with self.lm.disable_adapter_if_any():
                    verifier_outputs = self.lm.encoder(
                        inputs[:, :-1],
                        attention_mask=None,
                        use_cache=True,
                        past_key_values=verifier_past_key_values,
                        position_ids=position_ids[:, :-1],
                        multibyte_decoding=False
                    )
                    verifier_past_key_values = verifier_outputs['past_key_values']
                draft_past_key_values = self.lm.lm_model._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(draft_past_key_values)
                verifier_past_key_values = self.lm.lm_model._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(verifier_past_key_values)
            else:
                # LL: below I am reusing the code for multi token generation in Evabyte
                # LL: the challenge is preparing all the masks and update the multi token KV cache accordingly
                #     to the number of tokens we sample each time
                past_seen_tokens = draft_past_key_values.get_seq_length()
                attn_mask = multi_byte_pred_prepare_attn_mask(self.lm.config, past_seen_tokens, past_num_tokens, device=inputs.device)
                position_ids = torch.arange(past_seen_tokens, inputs.shape[1], device=inputs.device, dtype=torch.int).unsqueeze(dim=0)
                outputs = self.lm.encoder(
                    input_ids=inputs[:, past_seen_tokens:],
                    use_cache=True,
                    attention_mask=attn_mask,
                    past_key_values=draft_past_key_values,
                    position_ids=position_ids,
                    multibyte_decoding=True
                )
        else:
            outputs = self.lm.encoder(input_ids=inputs, use_cache=False)
        # Parameterize the circuit
        xx = outputs['last_hidden_state']
        next_head_past_key_values = self._parameterize_circuit(
            xx,
            use_cache=use_cache,
            attention_mask=attention_mask,
            past_key_values=head_past_key_values,
            position_ids=position_ids,
            generate=True,
            top_p=draft_top_p
        )

        # Update caches for the next iteration
        if use_cache:
            head_past_key_values = next_head_past_key_values

        # Sample the next H tokens
        # tokens: (B=1, H)
        if use_argmax:
            tokens = self.circuit.argmax()
        else:
            tokens = self.circuit.sample(num_samples=1)

        # Concatenate the tokens with the current sequence,
        # which gives the candidate next sequence
        # gen_seq: (B, S + H)
        gen_seq = torch.cat([inputs, tokens], dim=1)

        # Compute the next-token probabilities in parallel
        with self.lm.disable_adapter_if_any():
            if use_cache:
                past_seen_tokens = verifier_past_key_values.get_seq_length()
                verifier_attn_mask = multi_byte_pred_prepare_attn_mask(self.lm.config, past_seen_tokens, gen_seq.shape[1] - past_seen_tokens, device=gen_seq.device)
                position_ids = torch.arange(past_seen_tokens, gen_seq.shape[1], device=gen_seq.device, dtype=torch.int).unsqueeze(dim=0)
                outputs = self.lm.encoder(
                    input_ids=gen_seq[:, past_seen_tokens:],
                    use_cache=True,
                    attention_mask=verifier_attn_mask,
                    past_key_values=verifier_past_key_values,
                    position_ids=position_ids,
                    multibyte_decoding=True
                )
                verifier_past_key_values = outputs['past_key_values']
            else:
                outputs = self.lm.encoder(input_ids=gen_seq, use_cache=False)
            # zz: (B, S + H, D) -> (B, H + 1, D)
            zz = outputs['last_hidden_state']
            assert not use_cache or zz.shape[1] == self.circuit.n_token + 1, zz.shape
            zz = zz[:, -tokens.shape[1] - 1 :]
            # logits: (B, H + 1, V)
            logits = self.lm.head_logits(zz)

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
        if use_argmax:
            log_marginal_probs = torch.zeros(size=(tokens.shape[0], tokens.shape[1], 1), dtype=logits.dtype, device=logits.device)
        else:
            log_marginal_probs = self.circuit.marginalizer(
                tokens.expand(size=(tokens.shape[1], -1)),
                integrate_vars=self.circuit._autoregressive_mar_mask,
            )
            log_marginal_probs = log_marginal_probs.squeeze(dim=1).unsqueeze(dim=0)
        #
        # Sample H uniform noise values in [0,1), and take their log
        # log_noise: (B=1, H)
        log_noise = torch.log(torch.rand(size=(tokens.shape[0], tokens.shape[1])))
        #
        # Compute target model token probabilities
        # lm_next_token_log_probs: (B, H + 1, V)
        lm_next_token_log_probs = torch.log_softmax(logits, dim=2)
        if target_top_p < 1.0:
            lm_next_token_log_probs = truncate_logprobs_top_p(lm_next_token_log_probs, p=target_top_p)
        # lm_next_token_log_probs_cpu: (B, H, 1)
        lm_next_token_log_probs_cpu = torch.gather(lm_next_token_log_probs[:, :-1], dim=2, index=tokens.unsqueeze(dim=-1)).cpu()
        # log_marginal_probs_cpu: (B, H, 1)
        log_marginal_probs_cpu = log_marginal_probs.cpu()  # Move to the CPU as it will be slightly faster due to non-vectorizable code below
        #
        # Compute the number of tokens to accept
        num_accepted_tokens = 0
        for j in range(tokens.shape[1]):
            # Check whether noise > ratio of conditional univariate probabilities,
            # i.e., we should stop accepting tokens
            # In the log space, this becomes log noise > difference of some log probabilities,
            # which avoids many floating point divisions and is more numerically stable
            # lm_jth_token_log_prob: (B, 1)
            lm_jth_token_log_prob = lm_next_token_log_probs_cpu[:, j]
            # Compute log conditional probabilities, conditioned on the context
            if j == 0:
                # q(x_{t+1}\mid x_{\leq t})
                # mtp_jth_token_log_prob: (B, 1)
                mtp_jth_token_log_prob = log_marginal_probs_cpu[:, 0]
            else:
                # q(x_{t+j}\mid x_{\leq t}, x_{t+1}, ..., x_{t+j-1}) = \
                #     q(x_{t+1}, ..., x_{t+j}\mid x_{\leq t}) / q(x_{t+1}, ..., x_{t+j-1}\mid x_{\leq t})
                # mtp_jth_token_log_prob: (B, 1)
                mtp_jth_token_log_prob = (
                    log_marginal_probs_cpu[:, j] - log_marginal_probs_cpu[:, j - 1]
                )
            # Check noise > \
            #     (p(x_{t+j}\mid x_{\leq t}, x_{t+1}, ..., x_{t+j-1}) / q(x_{t+j}\mid x_{\leq t}, x_{t+1}, ..., x_{t+j-1}))
            if log_noise[:, j] > (lm_jth_token_log_prob - mtp_jth_token_log_prob):
                break
            num_accepted_tokens += 1

        if num_accepted_tokens == tokens.shape[1]:
            # We are so lucky! We accept all the H tokens
            # Let's index the probabilities to sample the H+1-th one
            lm_last_probs = torch.exp(lm_next_token_log_probs[:, -1])
            # Sample the last token
            last_token = torch.multinomial(lm_last_probs, num_samples=1)
        else:  # num_accepted_tokens < tokens.shape[1]
            # We accepted H' < H tokens
            # Let's adjust the probabilities to sample the H'+1-th one
            # lm_last_probs: (B, V)
            lm_last_probs = torch.exp(lm_next_token_log_probs[:, num_accepted_tokens])
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
                    integrate_vars=self.circuit._autoregressive_mar_mask[
                        num_accepted_tokens
                    ],
                )
            # mtp_jp1th_token_log_probs: (B * V, 1, 1) -> (B, V)
            mtp_jp1th_token_log_probs = mtp_jp1th_token_log_probs.view(
                tokens.shape[0], self.vocab_size
            )
            # mtp_last_probs: (B, V)
            if num_accepted_tokens == 0:
                mtp_last_probs = torch.exp(mtp_jp1th_token_log_probs)
            else:
                mtp_last_probs = torch.exp(
                    mtp_jp1th_token_log_probs
                    - log_marginal_probs[:, num_accepted_tokens - 1]
                )
            adj_last_probs = torch.clamp_min(lm_last_probs - mtp_last_probs, min=1e-15)
            adj_last_probs = adj_last_probs / torch.sum(
                adj_last_probs, dim=1, keepdim=True
            )
            # Sample the last token
            last_token = torch.multinomial(adj_last_probs, num_samples=1)

        # Retrieve the accepted tokens, plus the last one
        tokens = torch.cat([tokens[:, :num_accepted_tokens], last_token], dim=1)
        num_generated_tokens = tokens.shape[1]  # num_accepted_tokens + 1

        # Update the KV cache, based on the number of tokens we have sampled previously
        if use_cache:
            if past_num_tokens is not None:
                draft_past_key_values = self.lm.lm_model.multi_byte_pred_update_cache(
                    draft_past_key_values,
                    torch.arange(past_num_tokens, device=gen_seq.device, dtype=torch.int).unsqueeze(dim=0),
                    0,
                    past_num_tokens,
                )
            verifier_past_key_values = self.lm.lm_model.multi_byte_pred_update_cache(
                verifier_past_key_values,
                torch.arange(self.circuit.n_token + 1, device=gen_seq.device, dtype=torch.int).unsqueeze(dim=0),
                0,
                num_generated_tokens,
            )

        return dict(
            tokens=tokens,
            draft_past_key_values=draft_past_key_values,
            verifier_past_key_values=verifier_past_key_values,
            head_past_key_values=head_past_key_values,
            past_num_tokens=num_generated_tokens
        )
