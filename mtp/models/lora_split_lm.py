import torch
from copy import deepcopy
from torch import Tensor

from mtp.models.evabyte.eva_cache import EvaStaticCacheForTriton
from mtp.models.evabyte.multibyte_decoding_evabyte import (
    multi_byte_pred_prepare_attn_mask,
)


def count_adapter_layers(lm):
    """Count transformer layers that have LoRA adapters"""

    count = 0
    for layer in lm.layers:
        # Check if this layer has any LoRA modules
        for module in layer.modules():
            if hasattr(module, "lora_A"):
                count += 1
                break  # Found adapter in this layer, move to next layer

    return count


def get_position_ids(inputs, cache, target=None):
    assert target in {"draft", "verifier", None}
    # Construct input_ids
    if cache is None:
        position_ids = torch.arange(
            0, inputs.shape[1], device=inputs.device, dtype=torch.int
        ).unsqueeze(dim=0)
    else:
        assert target is not None
        past_seen_tokens = cache.get_seq_length()
        start = past_seen_tokens if target == "draft" else past_seen_tokens - 1
        position_ids = torch.arange(
            start,
            inputs.shape[1],
            device=inputs.device,
            dtype=torch.int,
        ).unsqueeze(dim=0)
    return position_ids


class LoRASplitLM(torch.nn.Module):
    """
    LM split into three components for efficient speculative decoding:
    1. Shared encoder (first N-K layers)
    2. Draft encoder (last K layers with LoRA)
    3. Verifier encoder (last K layers without LoRA)
    """

    def __init__(
        self, shared_encoder, draft_encoder, verifier_encoder, lm_head, split_layer_idx
    ):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.draft_encoder = draft_encoder
        self.verifier_encoder = verifier_encoder
        self.lm_head = lm_head
        self.split_layer_idx = split_layer_idx

        self.reset_caches()

    @classmethod
    def from_lm(cls, lm, split_layer_idx: int = None):
        """
        Create SplitLM from an existing LM instance

        Args:
            lm: The original LM instance
            split_layer_idx: Layer index to split at. If None, uses num_layers - num_adapter_layers
        """

        lm_head = deepcopy(lm.model.lm_head)
        del lm.model.lm_head

        num_adapter_layers = count_adapter_layers(lm.model.model)
        num_total_layers = len(lm.model.model.layers)
        if num_adapter_layers == 0:
            raise ValueError("LM instance must have adapters to create SplitLM")

        if split_layer_idx is None:
            # Default: split where LoRA layers start
            split_layer_idx = num_total_layers - num_adapter_layers
        else:
            assert 0 < split_layer_idx < num_total_layers
        print(f"Found {num_adapter_layers} adapter layers..")

        all_layers = deepcopy(lm.model.model.layers)

        # ===================== Shared Encoder ================================
        shared_encoder = deepcopy(lm)
        shared_encoder.model.model.layers = None
        draft_encoder = deepcopy(shared_encoder)
        verifier_encoder = deepcopy(shared_encoder)

        shared_encoder.model.model.layers = deepcopy(all_layers[:split_layer_idx])
        # NOTE: ! Important !
        # Monkey-patch norm since it would be applied to the last activation giving wrong result
        shared_encoder.model.model.norm = torch.nn.Identity()
        shared_encoder = shared_encoder.unload()
        shared_encoder.config.num_hidden_layers = len(shared_encoder.model.layers)

        # ===================== Draft LoRA Encoder ============================
        draft_encoder.model.model.layers = deepcopy(all_layers[split_layer_idx:])
        # Correct the layer idx to match the new cache
        for idx, layer in enumerate(draft_encoder.model.model.layers):
            layer.self_attn.layer_idx = idx
        # Merge LoRA weights
        draft_encoder = draft_encoder.merge_and_unload()
        draft_encoder.config.num_hidden_layers = len(draft_encoder.model.layers)

        # ================== Verifier no LoRA Encoder =========================
        verifier_encoder.model.model.layers = deepcopy(all_layers[split_layer_idx:])
        # Correct the layer idx to match the new cache
        for idx, layer in enumerate(verifier_encoder.model.model.layers):
            layer.self_attn.layer_idx = idx
        verifier_encoder = verifier_encoder.unload()
        verifier_encoder.config.num_hidden_layers = len(verifier_encoder.model.layers)

        del all_layers

        return cls(shared_encoder, draft_encoder, verifier_encoder, lm_head, split_layer_idx)

    @property
    def has_adapter(self):
        return True

    @torch.no_grad()
    def prefill(self, input_ids, circuit_n_token):
        use_cache = True

        # Initialises the cache
        self.init_caches(circuit_n_token, batch_size=input_ids.shape[0])

        position_ids = get_position_ids(input_ids, None)

        # NOTE: No need to update cache as multibyte_decoding=False
        # appends to the cache in the forward pass
        # ============ Prefill: Shared Encoder ========================
        shared_outputs = self.shared_encoder.model(
            input_ids=input_ids,
            use_cache=use_cache,
            position_ids=position_ids,
            past_key_values=self.shared_encoder_cache,
            multibyte_decoding=False,
        )
        shared_hidden_state = shared_outputs["last_hidden_state"]

        # ============ Prefill: Draft Encoder ========================
        draft_outputs = self.draft_encoder.model(
            input_ids=input_ids,
            inputs_embeds=shared_hidden_state,
            use_cache=use_cache,
            past_key_values=self.draft_encoder_cache,
            position_ids=position_ids,
            multibyte_decoding=False,
        )
        draft_hidden_state = draft_outputs["last_hidden_state"]
        # ============ Prefill: Verifier Encoder ========================
        verifier_outputs = self.verifier_encoder.model(
            input_ids=input_ids,
            inputs_embeds=shared_hidden_state,
            use_cache=use_cache,
            past_key_values=self.verifier_encoder_cache,
            position_ids=position_ids,
            multibyte_decoding=False,
        )
        verifier_hidden_state = verifier_outputs["last_hidden_state"]

        # Update kv cache
        shared_outputs["past_key_values"] = (
            self.shared_encoder._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(
                shared_outputs["past_key_values"]
            )
        )
        draft_outputs["past_key_values"] = (
            self.draft_encoder._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(
                draft_outputs["past_key_values"]
            )
        )
        verifier_outputs["past_key_values"] = (
            self.verifier_encoder._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(
                verifier_outputs["past_key_values"]
            )
        )

        self.set_caches(
            shared_cache=shared_outputs["past_key_values"],
            draft_cache=draft_outputs["past_key_values"],
            verifier_cache=verifier_outputs["past_key_values"],
        )

        results = dict(
            shared_last_hidden_state=shared_hidden_state,
            draft_last_hidden_state=draft_hidden_state,
            verifier_last_hidden_state=verifier_hidden_state,
            shared_past_key_values=shared_outputs["past_key_values"],
            draft_past_key_values=shared_outputs["past_key_values"],
            verifier_past_key_values=shared_outputs["past_key_values"],
        )
        return results

    @property
    def head(self):
        return getattr(self, "lm_head")

    def head_logits(self, xx: Tensor) -> Tensor:
        # Compute the logits with the head
        logits = self.head(xx)

        # Checker whether the LM is multi-token model
        # In that case, return the logits of the first part of the head only
        if (
            hasattr(self.shared_encoder.config, "num_pred_heads")
            and self.shared_encoder.config.num_pred_heads > 1
        ):
            num_pred_heads, vocab_size = (
                self.shared_encoder.config.num_pred_heads,
                self.shared_encoder.config.vocab_size,
            )
            assert logits.shape == (
                logits.shape[0],
                logits.shape[1],
                num_pred_heads * vocab_size,
            )
            logits = logits.view(
                logits.shape[0], logits.shape[1], num_pred_heads, vocab_size
            )
            logits = logits[:, :, 0]  # (B, S, V)

        # Cast to float32
        return logits.float()

    @torch.no_grad()
    def draft(self, input_ids, shared_hidden_state=None, use_cache=True):

        if not use_cache:
            raise ValueError("use_cache=False not fully implemented")

        if use_cache:
            # NOTE: shared is usually ahead of draft because we accepted X tokens
            # so we need to compute separate position ids and attn weights
            shared_position_ids = get_position_ids(input_ids, self.shared_encoder_cache, "draft")
            draft_position_ids = get_position_ids(input_ids, self.draft_encoder_cache, "draft")
            assert self.shared_encoder_cache is not None, "Prefilling required"
            assert self.draft_encoder_cache is not None, "Prefilling required"
            shared_past_key_values = self.shared_encoder_cache
            draft_past_key_values = self.draft_encoder_cache
            shared_past_seen_tokens = shared_past_key_values.get_seq_length()
            draft_past_seen_tokens = draft_past_key_values.get_seq_length()
            shared_attn_mask = multi_byte_pred_prepare_attn_mask(
                self.shared_encoder.config,
                shared_past_seen_tokens,
                input_ids.shape[1] - shared_past_seen_tokens,
                device=input_ids.device,
            )
            draft_attn_mask = multi_byte_pred_prepare_attn_mask(
                self.draft_encoder.config,
                draft_past_seen_tokens,
                input_ids.shape[1] - draft_past_seen_tokens,
                device=input_ids.device,
            )
        else:
            shared_position_ids = None
            draft_position_ids = None
            shared_past_key_values = None
            draft_past_key_values = None
            shared_attn_mask = None
            draft_attn_mask = None

        if shared_hidden_state is None:
            # Run shared_encoder
            shared_outputs = self.shared_encoder.model(
                input_ids=input_ids[:, shared_past_seen_tokens:],
                use_cache=use_cache,
                attention_mask=shared_attn_mask,
                position_ids=shared_position_ids,
                past_key_values=shared_past_key_values,
                multibyte_decoding=use_cache,
            )
            shared_hidden_state = shared_outputs["last_hidden_state"]
            shared_past_key_values = shared_outputs["past_key_values"]

        # Run draft_encoder
        draft_outputs = self.draft_encoder.model(
            input_ids=input_ids[:, draft_past_seen_tokens:],
            inputs_embeds=shared_hidden_state,
            use_cache=use_cache,
            attention_mask=draft_attn_mask,
            past_key_values=draft_past_key_values,
            position_ids=draft_position_ids,
            multibyte_decoding=use_cache,
        )
        draft_past_key_values = draft_outputs["past_key_values"]

        results = dict(
            shared_last_hidden_state=shared_outputs["last_hidden_state"],
            draft_last_hidden_state=draft_outputs["last_hidden_state"],
            shared_past_key_values=shared_past_key_values,
            draft_past_key_values=draft_past_key_values,
        )
        return results

    @torch.no_grad()
    def verify(self, input_ids, shared_hidden_state=None, use_cache=True):

        if not use_cache:
            raise ValueError("use_cache=False not fully implemented")

        if use_cache:
            assert self.shared_encoder_cache is not None, "Prefilling required"
            assert self.verifier_encoder_cache is not None, "Prefilling required"
            shared_past_key_values = self.shared_encoder_cache
            verifier_past_key_values = self.verifier_encoder_cache
            past_seen_tokens = self.shared_encoder_cache.get_seq_length()
            assert (
                past_seen_tokens == self.verifier_encoder_cache.get_seq_length()
            ), "verifier and shared cache out of sync"
            attn_mask = multi_byte_pred_prepare_attn_mask(
                self.shared_encoder.config,
                past_seen_tokens,
                input_ids.shape[1] - past_seen_tokens + 1,
                device=input_ids.device,
            )
            position_ids = get_position_ids(input_ids, self.shared_encoder_cache, "verifier")
        else:
            position_ids = None
            shared_past_key_values = None
            verifier_past_key_values = None

        if shared_hidden_state is None:
            # Run shared_encoder
            # TODO: Below should be input_ids[:, past_seen_tokens - 1 :]
            shared_outputs = self.shared_encoder.model(
                input_ids=input_ids[:, past_seen_tokens - 1 :],
                use_cache=use_cache,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=shared_past_key_values,
                multibyte_decoding=use_cache,
            )
            shared_hidden_state = shared_outputs["last_hidden_state"]
            shared_past_key_values = shared_outputs["past_key_values"]

        # Run verifier_encoder
        verifier_outputs = self.verifier_encoder.model(
            input_ids=input_ids[:, past_seen_tokens - 1 :],
            inputs_embeds=shared_hidden_state,
            use_cache=use_cache,
            attention_mask=attn_mask,
            past_key_values=verifier_past_key_values,
            position_ids=position_ids,
            multibyte_decoding=use_cache,
        )
        verifier_past_key_values = verifier_outputs["past_key_values"]

        results = dict(
            verifier_last_hidden_state=verifier_outputs["last_hidden_state"],
            shared_last_hidden_state=shared_outputs["last_hidden_state"],
            verifier_past_key_values=verifier_past_key_values,
            shared_past_key_values=shared_past_key_values,
        )
        return results

    def init_caches(self, num_tokens_speculate, batch_size=1):
        self.shared_encoder_cache = EvaStaticCacheForTriton(
            batch_size,
            self.shared_encoder.config.num_attention_heads,
            self.shared_encoder.config.window_size + num_tokens_speculate,
            self.shared_encoder.config.hidden_size
            // self.shared_encoder.config.num_attention_heads,
            self.shared_encoder.config.num_hidden_layers,
            torch.bfloat16,
            self.shared_encoder.device,
        )
        self.draft_encoder_cache = EvaStaticCacheForTriton(
            batch_size,
            self.draft_encoder.config.num_attention_heads,
            self.draft_encoder.config.window_size + num_tokens_speculate,
            self.draft_encoder.config.hidden_size
            // self.draft_encoder.config.num_attention_heads,
            self.draft_encoder.config.num_hidden_layers,
            torch.bfloat16,
            self.draft_encoder.device,
        )
        self.verifier_encoder_cache = EvaStaticCacheForTriton(
            batch_size,
            self.verifier_encoder.config.num_attention_heads,
            self.verifier_encoder.config.window_size + num_tokens_speculate,
            self.verifier_encoder.config.hidden_size
            // self.verifier_encoder.config.num_attention_heads,
            self.verifier_encoder.config.num_hidden_layers,
            torch.bfloat16,
            self.verifier_encoder.device,
        )

    def set_caches(self, shared_cache=None, draft_cache=None, verifier_cache=None):
        if shared_cache is not None:
            self.shared_encoder_cache = shared_cache
        if draft_cache is not None:
            self.draft_encoder_cache = draft_cache
        if verifier_cache is not None:
            self.verifier_encoder_cache = verifier_cache

    def reset_caches(self):
        self.shared_encoder_cache = None
        self.draft_encoder_cache = None
        self.verifier_encoder_cache = None
