import torch

from copy import deepcopy
from torch import Tensor
from peft import PeftModelForCausalLM

from mtp.models.cache import KVCacheWrapper
from mtp.utils.model_types import get_model_type


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


class LoRASplitLM(torch.nn.Module):
    """
    LM split into three components for efficient speculative decoding:
    1. Shared encoder (first N-K layers)
    2. Draft encoder (last K layers with LoRA)
    3. Verifier encoder (last K layers without LoRA)

    If no LoRA layers are present, this class acts like a single encoder:
    1. Shared encoder (N layers)

    Note: we enforce that the verifier is always one token behind the draft.
    """

    def __init__(self, shared_encoder, draft_encoder, verifier_encoder, lm_head):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.draft_encoder = draft_encoder
        self.verifier_encoder = verifier_encoder
        self.lm_head = lm_head

        # Create KVCacheWrapper instances
        self.shared_kv_cache = KVCacheWrapper.for_model(shared_encoder)
        if draft_encoder is not None:
            self.draft_kv_cache = KVCacheWrapper.for_model(draft_encoder)
            self.verifier_kv_cache = KVCacheWrapper.for_model(verifier_encoder)
        else:
            self.draft_kv_cache = None
            self.verifier_kv_cache = None

        self.arch_specific_prefill_kwargs = dict()
        self.arch_specific_inference_kwargs = dict()
        if self.model_type == "evabyte":
            self.arch_specific_prefill_kwargs = {"multibyte_decoding": False}
            self.arch_specific_inference_kwargs = {"multibyte_decoding": True}

    @classmethod
    def from_lm(cls, lm):
        """
        Create SplitLM from an existing LM instance. We detect the number of
        LoRA layers (if any) and split the LM into parts.

        Args:
            lm: The original LM instance
        """

        model_type = get_model_type(lm)

        if isinstance(lm, PeftModelForCausalLM):
            lm_head = deepcopy(lm.model.lm_head)
            del lm.model.lm_head

            num_adapter_layers = count_adapter_layers(lm.model.model)
            num_total_layers = len(lm.model.model.layers)
            # Default: split where LoRA layers start
            split_layer_idx = num_total_layers - num_adapter_layers
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
            # Note: This works both for EvaByte and TPULlamaModel
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
            verifier_encoder.config.num_hidden_layers = len(
                verifier_encoder.model.layers
            )

            del all_layers
        elif model_type in ("llama", "evabyte"):
            shared_encoder = lm
            draft_encoder = None
            verifier_encoder = None
            print("Found no adapter layers..")

            lm_head = deepcopy(lm.lm_head)
            del lm.lm_head
        else:
            raise ValueError(f"Unexpected LM of type: {lm.__class__}")

        return cls(shared_encoder, draft_encoder, verifier_encoder, lm_head)

    @property
    def has_adapter(self):
        return self.draft_encoder is not None

    @property
    def head(self):
        return getattr(self, "lm_head")

    @property
    def model_type(self):
        """Detect model type from shared_encoder."""
        return get_model_type(self.shared_encoder)

    @property
    def shared_seen_tokens(self):
        """Access shared cache's seen tokens counter."""
        return self.shared_kv_cache.seen_tokens

    @property
    def draft_seen_tokens(self):
        """Access draft cache's seen tokens counter."""
        return self.draft_kv_cache.seen_tokens if self.draft_kv_cache else self.shared_kv_cache.seen_tokens

    @property
    def verifier_seen_tokens(self):
        """Access verifier cache's seen tokens counter."""
        return self.verifier_kv_cache.seen_tokens if self.verifier_kv_cache else self.shared_kv_cache.seen_tokens - 1

    @property
    def shared_encoder_cache(self):
        """Access shared cache's underlying cache object."""
        return self.shared_kv_cache.cache

    @property
    def draft_encoder_cache(self):
        """Access draft cache's underlying cache object."""
        return self.draft_kv_cache.cache if self.draft_kv_cache else None

    @property
    def verifier_encoder_cache(self):
        """Access verifier cache's underlying cache object."""
        return self.verifier_kv_cache.cache if self.verifier_kv_cache else None

    @torch.no_grad()
    def prefill(self, input_ids, circuit_n_token):

        self.reset_caches()
        position_ids = self.shared_kv_cache.get_position_ids(input_ids)

        # ============ Prefill: Shared Encoder ========================
        shared_outputs = self.shared_encoder.model(
            input_ids=input_ids,
            use_cache=True,
            position_ids=position_ids,
            past_key_values=self.shared_encoder_cache,
            **self.arch_specific_prefill_kwargs,
        )
        shared_last_hidden_state = shared_outputs["last_hidden_state"]
        shared_past_key_values = shared_outputs["past_key_values"]

        # Update kv cache using wrapper's prefill_update
        shared_past_key_values = self.shared_kv_cache.prefill_update(shared_past_key_values)

        if self.has_adapter:
            # ============ Prefill: Draft Encoder ========================
            draft_outputs = self.draft_encoder.model(
                input_ids=input_ids,
                inputs_embeds=shared_last_hidden_state,
                use_cache=True,
                past_key_values=self.draft_encoder_cache,
                position_ids=position_ids,
                **self.arch_specific_prefill_kwargs,
            )
            draft_last_hidden_state = draft_outputs["last_hidden_state"]
            draft_past_key_values = draft_outputs["past_key_values"]

            # Update kv cache using wrapper's prefill_update
            draft_past_key_values = self.draft_kv_cache.prefill_update(draft_past_key_values)
            # ============ Prefill: Verifier Encoder ========================
            # NOTE: Verifier must stay one step behind Draft
            verifier_outputs = self.verifier_encoder.model(
                input_ids=input_ids[:, :-1],
                inputs_embeds=shared_last_hidden_state[:, :-1],
                use_cache=True,
                past_key_values=self.verifier_encoder_cache,
                position_ids=position_ids[:, :-1],
                **self.arch_specific_prefill_kwargs,
            )
            verifier_last_hidden_state = verifier_outputs["last_hidden_state"]
            verifier_past_key_values = verifier_outputs["past_key_values"]

            # Update kv cache using wrapper's prefill_update
            verifier_past_key_values = self.verifier_kv_cache.prefill_update(verifier_past_key_values)
        else:
            draft_last_hidden_state = shared_last_hidden_state
            draft_past_key_values = shared_past_key_values
            verifier_last_hidden_state = None
            verifier_past_key_values = None

        results = dict(
            shared_last_hidden_state=shared_last_hidden_state,
            draft_last_hidden_state=draft_last_hidden_state,
            verifier_last_hidden_state=verifier_last_hidden_state,
            shared_past_key_values=shared_past_key_values,
            draft_past_key_values=draft_past_key_values,
            verifier_past_key_values=verifier_past_key_values,
        )
        return results

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
    def draft(self, input_ids, shared_last_hidden_state, use_cache=True):

        # NOTE: shared_last_hidden_state needs to encode all history
        if not use_cache:
            raise NotImplementedError("use_cache=False not supported")

        if use_cache:
            if self.has_adapter:
                assert self.shared_encoder_cache is not None, "Prefilling required"
                assert self.draft_encoder_cache is not None, "Prefilling required"
                draft_kvs = self.draft_kv_cache.get_encode_kwargs(
                    input_ids=input_ids,
                )
            shared_kvs = self.shared_kv_cache.get_encoder_kwargs(
                input_ids=input_ids,
            )
            # For Llama we need to pass in the whole history
            if self.shared_kv_cache.model_type == "llama":
                shared_kvs["past_input_ids"] = input_ids

        # If our current hidden state is not up to date
        if shared_last_hidden_state.shape[1] != input_ids.shape[1]:
            # Run shared_encoder
            shared_outputs = self.shared_encoder.model(
                input_ids=input_ids[:, self.shared_seen_tokens :],
                use_cache=use_cache,
                **shared_kvs,
                **self.arch_specific_inference_kwargs,
            )
            shared_last_hidden_state = torch.cat(
                [shared_last_hidden_state, shared_outputs["last_hidden_state"]], axis=1
            )
            shared_past_key_values = shared_outputs["past_key_values"]
        else:
            shared_past_key_values = self.shared_encoder_cache

        if self.has_adapter:
            # Run draft_encoder
            draft_outputs = self.draft_encoder.model(
                input_ids=input_ids[:, self.draft_seen_tokens :],
                inputs_embeds=shared_last_hidden_state[:, self.draft_seen_tokens :],
                use_cache=use_cache,
                **draft_kvs,
                **self.arch_specific_inference_kwargs,
            )
            draft_last_hidden_state = draft_outputs["last_hidden_state"]
            draft_past_key_values = draft_outputs["past_key_values"]
        else:
            # In this case shared = draft
            draft_last_hidden_state = shared_last_hidden_state
            draft_past_key_values = shared_past_key_values

        results = dict(
            shared_last_hidden_state=shared_last_hidden_state,
            draft_last_hidden_state=draft_last_hidden_state,
            shared_past_key_values=shared_past_key_values,
            draft_past_key_values=draft_past_key_values,
        )
        return results

    @torch.no_grad()
    def verify(self, input_ids, shared_last_hidden_state, use_cache=True):

        if not use_cache:
            raise NotImplementedError("use_cache=False not supported")

        if use_cache:
            if self.has_adapter:
                assert self.shared_encoder_cache is not None, "Prefilling required"
                assert self.verifier_encoder_cache is not None, "Prefilling required"
                verifier_kvs = self.verifier_kv_cache.get_encode_kwargs(
                    input_ids=input_ids,
                )
            shared_kvs = self.shared_kv_cache.get_encoder_kwargs(
                input_ids=input_ids,
            )
            # For Llama we need to pass in the whole history
            if self.shared_kv_cache.model_type == "llama":
                shared_kvs["past_input_ids"] = input_ids

        # If our current hidden state is not up to date
        if shared_last_hidden_state.shape[1] != input_ids.shape[1]:
            # Run shared_encoder
            shared_outputs = self.shared_encoder.model(
                input_ids=input_ids[:, self.shared_seen_tokens :],
                use_cache=use_cache,
                **shared_kvs,
                **self.arch_specific_inference_kwargs,
            )
            shared_last_hidden_state = torch.cat(
                [shared_last_hidden_state, shared_outputs["last_hidden_state"]], axis=1
            )
            shared_past_key_values = shared_outputs["past_key_values"]
        else:
            shared_past_key_values = self.shared_encoder_cache

        if self.has_adapter:
            # Run verifier_encoder
            verifier_outputs = self.verifier_encoder.model(
                input_ids=input_ids[:, self.verifier_seen_tokens :],
                inputs_embeds=shared_last_hidden_state[:, self.verifier_seen_tokens :],
                use_cache=use_cache,
                **verifier_kvs,
                **self.arch_specific_inference_kwargs,
            )
            verifier_last_hidden_state = verifier_outputs["last_hidden_state"]
            verifier_past_key_values = verifier_outputs["past_key_values"]
        else:
            # In this case shared = verifier
            verifier_last_hidden_state = shared_last_hidden_state[
                :, self.verifier_seen_tokens :
            ]
            verifier_past_key_values = shared_past_key_values

        results = dict(
            shared_last_hidden_state=shared_last_hidden_state,
            verifier_last_hidden_state=verifier_last_hidden_state,
            shared_past_key_values=shared_past_key_values,
            verifier_past_key_values=verifier_past_key_values,
        )
        return results

    def reset_caches(self):
        self.shared_kv_cache.reset()
        if self.has_adapter:
            self.draft_kv_cache.reset()
            self.verifier_kv_cache.reset()

    def update_shared_cache(self, past_key_values, num_candidates, num_valid):
        self.shared_kv_cache.update(past_key_values, num_candidates, num_valid)

    def update_draft_cache(self, past_key_values, num_candidates, num_valid):
        if self.has_adapter:
            self.draft_kv_cache.update(past_key_values, num_candidates, num_valid)

    def update_verifier_cache(self, past_key_values, num_candidates, num_valid):
        if self.has_adapter:
            self.verifier_kv_cache.update(past_key_values, num_candidates, num_valid)
