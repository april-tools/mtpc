import torch

from copy import deepcopy
from torch import Tensor
from peft import PeftModelForCausalLM
from transformers.cache_utils import DynamicCache

from mtp.models.evabyte.eva_cache import EvaStaticCacheForTriton
from mtp.models.evabyte.training_utils import get_model_class_name
from mtp.models.evabyte.multibyte_decoding_evabyte import (
    multi_byte_pred_prepare_attn_mask,
)


def prepare_encode_kwargs(input_ids, cache, encoder, num_past_seen_tokens, model_type):

    # Produce attention, position ids and past key values
    if model_type == "evabyte":
        attn_mask = multi_byte_pred_prepare_attn_mask(
            encoder.config,
            num_past_seen_tokens,
            input_ids.shape[1] - num_past_seen_tokens,
            device=input_ids.device,
        )
    else:
        attn_mask = None
    position_ids = get_position_ids(input_ids, num_past_seen_tokens)
    result = {
        "attention_mask": attn_mask,
        "past_key_values": cache,
        "position_ids": position_ids,
    }
    return result


def get_model_type(model):
    class_name = get_model_class_name(model)

    if class_name == "EvaByteForCausalLM":
        return "evabyte"
    elif class_name == "TPULlamaForCausalLM":
        return "llama"
    else:
        raise ValueError(f"Unsupported model type: {class_name}")
    return class_name


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


def get_position_ids(inputs, num_past_seen_tokens=0):
    assert num_past_seen_tokens >= 0
    # Construct input_ids
    if num_past_seen_tokens == 0:
        position_ids = torch.arange(
            0, inputs.shape[1], device=inputs.device, dtype=torch.int
        ).unsqueeze(dim=0)
    else:
        position_ids = torch.arange(
            num_past_seen_tokens,
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

        self.shared_seen_tokens = 0
        self.draft_seen_tokens = 0
        self.verifier_seen_tokens = 0

        self.reset_caches()

        self.arch_specific_inference_kwargs = dict()
        if self.model_type == "evabyte":
            self.arch_specific_inference_kwargs = {"multibyte_decoding": False}

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

    @torch.no_grad()
    def prefill(self, input_ids, circuit_n_token):

        # Initialises the cache
        self.init_caches(circuit_n_token, batch_size=input_ids.shape[0])

        position_ids = get_position_ids(input_ids, num_past_seen_tokens=0)

        # ============ Prefill: Shared Encoder ========================
        shared_outputs = self.shared_encoder.model(
            input_ids=input_ids,
            use_cache=True,
            position_ids=position_ids,
            past_key_values=self.shared_encoder_cache,
            **self.arch_specific_inference_kwargs,
        )
        shared_last_hidden_state = shared_outputs["last_hidden_state"]
        shared_past_key_values = shared_outputs["past_key_values"]

        if self.model_type == "evabyte":
            # Update kv cache
            shared_outputs["past_key_values"] = (
                self.shared_encoder._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(
                    shared_outputs["past_key_values"]
                )
            )

        if self.has_adapter:
            # ============ Prefill: Draft Encoder ========================
            draft_outputs = self.draft_encoder.model(
                input_ids=input_ids,
                inputs_embeds=shared_last_hidden_state,
                use_cache=True,
                past_key_values=self.draft_encoder_cache,
                position_ids=position_ids,
                **self.arch_specific_inference_kwargs,
            )
            draft_last_hidden_state = draft_outputs["last_hidden_state"]
            draft_past_key_values = draft_outputs["past_key_values"]

            if self.model_type == "evabyte":
                # Update kv cache
                draft_outputs["past_key_values"] = (
                    self.draft_encoder._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(
                        draft_past_key_values
                    )
                )
            # ============ Prefill: Verifier Encoder ========================
            # NOTE: Verifier must stay one step behind Draft
            verifier_outputs = self.verifier_encoder.model(
                input_ids=input_ids[:, :-1],
                inputs_embeds=shared_last_hidden_state[:, :-1],
                use_cache=True,
                past_key_values=self.verifier_encoder_cache,
                position_ids=position_ids[:, :-1],
                **self.arch_specific_inference_kwargs,
            )
            verifier_last_hidden_state = verifier_outputs["last_hidden_state"]
            verifier_past_key_values = verifier_outputs["past_key_values"]

            if self.model_type == "evabyte":
                # Update kv cache
                verifier_outputs["past_key_values"] = (
                    self.verifier_encoder._multi_byte_pred_update_cache_when_prefil_len_eq_window_size(
                        verifier_past_key_values
                    )
                )
        else:
            draft_last_hidden_state = shared_last_hidden_state
            draft_past_key_values = shared_past_key_values
            verifier_last_hidden_state = None
            verifier_past_key_values = None

        self.set_caches(
            shared_cache=shared_past_key_values,
            draft_cache=draft_past_key_values,
            verifier_cache=verifier_past_key_values,
        )
        # Update sequence trackers
        seq_len = input_ids.shape[1]
        self.shared_seen_tokens = seq_len
        self.draft_seen_tokens = seq_len
        self.verifier_seen_tokens = seq_len - 1  # One token behind!

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
                draft_kvs = prepare_encode_kwargs(
                    input_ids=input_ids,
                    cache=self.draft_encoder_cache,
                    encoder=self.draft_encoder,
                    num_past_seen_tokens=self.draft_seen_tokens,
                    model_type=self.model_type,
                )
            shared_kvs = prepare_encode_kwargs(
                input_ids=input_ids,
                cache=self.shared_encoder_cache,
                encoder=self.shared_encoder,
                num_past_seen_tokens=self.shared_seen_tokens,
                model_type=self.model_type,
            )

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
                verifier_kvs = prepare_encode_kwargs(
                    input_ids=input_ids,
                    cache=self.verifier_encoder_cache,
                    encoder=self.verifier_encoder,
                    num_past_seen_tokens=self.verifier_seen_tokens,
                    model_type=self.model_type,
                )
            shared_kvs = prepare_encode_kwargs(
                input_ids=input_ids,
                cache=self.shared_encoder_cache,
                encoder=self.shared_encoder,
                num_past_seen_tokens=self.shared_seen_tokens,
                model_type=self.model_type,
            )

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
                :, self.verifier_seen_tokens:
            ]
            verifier_past_key_values = shared_past_key_values

        results = dict(
            shared_last_hidden_state=shared_last_hidden_state,
            verifier_last_hidden_state=verifier_last_hidden_state,
            shared_past_key_values=shared_past_key_values,
            verifier_past_key_values=verifier_past_key_values,
        )
        return results

    def init_caches(self, num_tokens_speculate, batch_size=1):
        # If we do not have adaptors, we do not need a cache
        # for draft and verifier (as they do not exist)
        self.draft_encoder_cache = None
        self.verifier_encoder_cache = None

        if self.model_type == "evabyte":
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
            if self.has_adapter:
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
        elif self.model_type == "llama":
            # Standard models use HuggingFace DynamicCache
            self.shared_encoder_cache = DynamicCache()
            if self.has_adapter:
                self.draft_encoder_cache = DynamicCache()
                self.verifier_encoder_cache = DynamicCache()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def set_caches(self, shared_cache=None, draft_cache=None, verifier_cache=None):
        if shared_cache is not None:
            self.shared_encoder_cache = shared_cache
        if draft_cache is not None:
            self.draft_encoder_cache = draft_cache
        if verifier_cache is not None:
            self.verifier_encoder_cache = verifier_cache

    def update_shared_cache(self, past_key_values, num_candidates, num_valid):
        assert num_valid <= num_candidates
        self.shared_seen_tokens += num_valid
        if self.model_type == "evabyte":
            self.shared_encoder_cache = self.shared_encoder.multi_byte_pred_update_cache(
                past_key_values,
                torch.arange(
                    num_candidates, device=self.shared_encoder.device, dtype=torch.int
                ).unsqueeze(dim=0),
                0,
                num_valid,
            )
        elif self.model_type == "llama":
            self.shared_encoder_cache.crop(self.shared_seen_tokens)
        assert self.shared_seen_tokens == self.shared_encoder_cache.get_seq_length()

    def update_draft_cache(self, past_key_values, num_candidates, num_valid):
        assert num_valid <= num_candidates
        self.draft_seen_tokens += num_valid
        if self.has_adapter:
            if self.model_type == "evabyte":
                self.draft_encoder_cache = self.draft_encoder.multi_byte_pred_update_cache(
                    past_key_values,
                    torch.arange(
                        num_candidates, device=self.draft_encoder.device, dtype=torch.int
                    ).unsqueeze(dim=0),
                    0,
                    num_valid,
                )
            elif self.model_type == "llama":
                self.draft_encoder_cache.crop(self.shared_seen_tokens)
            assert self.draft_seen_tokens == self.draft_encoder_cache.get_seq_length()

    def update_verifier_cache(self, past_key_values, num_candidates, num_valid):
        assert num_valid <= num_candidates
        self.verifier_seen_tokens += num_valid
        if self.has_adapter:
            if self.model_type == "evabyte":
                self.verifier_encoder_cache = (
                    self.verifier_encoder.multi_byte_pred_update_cache(
                        past_key_values,
                        torch.arange(
                            num_candidates,
                            device=self.verifier_encoder.device,
                            dtype=torch.int,
                        ).unsqueeze(dim=0),
                        0,
                        num_valid,
                    )
                )
            elif self.model_type == "llama":
                self.verifier_encoder_cache.crop(self.shared_seen_tokens)
            assert (
                self.verifier_seen_tokens
                == self.verifier_encoder_cache.get_seq_length()
            )

    def reset_caches(self):
        self.shared_encoder_cache = None
        self.draft_encoder_cache = None
        self.verifier_encoder_cache = None

        self.shared_seen_tokens = 0
        self.draft_seen_tokens = 0
        self.verifier_seen_tokens = 0
