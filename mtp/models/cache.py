import torch

from transformers.cache_utils import DynamicCache

from mtp.models.evabyte.multibyte_decoding_evabyte import (
    multi_byte_pred_prepare_attn_mask,
)
from mtp.models.evabyte.eva_cache import EvaStaticCacheForTriton
from mtp.utils.model_types import get_model_type


class KVCacheWrapper(object):
    """This KVCacheWrapper implementation provides a unified interface for the
    kv-cache of all models we support in the codebase"""

    def __init__(
        self,
        model_type,
        config,
        device,
        dtype,
        batch_size=1,
        update_fn=None,
        prefill_update_fn=None,
    ):
        super().__init__()
        self.model_type = model_type
        self.config = config
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.cache = None

        # Store function references for EvaByte
        self._update_fn = update_fn
        self._prefill_update_fn = prefill_update_fn

        # Initialize cache
        self.init()

    @classmethod
    def for_model(cls, encoder, batch_size=1):
        """Create a KVCacheWrapper for the given encoder model.

        Args:
            encoder: The encoder model to create cache for
            batch_size: Batch size for cache allocation

        Returns:
            KVCacheWrapper instance configured for the encoder
        """
        model_type = get_model_type(encoder)

        # Store config, device, and dtype
        config = encoder.config
        device = encoder.device

        dtype = encoder.config.torch_dtype

        # For EvaByte, store function references
        if model_type == "evabyte":
            update_fn = encoder.multi_byte_pred_update_cache
            prefill_update_fn = (
                encoder._multi_byte_pred_update_cache_when_prefil_len_eq_window_size
            )
        else:
            update_fn = None
            prefill_update_fn = None

        return cls(
            model_type=model_type,
            config=config,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            update_fn=update_fn,
            prefill_update_fn=prefill_update_fn,
        )

    def init(self):
        """Initialize the underlying cache based on model type."""
        self.seen_tokens = 0
        if self.model_type == "evabyte":
            self.cache = EvaStaticCacheForTriton(
                self.batch_size,
                self.config.num_attention_heads,
                self.config.window_size,
                self.config.hidden_size // self.config.num_attention_heads,
                self.config.num_hidden_layers,
                self.dtype,
                self.device,
            )
        elif self.model_type == "llama":
            self.cache = DynamicCache()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def update(self, past_key_values, num_candidates, num_valid):
        """Update cache after generation, keeping only valid tokens.

        Args:
            past_key_values: The current cache state
            num_candidates: Number of candidate tokens generated
            num_valid: Number of valid tokens to keep

        Returns:
            Updated cache
        """
        assert num_valid <= num_candidates
        self.seen_tokens += num_valid

        if self.model_type == "evabyte":
            self.cache = self._update_fn(
                past_key_values,
                torch.arange(
                    num_candidates, device=self.device, dtype=torch.int
                ).unsqueeze(dim=0),
                0,
                num_valid,
            )
        elif self.model_type == "llama":
            self.cache.crop(self.seen_tokens)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        assert self.seen_tokens == self.cache.get_seq_length()
        return self.cache

    def prefill_update(self, past_key_values):
        """Special update for EvaByte during prefill phase.

        Args:
            past_key_values: The cache state after prefill

        Returns:
            Updated cache (for evabyte) or original cache (for llama)
        """
        if self.model_type == "evabyte":
            self.cache = self._prefill_update_fn(past_key_values)
        # No special prefill update needed for llama
        elif self.model_type == "llama":
            pass
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.seen_tokens = self.cache.get_seq_length()
        return self.cache

    def reset(self):
        """Reset the cache and seen tokens counter."""
        self.init()

    # TBH the functions below do not belong here
    # but they are convenient helpers - keep everything in one place
    def get_position_ids(self, input_ids):
        # Build position IDs for new tokens starting from seen_tokens
        assert self.seen_tokens >= 0
        assert self.seen_tokens <= input_ids.shape[1]
        # Construct input_ids
        position_ids = torch.arange(
            self.seen_tokens,
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.int,
        ).unsqueeze(dim=0)
        return position_ids

    def get_attn_mask(self, input_ids):
        attn_mask = None
        if self.model_type == "evabyte" and self.seen_tokens > 0:
            attn_mask = multi_byte_pred_prepare_attn_mask(
                self.config,
                self.seen_tokens,
                input_ids.shape[1] - self.seen_tokens,
                device=input_ids.device,
            )
        return attn_mask

    def get_encoder_kwargs(self, input_ids, prefill=False):
        # Produce attention, position ids and past key values
        kwargs = {
            "attention_mask": self.get_attn_mask(input_ids),
            "position_ids": self.get_position_ids(input_ids),
            "past_key_values": self.cache,
        }
        return kwargs
