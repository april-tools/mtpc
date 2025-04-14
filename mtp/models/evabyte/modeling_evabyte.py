from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

from .configuration_evabyte import EvaByteConfig
try:
    import triton
    USE_TRITON_IMPL = True
    from .eva import EvaAttention
except ImportError:
    USE_TRITON_IMPL = False
    print("WARNING: triton is not installed, using fallback EVA which might be slow and throw errors")
    from .eva_pt_ref import EvaAttention

MASK_MIN_VALUE = -10e10

def prepare_eva_attention_mask(
        seq_len, 
        device, 
        chunk_size, 
        window_size,
        use_cache=False, 
        cache=None
    ):
    """
    Prepare attention masks for EVA.
    
    """
    chunk_causal_mask  = None
    window_causal_mask = None
    if use_cache:
        cached_seq_len = cache.get_seq_length()
        total_seq_len = seq_len + cached_seq_len
        # cached_seq_len will be 0 during prefilling
        # padded_seq_len = chunk_size * math.ceil(total_seq_len / chunk_size)
        padded_seq_len = window_size * math.ceil(total_seq_len / window_size)
        num_chunks = padded_seq_len // chunk_size
    else:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        assert seq_len % chunk_size == 0
        num_chunks = seq_len // chunk_size

        assert seq_len % window_size == 0

    # create causal mask
    ################################
    # generate chunked causal masks
    ################################
    # [b, h, j, c, c]
    chunks_per_window = window_size // chunk_size
    if num_chunks >= chunks_per_window:
        chunk_causal_mask = torch.ones(
            (chunk_size, num_chunks, num_chunks), 
            device=device,
            dtype=torch.bool
        ).triu(0)
        
        num_blocks = num_chunks // chunks_per_window
        chunk_causal_mask = chunk_causal_mask.reshape(
            chunk_size,
            num_blocks, 
            chunks_per_window, 
            num_blocks, 
            chunks_per_window
        ).transpose(-2, -3)

        block_diag_zero = (
            torch.eye(num_blocks, device=device, dtype=torch.bool)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(0)
        )

        # Set diagonal blocks to zero
        chunk_causal_mask = chunk_causal_mask.masked_fill(block_diag_zero, True)

        # Reshape back to original size
        chunk_causal_mask = (
            chunk_causal_mask
            .transpose(-2, -3)
            .reshape(chunk_size, num_chunks, num_chunks)
            .transpose(-2, -3)
            .reshape(chunk_size * num_chunks, num_chunks)
            .unsqueeze(0)
            .unsqueeze(0)
        )
    else:
        chunk_causal_mask = torch.ones(
            (1, 1, chunk_size, num_chunks, num_chunks), 
            device=device,
            dtype=torch.bool,
        ).triu(0).transpose(-2, -3) # [1, 1, c, j, c]
        chunk_causal_mask = chunk_causal_mask.reshape(
            1, 1, chunk_size * num_chunks, num_chunks
        ) # [1, 1, n, c]

    if use_cache:
        chunk_causal_mask = chunk_causal_mask[..., cached_seq_len : cached_seq_len + seq_len, :]

    window_causal_mask = torch.ones(
        (1, 1, 1, window_size, window_size), 
        device=device
    ).triu(1).to(torch.bool)
    return (chunk_causal_mask, window_causal_mask)

def pad_to_multiple(tensor, multiple, dim=-2, value=0, create_mask=False, left_padding=False):
    assert dim < 0 # only accept ``dim'' index in a reverse manner
    seqlen = int(tensor.shape[dim])
    m = seqlen / multiple
    if m.is_integer():
        if create_mask:
            return tensor, torch.ones(size=(tensor.shape[0], tensor.shape[dim]), dtype=torch.bool, device=tensor.device)
        else:
            return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    if left_padding:
        padded_res = F.pad(tensor, (*pad_offset, remainder, 0), value=value)
    else:
        padded_res = F.pad(tensor, (*pad_offset, 0, remainder), value=value)
    if create_mask:
        # assume dim 0 is the batch size
        padding_mask = torch.ones(size=(padded_res.shape[0], padded_res.shape[dim]), dtype=torch.bool, device=padded_res.device)
        if left_padding:
            padding_mask[:, :remainder] = False
        else:
            padding_mask[:, -remainder:] = False
        return padded_res, padding_mask
    else:
        return padded_res

class EvaByteRMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fp32_ln = True
        self.variance_epsilon = config.rms_norm_eps
        self.add_unit_offset = config.norm_add_unit_offset
        if self.add_unit_offset:
            self.weight = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states):
        _hidden_states = hidden_states.to(torch.float32 if self.fp32_ln else torch.bfloat16)

        variance = _hidden_states.pow(2).mean(-1, keepdim=True)
        _hidden_states = _hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.add_unit_offset:
            return ((1 + self.weight) * _hidden_states).type_as(hidden_states)
        else:
            return (self.weight * _hidden_states).type_as(hidden_states)

class EvaByteRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # return (
        #     self.cos_cached[:seq_len].to(dtype=x.dtype),
        #     self.sin_cached[:seq_len].to(dtype=x.dtype),
        # )
        if seq_len < self.max_seq_len_cached:
            cos_slice = self.cos_cached.split(seq_len, dim=0)[0]
            sin_slice = self.sin_cached.split(seq_len, dim=0)[0]
        else:
            cos_slice = self.cos_cached
            sin_slice = self.sin_cached

        return (
            cos_slice.to(dtype=x.dtype),
            sin_slice.to(dtype=x.dtype),
        )


class EvaByteMLP(nn.Module):
    def __init__(self, config, layer_idx: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = layer_idx
        self.config = config

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class EvaByteDecoderLayer(nn.Module):
    def __init__(self, config: EvaByteConfig, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = EvaAttention(config=config, layer_idx=layer_idx)
        self.mlp = EvaByteMLP(config, layer_idx=layer_idx)
        self.input_layernorm = EvaByteRMSNorm(config)
        self.post_attention_layernorm = EvaByteRMSNorm(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cos: Optional[torch.Tensor] = None,
            sin: Optional[torch.Tensor] = None,
            multibyte_decoding: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        if self.config.fp32_skip_add:
            residual = residual.float()

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(hidden_states=hidden_states,
                                                                            attention_mask=attention_mask,
                                                                            position_ids=position_ids,
                                                                            past_key_value=past_key_value,
                                                                            output_attentions=output_attentions,
                                                                            use_cache=use_cache,
                                                                            cos=cos,
                                                                            sin=sin,
                                                                            multibyte_decoding=multibyte_decoding)
        hidden_states = (residual + hidden_states).to(hidden_states.dtype)

        # Fully Connected
        residual = hidden_states
        if self.config.fp32_skip_add:
            residual = residual.float()
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = (residual + hidden_states).to(hidden_states.dtype)

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (self_attn_weights, )

        if use_cache:
            outputs += (present_key_value, )
        return outputs


class EvaByteRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=self.inv_freq.device,
                                dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # return (
        #     self.cos_cached[:seq_len].to(dtype=x.dtype),
        #     self.sin_cached[:seq_len].to(dtype=x.dtype),
        # )
        if seq_len < self.max_seq_len_cached:
            cos_slice = self.cos_cached.split(seq_len, dim=0)[0]
            sin_slice = self.sin_cached.split(seq_len, dim=0)[0]
        else:
            cos_slice = self.cos_cached
            sin_slice = self.sin_cached

        return (
            cos_slice.to(dtype=x.dtype),
            sin_slice.to(dtype=x.dtype),
        )


def pad_to_multiple(tensor, multiple, dim=-2, value=0, create_mask=False, left_padding=False):
    assert dim < 0 # only accept ``dim'' index in a reverse manner
    seqlen = int(tensor.shape[dim])
    m = seqlen / multiple
    if m.is_integer():
        if create_mask:
            return tensor, torch.ones(size=(tensor.shape[0], tensor.shape[dim]), dtype=torch.bool, device=tensor.device)
        else:
            return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    if left_padding:
        padded_res = F.pad(tensor, (*pad_offset, remainder, 0), value=value)
    else:
        padded_res = F.pad(tensor, (*pad_offset, 0, remainder), value=value)
    if create_mask:
        # assume dim 0 is the batch size
        padding_mask = torch.ones(size=(padded_res.shape[0], padded_res.shape[dim]), dtype=torch.bool, device=padded_res.device)
        if left_padding:
            padding_mask[:, :remainder] = False
        else:
            padding_mask[:, -remainder:] = False
        return padded_res, padding_mask
    else:
        return padded_res


def prepare_eva_generation_attn_mask_triton(
    xx: Tensor,
    attention_mask: Tensor = None,
    use_cache: bool = False,
    past_key_values: Cache = None,
    *,
    config: EvaByteConfig
) -> tuple:
    batch_size, seq_len = xx.shape[0], xx.shape[1]
    if use_cache and past_key_values.get_seq_length() > 0:
        # decoding phase
        if past_key_values.rf_mask[0] is not None:
            cur_rf_mask = torch.zeros(
                (batch_size, 1, seq_len, 1),
                dtype=past_key_values.rf_mask[0].dtype,
                device=past_key_values.rf_mask[0].device
            )
        else:
            cur_rf_mask = None
            
        if past_key_values.s_mask[0] is not None:
            cur_s_mask = torch.zeros(
                (batch_size, 1, seq_len, 1),
                dtype=past_key_values.s_mask[0].dtype,
                device=past_key_values.s_mask[0].device
            )
        else:
            cur_s_mask = None
            
        seen_tokens = past_key_values.get_seq_length()
        if seen_tokens <= config.window_size:
            rfa_chunks_dummy_mask = None
        else:
            if cur_s_mask is not None: 
                chunks_per_window = int(config.window_size // config.chunk_size)
                # the ongoing decoding step would be (seen_seq_len + 1)-th token
                num_windows_seen_so_far = seen_tokens // config.window_size
                rfa_chunks_dummy_mask = torch.zeros(
                    (batch_size, 1, seq_len, num_windows_seen_so_far * chunks_per_window),
                    dtype=past_key_values.s_mask[0].dtype,
                    device=past_key_values.s_mask[0].device
                )
            else:
                rfa_chunks_dummy_mask = None
        # rf_mask and cur_mask are 0s because we do not want to mask them
        return (cur_s_mask, cur_rf_mask, rfa_chunks_dummy_mask)

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        # convert 0 -> padding to 1 -> padding
        padded_attention_mask = pad_to_multiple(
            attention_mask, 
            config.window_size, 
            dim=-1,
            value=0, 
            create_mask=False,
            left_padding=False
        )
        # convert 0 -> padding to 1 -> padding
        padded_rf_mask = ~padded_attention_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]
        # [b, 1, w, j, 1]
        padded_w_attn_mask = padded_rf_mask.reshape(batch_size, 1, -1, config.window_size, 1).to(torch.bool)
        # [b, 1, w, j, 1] [b, 1, w, 1, j] -> [b, 1, w, j, j]
        w_padding_mask = torch.logical_or(padded_w_attn_mask, padded_w_attn_mask.transpose(-1, -2))
        w_causal_mask = torch.ones(
            (1, 1, 1, config.window_size, config.window_size),
            device=xx.device
        ).triu(1).to(torch.bool)
        s_mask = torch.logical_or(w_padding_mask, w_causal_mask)
        s_mask = s_mask.reshape(batch_size, 1, -1, config.window_size)
        s_mask = s_mask[..., :seq_len, :]
        # negate the attention mask to get the padding mask
        rf_mask = ~attention_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool) # [b, 1, n, 1]
        return (s_mask, rf_mask)
    else:
        return (None, None)
