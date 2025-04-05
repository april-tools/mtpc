from torch import nn, Tensor, LongTensor

import torch
import torch.nn.functional as F

from transformers.configuration_utils import PretrainedConfig

from .mlp import Block


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = Rotary.apply_rotary(q, cos, sin), Rotary.apply_rotary(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y

    def reset_parameters(self):
        self.c_q.reset_parameters()
        self.c_k.reset_parameters()
        self.c_v.reset_parameters()
        self.c_proj.reset_parameters()


def _init_weights(module: nn.Module):
    from .mlp import MLP

    if isinstance(module, CausalSelfAttention):
        nn.init.zeros_(module.c_proj.weight)
    elif isinstance(module, MLP):
        nn.init.zeros_(module.c_proj.weight)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class GPTHead(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self._default_dtype = torch.get_default_dtype()

    def forward(self, x: Tensor, cast_default_dtype: bool = True) -> Tensor:
        # Compute the logits
        logits = self.lm_head(x)
        # NOTE: I am disabling logit softcapping
        # TODO: Add option to enable and check if it helps
        # logits = 30 * torch.tanh(logits / 30)
        if cast_default_dtype:
            # e.g., use fp32 for logits
            logits = logits.to(self._default_dtype)
        return logits

    @property
    def weight(self):
        return self.lm_head.weight


class GPTEncoder(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int = 12, n_head: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embd),
                h=nn.ModuleList([Block(n_head, n_embd) for _ in range(n_layer)]),
            )
        )
        self.apply(_init_weights)

    def forward(self,
                input_ids: LongTensor,
                attention_mask: LongTensor | None = None,
                ) -> Tensor:

        if not (attention_mask is None or torch.all(attention_mask == 1)):
            raise NotImplementedError('NanoGPT transformers cannot handle attention mask that is not all ones')

        xx = self.transformer.wte(input_ids)  # token embeddings of shape (B, S, n_embd)
        # TODO: Decide RMS_NORM positioning
        xx = F.rms_norm(xx, (xx.size(-1),))
        for block in self.transformer.h:
            xx = block(xx)
        xx = F.rms_norm(xx, (xx.size(-1),))
        return dict(last_hidden_state=xx)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_layer: int = 12,
        n_head: int = 6,
        encoder_only: bool = False
    ):
        super().__init__()
        self.config = PretrainedConfig(
            model_type='nanoGPT',
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head
        )
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.encoder: GPTEncoder = GPTEncoder(vocab_size, n_embd, n_layer, n_head)
        self.head: GPTHead | None = None if encoder_only else GPTHead(n_embd, vocab_size)
        self.apply(_init_weights)

    def forward(self,
                input_ids: LongTensor,
                labels: LongTensor | None = None,
                attention_mask: LongTensor | None = None,
                return_logits: bool = True
    ) -> tuple[Tensor | None, None]:
        assert self.head is not None, "The forward of GPT can only be called if encoder_only=False"

        if not (attention_mask is None or torch.all(attention_mask == 1)):
            raise NotImplementedError('NanoGPT transformers cannot handle attention mask that is not all ones')

        # forward the GPT model itself
        x = self.encoder(input_ids)['last_hidden_state']  # token embeddings of shape (b, t, n_embd)

        if labels is not None:
            # if we are given some desired labels also calculate the loss
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return dict(logits=logits, loss=loss)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.register_buffer('inv_freq', 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

    @staticmethod
    def apply_rotary(x, cos, sin):
        assert x.ndim == 4  # multihead attention
        d = x.shape[3]//2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)
    #
    # @torch.no_grad()
    # def generate(self, inputs: torch.Tensor, use_argmax: bool = False, mode: str = 'stp') -> Tensor:
    #     if mode != 'stp':
    #         raise ValueError('Only single token generation is supported')
    #     results = self.forward(inputs, return_logits=True)
    #     logits = results['logits']
    #     if use_argmax:
    #         toks = torch.argmax(logits, dim=2)
    #     else:
    #         probs = torch.softmax(logits, dim=2)
    #         toks = torch.multinomial(probs.squeeze(dim=1), num_samples=1)
    #     return toks
