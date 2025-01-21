from torch import nn, Tensor

import torch
import torch.nn.functional as F

from .mlp import Block


def _init_weights(module: nn.Module):
    from .attention import CausalSelfAttention
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
        logits = 30 * torch.tanh(logits / 30)
        if cast_default_dtype:
            # e.g., use fp32 for logits
            logits = logits.to(self._default_dtype)
        return logits


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

    def forward(self, xx: Tensor) -> Tensor:
        xx = self.transformer.wte(xx)  # token embeddings of shape (B, S, n_embd)
        # TODO: Decide RMS_NORM positioning
        xx = F.rms_norm(xx, (xx.size(-1),))
        for block in self.transformer.h:
            xx = block(xx)
        xx = F.rms_norm(xx, (xx.size(-1),))
        return xx


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_layer: int = 12, n_head: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.encoder = GPTEncoder(vocab_size, n_embd, n_layer, n_head)
        self.head = GPTHead(n_embd, vocab_size)
        self.apply(_init_weights)

    def forward(
        self, idx: Tensor, targets: Tensor | None = None, return_logits: bool = True
    ) -> tuple[Tensor | None, None]:
        # forward the GPT model itself
        x = self.encoder(idx)  # token embeddings of shape (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    @torch.no_grad()
    def generate(self, inputs: torch.Tensor, use_argmax: bool = True) -> Tensor:
        logits, _ = self.forward(inputs, return_logits=True)
        if use_argmax:
            toks = torch.argmax(logits, dim=2)
        else:
            probs = torch.softmax(logits, dim=2)
            toks = torch.multinomial(probs.squeeze(dim=1), num_samples=1)
        return toks
