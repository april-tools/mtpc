from dataclasses import dataclass
from torch import nn

import torch
import torch.nn.functional as F

from .mlp import Block


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
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

    def forward(self, idx, targets=None, return_logits=True):
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = 30 * torch.tanh(logits / 30)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            logits = 30 * torch.tanh(logits / 30)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return dict(logits=logits, loss=loss, stp_loss=loss)

    @torch.no_grad()
    def generate(self, inputs: torch.Tensor, mode='stp'):

        if mode != 'stp':
            raise ValueError('Only single token generation supported')

        results = self.forward(inputs, return_logits=True)

        sample = torch.argmax(results['logits'], dim=2)
        return sample


# For our experiments, we only need the encoder of nanoGPT
class GPTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
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

    def forward(self, xx):
        xx = self.transformer.wte(xx)  # token embeddings of shape (B, S, n_embd)
        # TODO: Decide RMS_NORM positioning
        xx = F.rms_norm(xx, (xx.size(-1),))
        for block in self.transformer.h:
            xx = block(xx)
        xx = F.rms_norm(xx, (xx.size(-1),))

        return xx
