import torch
import torch.nn.functional as F

from torch import nn


class MLP(nn.Module):
    def __init__(self, in_out_features: int):
        super().__init__()
        self.c_fc = nn.Linear(in_out_features, 4 * in_out_features, bias=False)
        self.c_proj = nn.Linear(4 * in_out_features, in_out_features, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        from .attention import CausalSelfAttention
        self.attn = CausalSelfAttention(n_head, n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x
