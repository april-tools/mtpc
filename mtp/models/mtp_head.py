import torch
from torch import Tensor
import torch.nn.functional as F
from .mlp import Block


# Taken from the Medusa paper's code
# https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py
class ResBlock(torch.nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = torch.nn.SiLU()

    def forward(self, xx: Tensor) -> Tensor:
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return xx + self.act(self.linear(xx))


class LinearExpanderHead(torch.nn.Module):
    # Expand parametrisation for mixture model

    def __init__(self, n_embd: int, n_component: int):
        super().__init__()
        self.n_embd = n_embd            # D
        self.n_component = n_component  # R
        self.gelu = torch.nn.GELU()
        # Below is equivalent to R square linear layers
        self.Wr = torch.nn.Parameter(torch.zeros(self.n_component,
                                                 self.n_embd,
                                                 self.n_embd))
        torch.nn.init.normal_(self.Wr, mean=0.0, std=0.02)

    def forward(self, xx: Tensor) -> Tensor:
        # Batch, Sentence Length, Embed Dim
        B, S, D = xx.shape

        # Collapse: B x S, D for the matmul
        xx = xx.reshape(-1, D)

        # Wr is R, D, D
        xx = xx @ self.Wr
        # xx is R, B x S, D
        xx = xx.permute(1, 0, 2)
        # xx is B x S, R, D
        xx = xx.reshape(B, S, -1, D)

        xx = F.rms_norm(xx, (xx.size(-1),))
        xx = self.gelu(xx)
        return xx


class MLPExpanderHead(torch.nn.Module):
    # Expand parametrisation for mixture model

    def __init__(self, n_embd: int, n_component: int, n_layer: int=1):
        super().__init__()
        self.n_embd = n_embd            # D
        self.n_component = n_component  # R
        self.n_layer = n_layer
        self.mlps = torch.nn.ModuleList([torch.nn.Sequential(*([ResBlock(self.n_embd)] * self.n_layer))
                                         for c in range(self.n_component)])

    def forward(self, xx: Tensor) -> Tensor:
        # Batch, Sentence Length, Embed Dim
        B, S, D = xx.shape

        xxs = []
        for mlp in self.mlps:
            act = mlp(xx)
            xxs.append(act)
        xxs = torch.stack(xxs, dim=-2)
        return xxs


class TransformerEncoderHead(torch.nn.Module):
    # Create custom parameterisation for each output token

    def __init__(self, n_embd: int, n_head: int = 6, n_layer: int = 2):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.transformer = torch.nn.ModuleList([Block(n_head, n_embd) for _ in range(self.n_layer)])

    def forward(self, xx):
        # Batch, Sentence Length, Embed Dim
        # B, S, D = xx.shape

        xx = F.rms_norm(xx, (xx.size(-1),))
        for block in self.transformer:
            xx = block(xx)
        xx = F.rms_norm(xx, (xx.size(-1),))
        return xx


class TokenHead(torch.nn.Module):

    def __init__(self, encoder: TransformerEncoderHead, expander: LinearExpanderHead | None = None):
        super().__init__()
        self.encoder = encoder
        # Expands parametrisation for mixture model
        self.expander = expander

    def forward(self, xx: Tensor, generate: bool = False) -> Tensor:
        # xx is B, S, D
        xx = self.encoder(xx)
        if generate:
            xx = xx[:, [-1]]

        # xx is B, S, D
        if self.expander is not None:
            xx = self.expander(xx)
        else:
            xx = xx.unsqueeze(dim=2)
        # xx is B, S, R, D
        return xx


class MultiTokenHead(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_layer: int = 2,
        n_head: int = 6,
        n_component: int = 1,
        n_token: int = 3
    ):
        super().__init__()
        self.vocab_size = vocab_size           # V
        self.n_embd = n_embd                   # D
        self.n_head = n_head
        self.n_component = n_component         # R
        self.n_token = n_token                 # H

        # number of heads and layers in the multi-token transformer
        self.n_head = n_head
        self.n_layer = n_layer

        # Projection to the Categorical log probs
        self.token_heads = torch.nn.ModuleList([
            TokenHead(
                encoder=TransformerEncoderHead(self.n_embd, n_head=self.n_head, n_layer=self.n_layer),
                expander=LinearExpanderHead(self.n_embd, self.n_component)
            )
            for _ in range(self.n_token)
        ])
        self.proj_cat_logits = torch.nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # Projection to the sum layer parameters
        self.sum_weight_head = TransformerEncoderHead(self.n_embd, self.n_head, n_layer=self.n_layer)
        self.proj_sum_weight = torch.nn.Linear(self.n_embd, self.n_component, bias=False)

    def forward(self, xx: Tensor, generate: bool = False) -> dict[str, Tensor]:
        # xx: (B, S, D)
        logits = []
        # TODO: Can we avoid the for loop?
        for token_head in self.token_heads:
            # head_xx: (B, S, R, D)
            head_xx = token_head(xx, generate=generate)
            # head_logits: (B, S, R, V)
            head_logits = self.proj_cat_logits(head_xx)
            logits.append(head_logits)
        # cat_log_probs: (H, B, S, R, V)
        cat_log_probs = torch.log_softmax(torch.stack(logits, dim=0), dim=-1)

        # sum_weight: (B, S, 1, R)
        sum_weight = self.sum_weight_head(xx)
        if generate:
            sum_weight = sum_weight[:, [-1]]
        sum_weight = torch.softmax(
            self.proj_sum_weight(sum_weight).unsqueeze(dim=2),
            dim=-1
        )

        return dict(cat_log_probs=cat_log_probs, sum_weight=sum_weight)
