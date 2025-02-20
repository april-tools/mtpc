import copy
import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from .mlp import Block


# TODO: Maybe move this to a medusa module - since we will likely compare to medusa anyway
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

    def reset_parameters(self):
        # NOTE: Below needed as we need to reset the bias too
        self.linear.reset_parameters()
        # TODO: Maybe better to just init all random but very small?
        torch.nn.init.zeros_(self.linear.weight)


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
        # xx is B, S, R, D

        xx = F.rms_norm(xx, (xx.size(-1),))
        xx = self.gelu(xx)
        return xx

    def reset_parameters(self):
        # TODO: Maybe better to just init all random but very small?
        torch.nn.init.normal_(self.Wr, mean=0.0, std=0.02)


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
        # xxs is B, S, R, D
        xxs = torch.stack(xxs, dim=-2)
        return xxs

    def reset_parameters(self):
        # TODO: Maybe better to just init all random but very small?
        for mlp in self.mlps:
            for ss in mlp:
                ss.reset_parameters()


class ExpanderHead(torch.nn.Module):
    """Wrapper class of expanders to make running from config easier."""
    def __init__(self, n_embd: int, n_component: int, n_layer: int=1, expander_type='linear'):
        super().__init__()
        self.n_embd = n_embd            # D
        self.n_component = n_component  # R
        assert expander_type in ['linear', 'mlp']
        if expander_type == 'linear':
            assert n_layer in (1, None), 'n_layer is only valid for MLP'
        self.n_layer = n_layer
        self.expander_type = expander_type
        if self.expander_type == 'linear':
            self.expander = LinearExpanderHead(self.n_embd, self.n_component)
        elif self.expander_type == 'mlp':
            self.expander = MLPExpanderHead(self.n_embd, self.n_component, self.n_layer)

    def forward(self, xx: Tensor) -> Tensor:
        return self.expander(xx)

    def reset_parameters(self):
        # TODO: Maybe better to just init all random but very small?
        self.expander.reset_parameters()


class TransformerEncoderHead(torch.nn.Module):
    # Create custom parameterisation for each output token

    def __init__(self, n_embd: int, n_head: int = 6, n_layer: int = 2):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        assert n_layer >= 0
        self.transformer = torch.nn.ModuleList([Block(n_head, n_embd) for _ in range(self.n_layer)])

    def forward(self, xx):
        # Batch, Sentence Length, Embed Dim
        # B, S, D = xx.shape

        xx = F.rms_norm(xx, (xx.size(-1),))
        for block in self.transformer:
            xx = block(xx)
        xx = F.rms_norm(xx, (xx.size(-1),))
        return xx

    def reset_parameters(self):
        for each in self.transformer:
            each.reset_parameters()


class OutputHead(torch.nn.Module):

    def __init__(self, encoder: TransformerEncoderHead, expander: ExpanderHead | Linear):
        super().__init__()
        self.encoder = encoder
        # Expands parametrisation for mixture model
        # NOTE: For the case of the categoricals, we want a tensor B, S, R, *D*  (expander)
        # For the case of the sum layer, we want                   B, S, R
        # We therefore allow the expander to be a simple linear layer (not linear expander)
        self.expander = expander

    def forward(self, xx: Tensor, generate: bool = False) -> Tensor:
        # xx is B, S, D

        # We can bypass the transformer encoder by setting it to have n_layer=0
        if self.encoder.n_layer > 0:
            xx = self.encoder(xx)
        if generate:
            xx = xx[:, [-1]]

        xx = self.expander(xx)
        return xx

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.expander.reset_parameters()


class MultiTokenHead(torch.nn.Module):
    def __init__(
        self,
        token_head: OutputHead,
        sum_weight_head: OutputHead,
        vocab_size: int,
        n_embd: int,
        n_component: int = 1,
        n_token: int = 3,
        freeze_unembedding=False
    ):
        super().__init__()
        self.vocab_size = vocab_size           # V
        self.token_head = token_head
        self.sum_weight_head = sum_weight_head
        self.n_embd = n_embd                   # D
        self.n_component = n_component         # R
        self.n_token = n_token                 # H
        self.freeze_unembedding = freeze_unembedding

        # Projection to the Categorical log probs
        self.token_heads = torch.nn.ModuleList([
            copy.deepcopy(self.token_head)
            for _ in range(self.n_token)
        ])
        for th in self.token_heads:
            th.reset_parameters()
        # Delete the original instance, since we took deep copies
        del self.token_head
        # If we only have one component we do not need a sum_weight_head
        if self.n_component == 1:
            del self.sum_weight_head
        # The shared unembedding matrix
        self.W = torch.nn.Linear(self.n_embd, self.vocab_size, bias=False)
        # Potentially freeze unembedding weights
        for p in self.W.parameters():
            p.requires_grad = not self.freeze_unembedding

    def set_unembedding_weights(self, weights):
        self.W.weight.data = weights

    def forward(self, xx: Tensor, generate: bool = False) -> dict[str, Tensor]:
        # xx: (B, S, D)
        logits = []
        # TODO: Can we avoid the for loop?
        for token_head in self.token_heads:
            # head_xx: (B, S, R, D)
            head_xx = token_head(xx, generate=generate)
            # head_logits: (B, S, R, V)
            head_logits = self.W(head_xx)
            logits.append(head_logits)
        # cat_log_probs: (H, B, S, R, V)
        cat_log_probs = torch.log_softmax(torch.stack(logits, dim=0), dim=-1)

        # sum_weight: (B, S, 1, R)
        if self.n_component > 1:
            sum_weight = self.sum_weight_head(xx, generate=generate)
            sum_weight = torch.softmax(
                sum_weight.unsqueeze(dim=2),
                dim=-1
            )
        else:
            B, S, D = xx.shape
            if generate:
                shape = (B, 1, 1, 1)
            else:
                shape = (B, S, 1, 1)
            sum_weight = torch.ones(*shape,
                                    device=xx.device,
                                    requires_grad=False)

        return dict(cat_log_probs=cat_log_probs, sum_weight=sum_weight)
