import math
import torch
import torch.nn.functional as F

from torch import Tensor
from torch import nn
from torch.nn import init

from mtp.models.circuits import ParametersConfig
from .mlp import Block


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    # This is part of the Medusa model. However, here we vectorize it over an extra batch dimension on the parameters.
    # https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, n_component: int, hidden_size: int):
        super().__init__()
        self.n_component = n_component
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(n_component, hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(n_component, hidden_size))

        # Initialize the weight tensors as identity mapping
        init.zeros_(self.weight)
        # Initialize the bias term
        self._init_bias()

        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, xx: Tensor) -> Tensor:
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        # xx: (B, S, R, D) or more in general (..., R, D)
        # zz: (B, S, R, D) or more in general (..., R, D)
        zz = torch.einsum('...rd,rdc->...rc', xx, self.weight)
        return xx + self.act(zz + self.bias)

    def _init_bias(self):
        # Intialize the bias matrix
        # This is currently using the default initialization of nn.Linear
        for i in range(self.n_component):
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias[i], -bound, bound)

    def reset_parameters(self):
        self._init_bias()
        # TODO: Maybe better to just init all random but very small?
        nn.init.zeros_(self.weight)


class LinearExpanderHead(nn.Module):
    def __init__(self, n_embd: int, n_expand: int):
        super().__init__()
        self.n_embd = n_embd  # D
        self.n_expand = n_expand  # R
        # Below is equivalent to R square linear layers
        selfWr = nn.Parameter(
            torch.zeros(self.n_expand, self.n_embd, self.n_embd)
        )
        self.reset_parameters()

    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) or more in general (..., D)
        xx = torch.einsum('...d,rdc->...rc', xx, self.Wr)
        return xx

    def reset_parameters(self):
        # At the beginning of training we want the linear layer to not
        # change the logits - so we want an identity matrix
        eye = torch.eye(self.n_embd, device=self.Wr.device)
        # Expand to R x n_embd x n_embd
        eye = eye.unsqueeze(0).repeat(self.n_component, 1, 1)
        self.Wr.data = eye


class MLPExpanderHead(nn.Module):
    def __init__(self, n_embd: int, n_expand: int, n_layer: int = 1):
        super().__init__()
        assert n_layer >= 1
        self.n_embd = n_embd  # D
        self.n_expand = n_expand  # R
        self.n_layer = n_layer
        self.mlp = nn.Sequential(
            [ResBlock(n_expand, self.n_embd) for _ in range(self.n_layer)]
        )

    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) -> (B, S, 1, D) -> (B, S, R, D)
        xx = xx.unsqueeze(dim=2)
        xx = self.mlp(xx)
        return xx

    def reset_parameters(self):
        # TODO: Maybe better to just init all random but very small?
        for ss in self.mlp:
            ss.reset_parameters()


class ExpanderHead(nn.Module):
    """Wrapper class of expanders to make running from config easier."""

    def __init__(self, n_embd: int, n_expand: int, n_layer: int = 1, expander_type: str = 'linear'):
        super().__init__()
        self.n_embd = n_embd  # D
        self.n_expand = n_expand  # R or 
        assert expander_type in ['linear', 'mlp']
        if expander_type == 'linear':
            assert n_layer in (1, None), 'n_layer is only valid for MLP'
        self.n_layer = n_layer
        self.expander_type = expander_type
        if self.expander_type == 'linear':
            self.expander = LinearExpanderHead(self.n_embd, self.n_expand)
        elif self.expander_type == 'mlp':
            self.expander = MLPExpanderHead(self.n_embd, self.n_expand, self.n_layer)

    def forward(self, xx: Tensor) -> Tensor:
        return self.expander(xx)

    def reset_parameters(self):
        # TODO: Maybe better to just init all random but very small?
        self.expander.reset_parameters()


class TransformerEncoderHead(nn.Module):
    # Create custom parameterisation for each output token

    def __init__(self, n_embd: int, n_head: int = 6, n_layer: int = 2):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        assert n_layer >= 0
        self.transformer = nn.ModuleList(
            [Block(n_head, n_embd) for _ in range(self.n_layer)]
        )

    def forward(self, xx):
        # Batch, Sentence Length, Embed Dim
        # B, S, D = xx.shape

        # xx = F.rms_norm(xx, (xx.size(-1),))
        for block in self.transformer:
            xx = block(xx)
        xx = F.rms_norm(xx, (xx.size(-1),))
        return xx

    def reset_parameters(self):
        for each in self.transformer:
            each.reset_parameters()


class OutputHead(nn.Module):
    def __init__(self, encoder: TransformerEncoderHead | None, expander: ExpanderHead):
        super().__init__()
        self.encoder = encoder
        # Expands parametrisation for mixture model
        self.expander = expander

    def forward(self, xx: Tensor, generate: bool = False) -> Tensor:
        # xx is B, S, D

        # We can bypass the transformer encoder by setting it to have n_layer=0
        if self.encoder is not None and self.encoder.n_layer > 0:
            xx = self.encoder(xx)
        if generate:
            xx = xx[:, [-1]]

        xx = self.expander(xx)
        return xx

    def reset_parameters(self):
        if self.encoder is not None:
            self.encoder.reset_parameters()
        self.expander.reset_parameters()


class FoldOutputHead(nn.Module):
    def __init__(self, heads: list):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D)
        # Stack tensors of shape (B, S, R, D) to a tensor of shape (B, S, F, R, D)
        return torch.stack([h(xx) for h in self.heads], dim=2)


class MultiTokenHead(nn.Module):
    def __init__(
        self,
        config: ParametersConfig,
        vocab_size: int,
        *,
        n_embd: int = 768,
        n_head: int = 6,
        transformer_n_layer: int = 2,
        expander_n_layer: int = 2,
        expander_type: str = 'linear',
        freeze_vocab_unembedding: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.transformer_n_layer = transformer_n_layer
        self.expander_n_layer = expander_n_layer
        self.expander_type = expander_n_layer
        self.freeze_vocab_unembedding = freeze_vocab_unembedding

        # The shared unembedding matrix
        self.vocab_proj = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        # Potentially freeze unembedding weights
        for p in self.W.parameters():
            p.requires_grad = not self.freeze_vocab_unembedding

        # Instantiate as many folded output heads as needed by the circuit parameters configuration
        sum_weights_heads, sum_weights_projs = [], []
        categorical_log_probs_heads, categorical_log_probs_projs = [], []
        for shape in config.sum_weights_shapes:
            n_folds, n_output_units, n_input_units = shape
            heads = [OutputHead(
                TransformerEncoderHead(n_embd, n_head=n_head, n_layer=transformer_n_layer),
                ExpanderHead(n_embd, n_output_units, n_layer=expander_n_layer, expander_type=expander_type)
            ) for _ in range(n_folds)]
            proj = nn.Linear(self.n_embd, n_input_units, bias=False)
            sum_weights_heads.append(FoldOutputHead(heads))
            sum_weights_projs.append(proj)
        for shape in config.categorical_log_probs_shapes:
            n_folds, n_components, vocab_size = shape
            heads = [OutputHead(
                TransformerEncoderHead(n_embd, n_head=n_head, n_layer=transformer_n_layer),
                ExpanderHead(n_embd, n_components, n_layer=expander_n_layer, expander_type=expander_type)
            ) for _ in range(n_folds)]
            categorical_log_probs_heads.append(FoldOutputHead(heads))
            categorical_log_probs_projs.append(self.W)  # Share the same unembedding matrix for each token
        self._sum_weights_heads = nn.ModuleList(sum_weights_heads)
        self._categorical_log_probs_heads = nn.ModuleList(categorical_log_probs_heads)

    def set_unembedding_weights(self, weights):
        self.vocab_proj.weight.data = weights

    def forward(self, xx: Tensor, generate: bool = False) -> dict:
        # xx: (B, S, D)
        # Pass through the transformer encoder first, if required
        if self.encoder is not None:
            xx = self.encoder(xx)
        if generate:
            xx = xx[:, [-1]]

        # xx: (B, S, D) or (B, 1, D) if generate=True
        # Compute the parameters of the circuit
        sum_weights = []            # A list of tensors (B, S, F, K, J)
        categorical_log_probs = []  # A list of tensors (B, S, F, K, V)
        for sum_weight_fn, sum_weight_proj in \
            zip(self._sum_weights_heads, self._sum_weights_projs):
            # sw: (B, S, F, K, D) -> (B, S, F, K, J) after projection
            sw = sum_weight_fn(xx)    # Apply output head network
            sw = sum_weight_proj(sw)  # Linear projection
            sum_weights.append(sw)
        for categorical_log_probs_fn, categorical_log_probs_proj in \
            zip(self._categorical_log_probs_heads, self._categorical_log_probs_projs):
            # clp: (B, S, F, K, D) -> (B, S, F, K, V) after projection
            clp = categorical_log_probs_fn(xx)    # Apply output head network
            clp = categorical_log_probs_proj(xx)  # Linear projection
            categorical_log_probs.append(clp)

        return {
            'sum': sum_weights,
            'categorical': categorical_log_probs
        }
