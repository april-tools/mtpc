import math
import torch
import torch.nn.functional as F

from torch import Tensor
from torch import nn

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

    def __init__(self, n_component: int, hidden_size: int, init: str = 'identity'):
        super().__init__()
        self.n_component = n_component
        self.hidden_size = hidden_size
        self.init = init
        self.weight = nn.Parameter(torch.empty(n_component, hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(n_component, hidden_size))

        if self.init == 'identity':
            # Initialize the weight tensors as identity mapping (because of residual)
            torch.nn.init.zeros_(self.weight)
        elif self.init == 'uniform':
            # This is currently using the default initialization of nn.Linear
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            raise ValueError('init must be one of "identity" or "uniform", got %s' % init)

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
        # If we initialise the model to be the identity, use the residual
        if self.init == 'identity':
            return xx + self.act(zz + self.bias)
        else:
            return self.act(zz + self.bias)

    def _init_bias(self):
        # Initialize the bias matrix
        # This is currently using the default initialization of nn.Linear
        for i in range(self.n_component):
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias[i], -bound, bound)


class MLPExpanderHead(nn.Module):
    def __init__(self, n_embd: int, n_expand: int, n_layer: int = 1, init: str = 'identity'):
        super().__init__()
        assert n_layer >= 1
        self.n_embd = n_embd  # D
        self.n_expand = n_expand  # R
        self.n_layer = n_layer
        self.init = init
        if self.init == 'identity':
            self.mlp = nn.Sequential(
                *[ResBlock(n_expand, self.n_embd, 'identity') for _ in range(self.n_layer)]
            )
        elif self.init == 'uniform':
            # Make last layer uniform
            self.mlp = nn.Sequential(
                *[ResBlock(n_expand, self.n_embd, 'identity')
                    if i < (self.n_layer - 1)
                    else ResBlock(n_expand, self.n_embd, 'uniform') 
                  for i in range(self.n_layer)]
            )

    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) -> (B, S, 1, D) -> (B, S, R, D)
        xx = xx.unsqueeze(dim=2)
        xx = self.mlp(xx)
        return xx


class LinearExpanderHead(nn.Module):
    def __init__(self, n_embd: int, n_expand: int, init: str = 'identity'):
        super().__init__()
        self.n_embd = n_embd  # D
        self.n_expand = n_expand  # e.g., the number of mixture components R
        self.init = init
        # Below is equivalent to R square linear layers
        self.Wr = nn.Parameter(
            torch.zeros(self.n_expand, self.n_embd, self.n_embd)
        )
        if self.init == 'identity':
            # At initialisation for the first token, we want the linear layer
            # to leave the logits unchanged - so we want an identity matrix
            eye = torch.eye(self.n_embd, device=self.Wr.device)
            # Expand to R x n_embd x n_embd
            eye = eye.unsqueeze(0).repeat(self.n_expand, 1, 1)
            self.Wr.data = eye
        elif self.init == 'uniform':
            # We want a uniform distribution, so use uniform
            # This is the default initialisation in pytorch:
            # https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/linear.py#L122
            torch.nn.init.kaiming_uniform_(self.Wr, a=math.sqrt(5))
        else:
            raise ValueError('Unknown init %s' % self.init)

    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) or more in general (..., D)
        xx = torch.einsum('...d,rdc->...rc', xx, self.Wr)
        return xx


class ExpanderHead(nn.Module):
    """Wrapper class of expanders to make running from config easier."""

    def __init__(self, n_embd: int, n_expand: int, n_layer: int = 1, expander_type: str = 'linear', init: str = 'identity'):
        super().__init__()
        self.n_embd = n_embd  # D
        self.n_expand = n_expand  # R or 
        assert expander_type in ['linear', 'mlp']
        if expander_type == 'linear':
            assert n_layer in (1, None), 'n_layer is only valid for MLP'
        self.n_layer = n_layer
        self.expander_type = expander_type
        self.init = init
        if self.init not in ('uniform', 'identity'):
            raise ValueError('init should be "uniform" or "identity", got "%s"' % self.init)
        if self.expander_type == 'linear':
            self.expander = LinearExpanderHead(self.n_embd, self.n_expand, self.init)
        elif self.expander_type == 'mlp':
            self.expander = MLPExpanderHead(self.n_embd, self.n_expand, self.n_layer, self.init)

    def forward(self, xx: Tensor) -> Tensor:
        return self.expander(xx)


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
    def __init__(self, heads: list, *, proj: nn.Linear):
        super().__init__()
        self.heads = nn.ModuleList(heads)
        self.proj = proj

    def forward(self, xx: Tensor, generate: bool = False) -> Tensor:
        # xx: (B, S, D)
        # Stack tensors of shape (B, S, K, D) to a tensor of shape (F, B, S, K, D)
        xxs = [h(xx, generate=generate) for h in self.heads]
        xx = torch.stack(xxs, dim=0)
        # output after projection: (B, S, F, K, C), e.g., C = V
        # Note that S = 1 if generate is True
        return self.proj(xx)


class MultiTokenHead(nn.Module):
    def __init__(
        self,
        config: ParametersConfig,
        vocab_size: int,
        *,
        n_embd: int = 768,
        transformer_n_head: int = 6,
        tok_transformer_n_layer: int = 2,
        sum_transformer_n_layer: int = 2,
        expander_n_layer: int = 2,
        expander_type: str = 'linear',
        freeze_vocab_unembedding: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.transformer_n_head = transformer_n_head
        self.tok_transformer_n_layer = tok_transformer_n_layer
        self.sum_transformer_n_layer = sum_transformer_n_layer
        self.expander_n_layer = expander_n_layer
        self.expander_type = expander_n_layer
        self.freeze_vocab_unembedding = freeze_vocab_unembedding

        # The shared unembedding matrix
        self.vocab_proj = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        # Potentially freeze unembedding weights
        for p in self.vocab_proj.parameters():
            p.requires_grad = not self.freeze_vocab_unembedding

        # Instantiate as many folded output heads as needed by the circuit parameters configuration
        sum_weights_heads = []
        categorical_log_probs_heads = []
        for k, shape in enumerate(config.sum_weights_shapes):
            n_folds, n_output_units, n_input_units = shape
            heads = [OutputHead(
                TransformerEncoderHead(n_embd, n_head=transformer_n_head, n_layer=sum_transformer_n_layer),
                ExpanderHead(n_embd, n_output_units, n_layer=expander_n_layer, expander_type=expander_type, init='uniform')
            ) for _ in range(n_folds)]
            proj = nn.Linear(self.n_embd, n_input_units, bias=False)
            sum_weights_heads.append(FoldOutputHead(heads, proj=proj))
        for k, shape in enumerate(config.categorical_log_probs_shapes):
            n_folds, n_components, vocab_size = shape
            # We make the first token head init as an identity function (to match teacher)
            # while the remaining heads are initialised to produce a uniform distribution
            get_init = lambda j: 'identity' if config.categorical_layers[k].scope_idx[j].item() == 0 else 'uniform'
            heads = [OutputHead(
                TransformerEncoderHead(n_embd, n_head=transformer_n_head, n_layer=tok_transformer_n_layer),
                ExpanderHead(n_embd, n_components, n_layer=expander_n_layer, expander_type=expander_type, init=get_init(i))
            ) for i in range(n_folds)]
            # Share the same unembedding matrix for each token
            categorical_log_probs_heads.append(FoldOutputHead(heads, proj=self.vocab_proj))
        self._sum_weights_heads = nn.ModuleList(sum_weights_heads)
        self._categorical_log_probs_heads = nn.ModuleList(categorical_log_probs_heads)

    @property
    def token_heads(self) -> list:
        return list(self._categorical_log_probs_heads)

    @property
    def sum_weight_heads(self) -> list:
        return list(self._sum_weights_heads)

    def set_unembedding_weights(self, weights):
        self.vocab_proj.weight.data = weights

    def forward(self, xx: Tensor, generate: bool = False) -> dict:
        # xx: (B, S, D)
        # Compute the parameters of the circuit
        sum_weights = []            # A list of tensors (B, S, F, K, J)
        categorical_log_probs = []  # A list of tensors (B, S, F, K, V)
        for sum_weight_fn in self._sum_weights_heads:
            # sw: (B, S, F, K, J)
            sum_logits = sum_weight_fn(xx, generate=generate)
            sw = torch.softmax(sum_logits, dim=-1)
            sum_weights.append(sw)
        for categorical_log_probs_fn in self._categorical_log_probs_heads:
            # clp: (B, S, F, K, V)
            categorical_logits = categorical_log_probs_fn(xx, generate=generate)
            clp = torch.log_softmax(categorical_logits, dim=-1)
            categorical_log_probs.append(clp)

        return {
            'sum': sum_weights,
            'categorical': categorical_log_probs
        }
