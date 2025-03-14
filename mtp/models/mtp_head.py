import torch
import torch.nn.functional as F

from torch import Tensor
from torch import nn

from mtp.models.circuits import ParametersConfig
from .mlp import Block


class ResBlock(torch.nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    #TODO:
    # This is part of the Medusa model. However, here we vectorize it over an extra batch dimension on the parameters.
    # https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py

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
        self.n_embd = n_embd  # D
        self.n_component = n_component  # R
        # Below is equivalent to R square linear layers
        self.Wr = torch.nn.Parameter(torch.zeros(self.n_component,
                                                 self.n_embd,
                                                 self.n_embd))
        self.reset_parameters()

    def forward(self, xx: Tensor) -> Tensor:
        # xx is B x S x R x D
        xx = torch.einsum('bsc,rcd->bsrd', xx, self.Wr)

        return xx

    def reset_parameters(self):
        # At the beginning of training we want the linear layer to not
        # change the logits - so we want an identity matrix
        eye = torch.eye(self.n_embd, device=self.Wr.device)
        # Expand to R x n_embd x n_embd
        eye = eye.unsqueeze(0).repeat(self.n_component, 1, 1)
        self.Wr.data = eye


class MLPExpanderHead(torch.nn.Module):
    # Expand parametrisation for mixture model

    def __init__(self, n_embd: int, n_component: int, n_layer: int = 1):
        super().__init__()
        assert n_layer >= 1
        self.n_embd = n_embd  # D
        self.n_component = n_component  # R
        self.n_layer = n_layer
        self.mlps = torch.nn.ModuleList(
            [torch.nn.Sequential(*([ResBlock(self.n_embd)] * self.n_layer))
            for _ in range(self.n_component)]
        )

    def forward(self, xx: Tensor) -> Tensor:
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

    def __init__(self, n_embd: int, n_component: int, n_layer: int = 1, expander_type='linear'):
        super().__init__()
        self.n_embd = n_embd  # D
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

        # xx = F.rms_norm(xx, (xx.size(-1),))
        for block in self.transformer:
            xx = block(xx)
        xx = F.rms_norm(xx, (xx.size(-1),))
        return xx

    def reset_parameters(self):
        for each in self.transformer:
            each.reset_parameters()


class MultiTokenHead(torch.nn.Module):
    def __init__(
        self,
        config: ParametersConfig,
        *,
        n_embd: int = 768,
        n_head: int = 6,
        transformer_n_layer: int = 2,
        expander_n_layer: int = 2,
        expander_type: str = 'linear',
        freeze_unembedding: bool = False
    ):
        super().__init__()
        self.encoder = TransformerEncoderHead(n_embd, n_head=n_head, n_layer=transformer_n_layer)
        self.freeze_unembedding = freeze_unembedding

        # Instantiate as many output head as needed by the circuit parameters configuration
        sum_weights_heads = []
        categorical_log_probs_heads = []
        for shape in config.sum_weights_shapes:
            # n_folds, n_components, vocab_size = shape
            sum_weights_heads.append(
                ExpanderHead(n_embd, n_layer=expander_n_layer, expander_type=expander_type, shape=shape)
            )
        for shape in config.categorical_log_probs_shapes:
            # n_folds, n_output_units, n_input_units = shape
            categorical_log_probs_heads.append(
                ExpanderHead(n_embd, n_layer=expander_n_layer, expander_type=expander_type, shape=shape)
            )
        self._sum_weights_heads = nn.ModuleList(sum_weights_heads)
        self._categorical_log_probs_heads = nn.ModuleList(categorical_log_probs_heads)

        # The shared unembedding matrix
        self.W = torch.nn.Linear(self.n_embd, self.vocab_size, bias=False)
        # Potentially freeze unembedding weights
        for p in self.W.parameters():
            p.requires_grad = not self.freeze_unembedding

    def set_unembedding_weights(self, weights):
        self.W.weight.data = weights

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
        for sum_weight_fn in self._sum_weights_heads:
            sum_weights.append(sum_weight_fn(xx))
        for categorical_log_probs_fn in self._categorical_log_probs_heads:
            categorical_log_probs.append(categorical_log_probs_fn(xx))

        return {
            'sum': sum_weights,
            'categorical': categorical_log_probs
        }
