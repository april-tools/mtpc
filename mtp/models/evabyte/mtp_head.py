import math
import torch

from torch import Tensor
from torch import nn

from transformers.cache_utils import Cache

from mtp.models.circuits import ParametersConfig
from mtp.models.evabyte.configuration_evabyte import EvaByteConfig
from mtp.models.evabyte.modeling_evabyte import EvaByteDecoderLayer, EvaByteRotaryEmbedding


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    # This is part of the Medusa model. However, here we vectorize it over an extra batch dimension on the parameters.
    # https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py
    """

    def __init__(self, n_fold: int, n_expand: int, hidden_size: int, output_size: int = None):
        super().__init__()
        self.n_fold = n_fold
        self.n_expand = n_expand
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(n_fold, n_expand, hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(n_fold, n_expand, output_size))

        # Initialize based on default initialization of nn.Linear
        with torch.no_grad():
            torch.vmap(
                torch.vmap(torch.nn.init.kaiming_uniform_, randomness='different'),
                randomness='different'
            )(self.proj)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[0, 0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

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
        # xx: (B, S, F, R, D)
        # zz: (B, S, F, R, D)
        zz = torch.einsum('frcd,bsfrd->bsfrc', self.weight, xx)
        return xx + self.act(zz + self.bias)


class LinearHead(nn.Module):
    def __init__(
        self,
        n_fold: int,
        n_expand: int,
        hidden_size: int,
        output_size: int,
        share_proj: bool = False
    ):
        super().__init__()
        self.n_fold = n_fold
        self.n_expand = n_expand  # R
        self.hidden_size = hidden_size  # D
        self.output_size = output_size
        
        # Instantiate the projection layer
        n_expand_proj = 1 if share_proj else n_expand
        self.proj = nn.Parameter(torch.empty(n_fold, n_expand_proj, output_size, hidden_size))
        # Initialize based on default initialization of nn.Linear
        with torch.no_grad():
            torch.vmap(
                torch.vmap(torch.nn.init.kaiming_uniform_, randomness='different'),
                randomness='different'
            )(self.proj)

    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) -> (B, S, F, R, O)
        return torch.einsum('frod,bsd->bsfro', self.proj, xx)


class MLPHead(nn.Module):
    def __init__(
        self,
        n_fold: int,
        n_expand: int,
        hidden_size: int,
        output_size: int,
        n_layer: int = 1,
        share_proj: bool = False
    ):
        super().__init__()
        assert n_layer >= 1
        self.n_fold = n_fold            # F
        self.n_expand = n_expand        # e.g., R or Ko
        self.hidden_size = hidden_size  # D
        self.output_size = output_size  # e.g., V or Ki
        self.n_layer = n_layer

        # Instantiate the MLPs with residual blocks
        self.mlp = nn.Sequential(*[ResBlock(n_expand, hidden_size) for _ in range(self.n_layer)])

        # Instantiate the projection layer
        n_fold_proj = 1 if share_proj else n_fold
        n_expand_proj = 1 if share_proj else n_expand
        self.proj = nn.Parameter(torch.empty(n_fold_proj, n_expand_proj, output_size, hidden_size))
        with torch.no_grad():
            torch.vmap(
                torch.vmap(torch.nn.init.kaiming_uniform_, randomness='different'),
                randomness='different'
            )(self.proj)

    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) -> (B, S, 1, D) -> (B, S, F, R, D)
        xx = xx.unsqueeze(dim=-2)
        xx = self.mlp(xx)
        # xx: (B, S, F, R, D) -> (B, S, F, R, O)
        return torch.einsum('frod,bsfrd->bsfro', self.proj, xx)


class ExpanderHead(nn.Module):
    def __init__(
        self,
        *,
        n_fold: int,
        n_expand: int,
        hidden_size: int,
        output_size: int,
        type: str = 'linear',
        n_layer: int = 1,
        share_proj: bool = False,
        **kwargs
    ):
        super().__init__()
        if type == 'linear':
            self.head = LinearHead(
                n_fold,
                n_expand,
                hidden_size,
                output_size,
                share_proj=share_proj
            )
        elif type == 'mlp':
            self.head = MLPHead(
                n_fold,
                n_expand,
                hidden_size,
                output_size,
                n_layer=n_layer,
                share_proj=share_proj
            )
        else:
            raise NotImplementedError(f"Unknown expander layer type called '{type}'")
        
    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) -> (B, S, F, R, V) or (B, S, F, Ko, Ki)
        return self.head(xx)


class TransformerHead(nn.Module):
    def __init__(self, config: EvaByteConfig, n_layer: int = 1):
        super().__init__()
        # Evabyte transformers necessarily work with bfloat16
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self._layers = nn.ModuleList([
            EvaByteDecoderLayer(config)
            for _ in range(n_layer)
   	    ])
        torch.set_default_dtype(prev_dtype)

    def forward(self, xx: Tensor, **kwargs):
        for layer in self._layers:
            xx, = layer(xx, **kwargs)
        return xx


class MultiTokenHead(nn.Module):
    def __init__(
        self,
        config: ParametersConfig,
        vocab_size: int,
        *,
        n_embd: int = 4096,
        transformer_n_head: int = 32,
        transformer_n_layer: int = 1,
        expander_type: str = 'linear',
        expander_n_layer: int = 2,
        freeze_vocab_unembedding: bool = False,
        share_vocab_proj: bool = False
    ):
        if freeze_vocab_unembedding:
            raise NotImplementedError()
        super().__init__()
        self.vocab_size = vocab_size
        self.n_token = config.n_token
        self.n_embd = n_embd
        self.transformer_n_head = transformer_n_head
        self.transformer_n_layer = transformer_n_layer
        self.expander_type = expander_n_layer
        self.expander_n_layer = expander_n_layer
        self.share_vocab_proj = share_vocab_proj

        # Instantiate transformers to parameterize the token's Categorical and the sum weights of the circuit
        if transformer_n_layer > 0:
            # Taking some values from the Evabyte model on HF
            # and hoping for the best
            # https://huggingface.co/EvaByte/EvaByte/blob/main/config.json
            transformer_config = EvaByteConfig(
                hidden_size=n_embd,
                num_attention_heads=transformer_n_head,
                intermediate_size=int(n_embd * 2.6875),
                fp32_ln=False,
                fp32_skip_add=True,
                mixedp_attn=True,
                lazy_init=True,
                init_fn="v2",
                init_std=0.01275,
                initializer_range=0.01275,
                max_position_embeddings=32768,
                chunk_size=16,
                window_size=2048,
                rms_norm_eps=1e-05,
            )
            self._rotary_emb = EvaByteRotaryEmbedding(
                transformer_config.hidden_size // transformer_config.num_attention_heads,
                max_position_embeddings=transformer_config.max_position_embeddings,
                base=transformer_config.rope_theta
            )
            self._tok_transformer_head = TransformerHead(transformer_config, transformer_n_layer)
            self._sum_transformer_head = TransformerHead(transformer_config, transformer_n_layer)
        else:
            self._rotary_emb = None
            self._tok_transformer_head = None
            self._sum_transformer_head = None

        # Instantiate as many expander heads as needed by the circuit parameters configuration
        sum_weights_heads = []
        categorical_log_probs_heads = []
        for k, shape in enumerate(config.sum_weights_shapes):
            n_folds, n_output_units, n_input_units = shape
            head = ExpanderHead(
                type=expander_type,
                n_fold=n_folds,
                n_expand=n_output_units,
                hidden_size=n_embd,
                output_size=n_input_units,
                n_layer=expander_n_layer,
                share_proj=False
            )
            sum_weights_heads.append(head)
        for k, shape in enumerate(config.categorical_log_probs_shapes):
            n_folds, n_components, vocab_size = shape
            head = ExpanderHead(
                type=expander_type,
                n_fold=n_folds,
                n_expand=n_components,
                hidden_size=n_embd,
                output_size=vocab_size,
                n_layer=expander_n_layer,
                share_proj=share_vocab_proj
            )
            categorical_log_probs_heads.append(head)
        self._sum_weights_heads = nn.ModuleList(sum_weights_heads)
        self._categorical_log_probs_heads = nn.ModuleList(categorical_log_probs_heads)

    @property
    def token_heads(self) -> list:
        return list(self._categorical_log_probs_heads)

    @property
    def sum_weight_heads(self) -> list:
        return list(self._sum_weights_heads)

    def forward(
        self,
        xx: Tensor,
        attention_mask: Tensor = None,
        past_key_values: Cache = None,
        position_ids: Tensor = None,
        generate: bool = False,
    ) -> dict:
        # xx: (B, S, D)

        if self._rotary_emb is not None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            max_seq_length = past_seen_tokens + xx.shape[1]
            cos, sin = self._rotary_emb(xx, max_seq_length)
            if position_ids is None:
                position_ids = torch.arange(past_seen_tokens, max_seq_length, device=xx.device, dtype=int).unsqueeze(dim=0).expand(xx.shape[0], -1)
            assert len(cos.shape) == 2, f"cos should be of shape (max_seq_len, head_dim), got {cos.shape} instead"
            assert sin.shape == cos.shape, f"sin should be of shape (max_seq_len, head_dim), got {sin.shape} instead"
            assert len(position_ids.shape) == 2, f"position_ids should be of 2D, got {position_ids.shape} instead"
            cos = cos[position_ids]
            sin = sin[position_ids]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            zz_sum = self._sum_transformer_head(
                xx, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_values, cos=cos, sin=sin, 
            )
            zz_tok = self._tok_transformer_head(
                xx, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_values, cos=cos, sin=sin, 
            )
        else:
            zz_sum = xx
            zz_tok = xx

        # Paramterize the sum layer weights of the circuit
        if generate:
            zz_sum = zz_sum[:, [-1]]
        else:
            zz_sum = zz_sum[:, :zz_sum.shape[1] - self.n_token + 1]
        sum_weights = []            # A list of tensors (B, S, F, Ko, Ki)
        for sum_weight_fn in self._sum_weights_heads:
            # sum_logits: (F, B, S, Ko, Ki)
            sum_logits = sum_weight_fn(zz_sum)
            sum_weights.append(
                torch.softmax(sum_logits.permute(2, 0, 1, 3, 4), dim=-1)
            )

        # Parameterize the token Categoricals of the circuit
        if generate:
            zz_tok = zz_tok[:, [-1]]
        else:
            zz_tok = zz_tok[:, :zz_tok.shape[1] - self.n_token + 1]
        categorical_log_probs = []  # A list of tensors (F, B, S, R, V)
        for categorical_log_probs_fn in self._categorical_log_probs_heads:
            # categorical_logits: (F, B, S, R, V)
            categorical_logits = categorical_log_probs_fn(zz_tok)
            categorical_log_probs.append(
                torch.log_softmax(categorical_logits.permute(2, 0, 1, 3, 4), dim=-1)
            )

        return {
            'sum': sum_weights,
            'categorical': categorical_log_probs
        }
