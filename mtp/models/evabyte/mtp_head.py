from functools import cached_property
import math
import torch

from torch import Tensor
from torch import nn

from transformers.cache_utils import Cache

from mtp.models.circuits import ParametersConfig
from mtp.models.evabyte.configuration_evabyte import EvaByteConfig
from mtp.models.evabyte.eva_cache import EvaStaticCacheForTriton
from mtp.models.evabyte.modeling_evabyte import EvaByteDecoderLayer, EvaByteRMSNorm, EvaByteRotaryEmbedding, prepare_eva_generation_attn_mask_triton


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    # This is part of the Medusa model. However, here we vectorize it over an extra batch dimension on the parameters.
    # https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py
    """

    def __init__(
        self,
        n_fold: int,
        n_expand: int,
        hidden_size: int,
        output_size: int = None,
        use_skip: bool = True
    ):
        super().__init__()
        self.n_fold = n_fold
        self.n_expand = n_expand
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_skip = use_skip
        self.weight = nn.Parameter(torch.empty(n_fold, n_expand, hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(n_fold, n_expand, output_size))

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
        zz = self.act(zz + self.bias)
        if not self.use_skip:
            return zz
        return xx + zz


class LinearHead(nn.Module):
    def __init__(
        self,
        n_fold: int,
        n_expand: int,
        hidden_size: int,
        output_size: int
    ):
        super().__init__()
        self.n_fold = n_fold
        self.n_expand = n_expand  # R
        self.hidden_size = hidden_size  # D
        self.output_size = output_size
        
        # Instantiate the projection layer
        self.proj = nn.Parameter(torch.empty(n_fold, n_expand, output_size, hidden_size))

    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) -> (B, S, F, R, O)
        return torch.einsum('bsd,frod->bsfro', xx, self.proj)
        #return (xx @ self.proj.view(-1, self.hidden_size).T).view(xx.shape[0], xx.shape[1], self.n_fold, self.n_expand, self.output_size)


class MLPHead(nn.Module):
    def __init__(
        self,
        n_fold: int,
        n_expand: int,
        hidden_size: int,
        output_size: int,
        n_layer: int = 1,
        use_skip: bool = True,
    ):
        super().__init__()
        assert n_layer >= 1
        self.n_fold = n_fold            # F
        self.n_expand = n_expand        # e.g., R or Ko
        self.hidden_size = hidden_size  # D
        self.output_size = output_size  # e.g., V or Ki
        self.n_layer = n_layer
        self.use_skip = use_skip

        # Instantiate the MLPs with residual blocks
        self.mlp = nn.Sequential(*[
            ResBlock(n_fold, n_expand, hidden_size, use_skip=use_skip)
            for _ in range(self.n_layer)
        ])

        # Instantiate the projection layer
        self.proj = nn.Parameter(torch.empty(n_fold, n_expand, output_size, hidden_size))

    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) -> (B, S, 1, D) -> (B, S, F, R, D)
        xx = xx.unsqueeze(dim=-2)
        xx = self.mlp(xx)
        # xx: (B, S, F, R, D) -> (B, S, F, R, O)
        return torch.einsum('bsfrd,frod->bsfro', xx, self.proj)


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
        use_skip: bool = True,
        **kwargs
    ):
        super().__init__()
        if type == 'linear':
            self.head = LinearHead(
                n_fold,
                n_expand,
                hidden_size,
                output_size,
            )
        elif type == 'mlp':
            self.head = MLPHead(
                n_fold,
                n_expand,
                hidden_size,
                output_size,
                n_layer=n_layer,
                use_skip=use_skip,
            )
        else:
            raise NotImplementedError(f"Unknown expander layer type called '{type}'")
        
    def forward(self, xx: Tensor) -> Tensor:
        # xx: (B, S, D) -> (B, S, F, R, V) or (B, S, F, Ko, Ki)
        return self.head(xx)


class TransformerHead(nn.Module):
    def __init__(self, config: EvaByteConfig, n_layer: int = 1, layer_start_idx: int = 0):
        super().__init__()
        self._layers = nn.ModuleList([
            EvaByteDecoderLayer(config, layer_idx=layer_start_idx + i)
            for i in range(n_layer)
        ])
        self._norm = EvaByteRMSNorm(config)

    def forward(self, xx: Tensor, **kwargs):
        for layer in self._layers:
            outputs = layer(xx, **kwargs)
            xx = outputs[0]
        return self._norm(xx)


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
        expander_use_skip: bool = True,
        freeze_vocab_unembedding: bool = False,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.n_token = config.n_token
        self.n_embd = n_embd
        self.transformer_n_head = transformer_n_head
        self.transformer_n_layer = transformer_n_layer
        self.expander_type = expander_n_layer
        self.expander_n_layer = expander_n_layer

        # Evabyte transformers necessarily work with bfloat16
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)

        # Instantiate transformers to parameterize the token's Categorical and the sum weights of the circuit
        if transformer_n_layer > 0:
            self._rotary_emb = EvaByteRotaryEmbedding(
                self._evabyte_config.hidden_size // self._evabyte_config.num_attention_heads,
                max_position_embeddings=self._evabyte_config.max_position_embeddings,
                base=self._evabyte_config.rope_theta
            )
            if len(config._sum_weights_shapes) > 0:
                self._sum_transformer_head = TransformerHead(
                    self._evabyte_config,
                    transformer_n_layer,
                    layer_start_idx=0
                )
            if len(config.categorical_log_probs_shapes) > 0:
                self._tok_transformer_head = TransformerHead(
                    self._evabyte_config,
                    transformer_n_layer,
                    layer_start_idx=transformer_n_layer
                )
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
                use_skip=expander_use_skip
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
                use_skip=expander_use_skip
            )
            categorical_log_probs_heads.append(head)
        self._sum_weights_heads = nn.ModuleList(sum_weights_heads)
        self._categorical_log_probs_heads = nn.ModuleList(categorical_log_probs_heads)

        # Initialize the parameters
        self.reset_parameters()

        # Reset torch dtype to the previous one
        torch.set_default_dtype(prev_dtype)

        if freeze_vocab_unembedding:
            for tok_head in self._categorical_log_probs_heads:
                assert isinstance(tok_head, (LinearHead, MLPHead))
                tok_head.head.proj.requires_grad = False

    @cached_property
    def _evabyte_config(self) -> EvaByteConfig:
        # Taking some values from the Evabyte model on HF
        # and hoping for the best
        # https://huggingface.co/EvaByte/EvaByte/blob/main/config.json
        return EvaByteConfig(
            num_hidden_layers=32,
            hidden_size=self.n_embd,
            num_attention_heads=self.transformer_n_head,
            intermediate_size=int(self.n_embd * 2.6875),
            fp32_ln=False,
            fp32_skip_add=True,
            mixedp_attn=True,
            lazy_init=True,
            init_fn="v2",
            init_std=0.01275,
            initializer_range=0.01275,
            norm_add_unit_offset=True,
            max_position_embeddings=32768,
            chunk_size=16,
            window_size=2048,
            rms_norm_eps=1e-05,
        )

    def reset_parameters(self):
        @torch.no_grad()
        def _init_weights(module):
            if isinstance(module, (nn.Linear, ResBlock)):
                std = getattr(self._evabyte_config, "initializer_range", 0.02)
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                std = getattr(self._evabyte_config, "initializer_range", 0.02)
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, (LinearHead, MLPHead)):
                bound = getattr(self._evabyte_config, "initializer_range", 0.02)
                module.proj.data.uniform_(-bound, bound)

        for module in self.modules():
            _init_weights(module)

    @property
    def token_heads(self) -> list:
        return list(self._categorical_log_probs_heads)

    @property
    def sum_weight_heads(self) -> list:
        return list(self._sum_weights_heads)

    @torch.no_grad()
    def set_unembedding_weights(self, weights: Tensor):
        assert weights.shape[0] % self.config.vocab_size == 0 and weights.shape[1] == self.n_embd, f"{weights.shape}"
        weights = weights.view(-1, self.config.vocab_size, self.n_embd)
        for k, tok_head in enumerate(self._categorical_log_probs_heads):
            assert isinstance(tok_head.head, (LinearHead, MLPHead))
            n_folds, n_components, _ = self.config.categorical_log_probs_shapes[k]
            assert tok_head.head.proj.shape == (n_folds, n_components, weights.shape[1], weights.shape[2]), f"{tok_head.head.proj.shape.shape}"
            for j in range(n_folds):
                var_idx = self.config.categorical_layers[k].scope_idx[j].item()
                if not (0 <= var_idx < weights.shape[0]):
                    continue
                assert tok_head.head.proj[j].data.dtype == weights.dtype
                tok_head.head.proj[j].data.copy_(weights[var_idx])

    def _prepare_transformer_heads(
        self,
        xx: Tensor,
        use_cache: bool = False,
        attention_mask: Tensor = None,
        past_key_values: Cache = None,
        position_ids: Tensor = None,
        multibyte_decoding: bool = None,
    ) -> tuple:
        if use_cache and multibyte_decoding:
            raise ValueError("Multi-byte decoding with caching enabled and transformers head is not yet supported")

        batch_size, seq_len = xx.shape[0], xx.shape[1]
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        max_seq_length = past_seen_tokens + seq_len
        # Shamelessly copying preparation of Evabyte transformer's arguments from the Evabyte pre-trained model
        if (not self.training) and (not use_cache) and (not multibyte_decoding):
            # forward-only inference mode. 
            # We tweak use_cache to be True to reuse code for generation
            use_cache = True
            if position_ids is None:
                position_ids = torch.arange(0, seq_len, device=xx.device, dtype=int).unsqueeze(dim=0).expand(batch_size, -1)
        if position_ids is None:
            assert not use_cache, "during decoding we must explicitly pass position_ids to the model call"
            position_ids = torch.arange(past_seen_tokens, max_seq_length, device=xx.device, dtype=int).unsqueeze(dim=0).expand(batch_size, -1)

        # Prepare caches and causal masks if in inference mode
        if use_cache:
            if past_key_values is not None:
                assert isinstance(past_key_values, Cache)
            else:
                past_key_values = EvaStaticCacheForTriton(
                    xx.shape[0],
                    self._evabyte_config.num_attention_heads,
                    self._evabyte_config.window_size,
                    self._evabyte_config.hidden_size // self._evabyte_config.num_attention_heads,
                    2 * self.transformer_n_layer,
                    xx.dtype,
                    xx.device,
                )

        if not multibyte_decoding:
            if use_cache:
                causal_mask = prepare_eva_generation_attn_mask_triton(
                    xx,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    config=self._evabyte_config
                )
            else:
                assert self.training
                assert xx.shape[1] % self._evabyte_config.window_size == 0, "Training is only tested for sequences that are a multiple of window_size"
                causal_mask = attention_mask
        else:
            assert use_cache
            causal_mask = attention_mask

        # Compute rotary embeddings
        cos, sin = self._rotary_emb(xx, seq_len=max_seq_length)
        assert len(cos.shape) == 2, f"cos should be of shape (max_seq_len, head_dim), got {cos.shape} instead"
        assert sin.shape == cos.shape, f"sin should be of shape (max_seq_len, head_dim), got {sin.shape} instead"
        assert len(position_ids.shape) == 2, f"position_ids should be of 2D, got {position_ids.shape} instead"
        cos = cos[position_ids]
        sin = sin[position_ids]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return use_cache, causal_mask, past_key_values, position_ids, cos, sin

    def forward(
        self,
        xx: Tensor,
        use_cache: bool = False,
        attention_mask: Tensor = None,
        past_key_values: Cache = None,
        position_ids: Tensor = None,
        multibyte_decoding: bool = None,
        generate: bool = False,
    ) -> dict:
        # xx: (B, S, D)
        if self._sum_transformer_head is not None or self._tok_transformer_head is not None:
            use_cache, causal_mask, past_key_values, position_ids, cos, sin = self._prepare_transformer_heads(
                xx,
                use_cache=use_cache,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                multibyte_decoding=multibyte_decoding,
            )
        else:
            cos = sin = None

        # Paramterize the sum layer weights of the circuit
        sum_weights = []            # A list of tensors (B, S, F, Ko, Ki)
        if len(self._sum_weights_heads) > 0:
            if self._sum_transformer_head is not None:
                zz_sum = self._sum_transformer_head(
                    xx,
                    use_cache=use_cache,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    multibyte_decoding=multibyte_decoding,
                    cos=cos,
                    sin=sin, 
                )
            else:
                zz_sum = xx
            if generate:
                zz_sum = zz_sum[:, [-1]]
            for sum_weight_fn in self._sum_weights_heads:
                # sum_logits: (F, B, S, Ko, Ki)
                sum_logits = sum_weight_fn(zz_sum)
                sum_logits = sum_logits.float()
                sum_weights.append(
                    torch.softmax(sum_logits.permute(2, 0, 1, 3, 4), dim=-1)
                )

        # Parameterize the token Categoricals of the circuit
        categorical_log_probs = []  # A list of tensors (F, B, S, R, V)
        if len(self._categorical_log_probs_heads) > 0:
            if self._tok_transformer_head is not None:
                zz_tok = self._tok_transformer_head(
                    xx,
                    use_cache=use_cache,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    multibyte_decoding=multibyte_decoding,
                    cos=cos,
                    sin=sin, 
                )
            else:
                zz_tok = xx
            if generate:
                zz_tok = zz_tok[:, [-1]]
            for categorical_log_probs_fn in self._categorical_log_probs_heads:
                # categorical_logits: (F, B, S, R, V)
                categorical_logits = categorical_log_probs_fn(zz_tok)
                categorical_logits = categorical_logits.float()
                categorical_log_probs.append(
                    torch.log_softmax(categorical_logits.permute(2, 0, 1, 3, 4), dim=-1)
                )

        return dict(sum=sum_weights, categorical=categorical_log_probs, past_key_values=past_key_values)
