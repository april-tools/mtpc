import torch
import torch.nn.functional as F

from collections import namedtuple

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import SumLayer, CategoricalLayer
from cirkit.utils.scope import Scope
from cirkit.templates import tensor_factorizations, utils

from .mlp import Block
from .pipeline import setup_pipeline_context


class TransformerExpanderHead(torch.nn.Module):
    # Expand parametrisation for mixture model

    def __init__(self, n_embd, n_component, num_heads=4, num_layers=2):
        super().__init__()
        self.n_embd = n_embd            # D
        self.n_component = n_component  # R
        self.num_heads = num_heads
        self.num_layers = num_layers

        # NOTE: Below need not be causal - since over "R" dimension
        te = torch.nn.TransformerEncoderLayer(d_model=n_embd, nhead=self.num_heads, batch_first=True)
        self.rf = torch.nn.TransformerEncoder(te, num_layers=self.num_layers)
        self.rep_pos_embeds = torch.nn.Embedding(self.n_component, self.n_embd)

    def forward(self, xx):
        # Batch, Embed Dim
        B, D = xx.shape
        pass


class LinearExpanderHead(torch.nn.Module):
    # Expand parametrisation for mixture model

    def __init__(self, n_embd, n_component):
        super().__init__()
        self.n_embd = n_embd            # D
        self.n_component = n_component  # R
        self.gelu = torch.nn.GELU()
        # Below is equivalent to R square linear layers
        self.Wr = torch.nn.Parameter(torch.zeros(self.n_component,
                                                 self.n_embd,
                                                 self.n_embd))
        torch.nn.init.normal_(self.Wr, mean=0.0, std=0.02)

    def forward(self, xx):
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


class TransformerEncoderHead(torch.nn.Module):
    # Create custom parameterisation for each output token

    def __init__(self, n_embd, num_heads=6, num_layers=2):
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.num_layers = num_layers

        config = namedtuple('opts', ['n_embd', 'n_head'])(self.n_embd, self.num_heads)
        self.transformer = torch.nn.ModuleList([Block(config) for _ in range(self.num_layers)])

    def forward(self, xx):
        # Batch, Sentence Length, Embed Dim
        B, S, D = xx.shape

        xx = F.rms_norm(xx, (xx.size(-1),))
        for block in self.transformer:
            xx = block(xx)
        xx = F.rms_norm(xx, (xx.size(-1),))
        return xx


class TokenHead(torch.nn.Module):

    def __init__(self, encoder, expander):
        super().__init__()
        self.encoder = encoder
        # Expands parametrisation for mixture model
        self.expander = expander

    def forward(self, xx):
        # xx is B, S, D
        xx = self.encoder(xx)
        # xx is B, S, D
        if self.expander is not None:
            xx = self.expander(xx)
        else:
            xx = xx.unsqueeze(dim=2)
        # xx is B, S, R, D
        return xx


class MultiTokenHead(torch.nn.Module):

    def __init__(self, vocab_size, n_embd, n_component=1, n_token=3):
        super().__init__()
        self.vocab_size = vocab_size           # V
        self.n_embd = n_embd                   # D
        self.n_component = n_component         # R
        self.n_token = n_token                 # H
        self.token_heads = torch.nn.ModuleList([
            TokenHead(encoder=TransformerEncoderHead(self.n_embd),
                      expander=LinearExpanderHead(self.n_embd, self.n_component))
            for i in range(self.n_token)
            ])
        # Unembedding matrix
        self.W = torch.nn.Linear(self.n_embd, self.vocab_size, bias=False)

    def forward(self, xx):

        # xx is  B, S, D
        logits = []
        # logits.append(self.W(xx))
        # TODO: Can we avoid the for loop with torch vmap?
        for token_head in self.token_heads:
            # head_xx is B, S, R, D
            head_xx = token_head(xx)
            # head_logits is  B, S, R, V
            head_logits = self.W(head_xx)
            logits.append(head_logits)
        categoricals = torch.stack(logits, dim=0)
        H, B, S, R, V = categoricals.shape
        # H, B * S, R, V
        categoricals = categoricals.reshape(H, B*S, R, V)
        return dict(categoricals=categoricals)


class CircuitCP(torch.nn.Module):
    def __init__(self, vocab_size, n_token, n_component):
        super().__init__()
        self.vocab_size = vocab_size     # V
        self.n_token = n_token           # H
        self.n_component = n_component   # R
        if self.n_token > 1:
            self.symb_circuit = tensor_factorizations.cp((self.vocab_size,) * self.n_token,
                                                         rank=self.n_component,
                                                         factor_param=utils.Parameterization(activation='none'),
                                                         weight_param=utils.Parameterization(activation='softmax'))
        else:
            cat = CategoricalLayer(scope=Scope([0]),
                                   num_output_units=self.n_component,
                                   num_channels=1,
                                   num_categories=self.vocab_size,
                                   # logits=None,
                                   # probs_factory = lambda shape: Parameter.from_input(
                                   #     TensorParameter(*shape, initializer=NormalInitializer()),
                                   #     )
                                   )
            out = SumLayer(num_input_units=self.n_component, num_output_units=1, arity=1)
            self.symb_circuit = Circuit(num_channels=1, layers=[cat, out], in_layers={out: [cat]}, outputs=[out])

        self._ctx: PipelineContext = setup_pipeline_context()
        self._circuit: TorchCircuit = self._ctx.compile(self.symb_circuit)

    @property
    def circuit(self) -> TorchCircuit:
        return self._circuit

    def forward(self, yy):
        return self._circuit(yy)
