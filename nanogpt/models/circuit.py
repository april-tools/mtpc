import torch
import torch.nn.functional as F

from collections import namedtuple

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import SumLayer, CategoricalLayer
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.utils.scope import Scope
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.parameters import Parameter, TensorParameter, LogSoftmaxParameter
from cirkit.symbolic.initializers import NormalInitializer
from cirkit.templates import tensor_factorizations, utils

from .mlp import Block


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

    def __init__(self, n_embd, vocab_size, n_component=1, num_tokens=3):
        super().__init__()
        self.n_embd = n_embd             # D
        self.vocab_size = vocab_size           # V
        self.n_component = n_component   # R
        self.num_tokens = num_tokens           # H
        self.token_heads = torch.nn.ModuleList([
            TokenHead(encoder=TransformerEncoderHead(self.n_embd),
                      expander=LinearExpanderHead(self.n_embd, self.n_component))
            for i in range(self.num_tokens)
            ])
        # self.token_heads = torch.nn.ModuleList([
        #     TokenHead(encoder=TransformerEncoderHead(self.n_embd),
        #               expander=None)
        #     for i in range(self.num_tokens)
        #     ])
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
        # B, S, H, R, V
        return torch.stack(logits, dim=2)


def logit_mapper(circuit, logits):
    # TODO: Handle parametrising the mixture
    assert len(logits.shape) == 3
    # Add the channel dimension
    mapped = logits.unsqueeze(dim=2)
    del circuit.layers[0].logits
    circuit.layers[0].logits = lambda: mapped
    return circuit


def multi_token_mixture(n, h=1, r=1):
    # We use a mixture of r softmaxes to predict the token at each position, h_i
    # and then mix the distrubutions for each h_i to form
    # a joint distribution over the next h tokens
    #
    # n is the number of categories (vocabulary size)
    # h is the number of tokens in the future window we wish to predict
    # r is the number of mixture components (overparameterisation)
    shape = (r, 1, n)
    cats = [CategoricalLayer(scope=Scope([i]),
                             num_output_units=r,
                             num_channels=1,
                             num_categories=n,
                             probs=None,
                             logits_factory = lambda shape: Parameter.from_input(
                                TensorParameter(*shape, initializer=NormalInitializer()),
                             )
                             )
            for i in range(h)]
    out = SumLayer(num_input_units=r, num_output_units=1, arity=h)
    circ = Circuit(num_channels=1, layers=[*cats, out], in_layers={out: cats}, outputs=[out])

    # TODO: Below is simpler/cleaner to write as suggested by Lorenzo
    # We are factorizing a tensor that is n x h , where n is the vocab size
    # and h is the number of tokens we are predicting for
    # circ = tensor_factorizations.cp((n,) * h,
    #                                 rank=r,
    #                                 factor_param=Parameterization(activation='none'),
    #                                 weight_param=Parameterization(activation='none'))
    ctx = PipelineContext(
    backend='torch',      # Use the PyTorch backend
    # Specify the backend compilation flags next
    # ---- Specify how to evaluate sum and product layers ---- #
    semiring='lse-sum',   # In this case we use the numerically-stable 'lse-sum' semiring (R, +, *), i.e.,
                          # where: + is the log-sum-exp operation, and * is the sum operation.
    # -------------------------------------------------------- #
    )
    cc = ctx.compile(circ)
    # Remove default parameters
    # cc.layers[0].probs = None
    # cc.layers[0].logits = None
    return cc


class CircuitCP(torch.nn.Module):
    def __init__(self, vocab_size, n_token, n_component):
        super().__init__()
        self.vocab_size = vocab_size     # V
        self.n_token = n_token           # H
        self.n_component = n_component   # R
        self.symb_circuit = tensor_factorizations.cp((self.vocab_size, self.n_token),
                                                     rank=self.n_component,
                                                     factor_param=utils.Parameterization(activation='none'),
                                                     weight_param=utils.Parameterization(activation='none'))
        self._ctx = self._setup_pipeline_context()
        self.circuit = self._ctx.compile(self.symb_circuit)  # nn.Module

    def _setup_pipeline_context(self):
        ctx = PipelineContext(
                backend='torch',
                semiring='lse-sum')
        return ctx


class CircHead(torch.nn.Module):
    def __init__(self, circuit, logit_mapper):
        super().__init__()
        assert isinstance(circuit, TorchCircuit)
        # The compiled circuit
        self.circuit = circuit
        self.logit_mapper = logit_mapper
        
    def forward(self, xx, targets=None, return_logits=True):
        # xx should be of shape (BS, H, R, N)
        self.circuit = logit_mapper(self.circuit, xx)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # logits = 30 * torch.tanh(logits / 30)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            raise NotImplemented()

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss
