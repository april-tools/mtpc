import torch

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import SumLayer, CategoricalLayer
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.utils.scope import Scope
from cirkit.pipeline import compile


class TransformerExpanderHead(torch.nn.Module):
    # Expand parametrisation for mixture model

    def __init__(self, embed_dim, num_components, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim            # D
        self.num_components = num_components  # R
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.te = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=self.num_heads, batch_first=True)
        self.rf = torch.nn.TransformerEncoder(te, num_layers=self.num_layers)
        self.rep_pos_embeds = torch.nn.Embedding(self.num_components, self.embed_dim)

    def __call__(self, xx):
        # Batch, Embed Dim
        B, D = xx.shape
        xx = xx.repeat()
        pass


class LinearExpanderHead(torch.nn.Module):
    # Expand parametrisation for mixture model

    def __init__(self, embed_dim, num_components):
        super().__init__()
        self.embed_dim = embed_dim            # D
        self.num_components = num_components  # R
        # Below is equivalent to R square linear layers
        self.Wr = torch.nn.Parameter(torch.empty(self.num_components,
                                                 self.embed_dim,
                                                 self.embed_dim))
        torch.nn.init.normal_(self.Wr, mean=0.0, std=0.02)

    def __call__(self, xx):
        # Batch, Embed Dim
        B, D = xx.shape

        xx = xx @ self.Wr
        # xx is R, B, D
        xx = xx.permute(1, 0, 2)
        # xx is B, R, D
        return xx


class TransformerEncoderHead(torch.nn.Module):
    # Create custom parameterisation for each output token

    def __init__(self, embed_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.te = torch.nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, batch_first=True)
        self.rf = torch.nn.TransformerEncoder(te, num_layers=self.num_layers)
        # TODO: Do we want positional embeddings that differ based on head?

    def __call__(self, xx):
        # Batch, Sentence Length, Embed Dim
        B, S, D = xx.shape
        return self.rf(xx)


class TokenHead(torch.nn.Module):

    def __init__(self, encoder, expander):
        super().__init__()
        self.encoder = encoder
        # Expands parametrisation for mixture model
        self.expander = expander

    def __call__(self, xx):
        xx = self.encoder(xx)
        # Choose last token representation
        xx = xx[:, -1, :]
        xx = self.expander(xx)
        return xx


class MultiTokenHead(torch.nn.Module):

    def __init__(self, embed_dim, vocab_size, num_components=1, num_tokens=3):
        super().__init__()
        self.embed_dim = embed_dim             # D
        self.vocab_size = vocab_size           # V
        self.num_components = num_components   # R
        self.num_tokens = num_tokens           # H
        self.gelu = torch.nn.GELU()
        for i in range(self.num_tokens):
            setattr(self, 'token_head_%d' % i, TokenHead(encoder=TransformerEncoderHead(self.embed_dim),
                                                         expander=LinearExpanderHead(self.embed_dim, self.num_components)))
        # Unembedding matrix
        self.lm_head = torch.nn.Linear(self.embed_dim, self.vocab_size, bias=False)

    def __call__(self, xx):

        # xx is  B, S, D
        logits = []
        for i in range(self.num_tokens):
            # B, R, D
            head_xx = getattr(self, 'token_head_%d' % i)(xx)
            head_xx = self.gelu(head_xx)
            # B, R, V
            head_logits = self.lm_head(head_xx)
            logits.append(head_logits)
        # B, H, R, V
        return torch.stack(logits, dim=1)


def logit_mapper(circuit, logits):
    # TODO: Handle parametrising the mixture
    assert len(logits.shape) == 3
    # Add the channel dimension
    mapped = logits.unsqueeze(dim=2)
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
    cats = [CategoricalLayer(scope=Scope([i]), num_output_units=r, num_channels=1, num_categories=n)
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
    cc = compile(circ)
    # Remove default parameters
    cc.layers[0].probs = None
    cc.layers[0].logits = None
    return cc


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
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            # logits = 30 * torch.tanh(logits / 30)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss


if __name__ == "__main__":

    BS = 1       # Batch Size
    SL = 10
    D = 128      # Embed dim
    V = 10000    # Number of Tokens in Vocabulary (Categories)
    H = 1        # Number of Next Tokens in Future Window
    R = 1        # Number of Mixture Components
    cc = multi_token_mixture(V, H, R)

    te = torch.nn.TransformerEncoderLayer(d_model=D, nhead=4, batch_first=True)
    rf = torch.nn.TransformerEncoder(te, num_layers=2)
    mth = MultiTokenHead(D, V, num_components=R, num_tokens=H)

    xx = torch.randn(BS, SL, D)

    zz = rf(xx)
    logits = mth(zz)
    print(logits.shape)
    # NOTE: Do this for now
    logits = logits.squeeze(dim=0)

    # # TODO: Logits must have a batch dim
    cc = logit_mapper(cc, logits)

    yy = torch.multinomial(torch.arange(V, dtype=torch.float), num_samples=BS, replacement=True)
    # Each mixture component counts as an input so duplicate the inputs in that dimension
    yy = yy.unsqueeze(dim=1)
    # TODO: Below should be token window, not repetition
    yy = yy.repeat(1, H)

    # Add the channel component in position 1
    yy = yy.unsqueeze(dim=1)
    print(yy.shape)
    print(cc(yy))
