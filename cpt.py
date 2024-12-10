import torch

from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import SumLayer, CategoricalLayer
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.utils.scope import Scope
from cirkit.pipeline import compile


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
            logits = 30 * torch.tanh(logits / 30)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            logits = 30 * torch.tanh(logits / 30)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss


if __name__ == "__main__":

    BS = 1       # Batch Size
    N = 10000    # Number of Categories
    H = 4        # Number of Next Tokens in Future Window
    R = 2        # Number of Mixture Components
    cc = multi_token_mixture(N, H, R)

    # TODO: Logits must have a batch dim
    logits = torch.zeros(H, R, N)
    cc = logit_mapper(cc, logits)

    yy = torch.multinomial(torch.arange(N, dtype=torch.float), num_samples=BS, replacement=True)
    # Each mixture component counts as an input so duplicate the inputs in that dimension
    yy = yy.unsqueeze(dim=1)
    # TODO: Below should be token window, not repetition
    yy = yy.repeat(1, H)

    # Add the channel component in position 1
    yy = yy.unsqueeze(dim=1)
    print(yy.shape)
    print(cc(yy))
