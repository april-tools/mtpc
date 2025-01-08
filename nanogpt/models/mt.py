import torch

from torch import Tensor

from nanogpt.models.circuit import CircuitCP
from nanogpt.models.layers import TorchBatchedCategoricalLayer, TorchBatchedSumLayer


class MultiTokenLM(torch.nn.Module):
    """ A MultiTokenLM comprises three parts:

    1. A LM encoder, which can be the encoder (i.e. arch without lm_head)
    of any pretrained LLM. The encoder provides contextual embeddings for
    tokens.

    2. A lm_head, which expands the contextual embeddings into parameters
    for the (circuit) output layer.

    3. A circuit which models the output tokens and encodes their dependencies.
    """
    def __init__(self, lm_encoder: torch.nn.Module, lm_head: torch.nn.Module, circuit: CircuitCP):
        super().__init__()
        self.lm_encoder = lm_encoder
        self.lm_head = lm_head
        self.circuit = circuit

        # Retrieve the circuit layers to parameterize
        layers = list(self.circuit.circuit.topological_ordering())
        self._cat_layer: TorchBatchedCategoricalLayer = layers[0]
        assert isinstance(self._cat_layer, TorchBatchedCategoricalLayer)
        self._sum_layer: TorchBatchedSumLayer = layers[2]
        assert isinstance(self._sum_layer, TorchBatchedSumLayer)

    def forward(self, xx: Tensor, yy: Tensor, return_logits: bool = False):
        # Compute sliding windows indices
        # from yy: (B, S) to yy: (B, S', H)
        # where S' = S - H + 1
        yy = yy.unfold(dimension=1, size=self.lm_head.n_token, step=1)
        # Note that we unsqueeze a channel dimension, as required by cirkit
        # yy: (B, S', H) -> (B * S', 1, H)
        yy = yy.reshape(-1, 1, yy.shape[2])

        # xx: (B, S, D)
        xx = self.lm_encoder(xx)

        # Obtain dict of circuit parameters
        circuit_params = self.lm_head(xx)
        # cat_logits: (H, B, S', R, V) -> (H, B * S', R, V)
        cat_log_probs = circuit_params['cat_log_probs']
        cat_log_probs = cat_log_probs.view(cat_log_probs.shape[0], -1, cat_log_probs.shape[3], cat_log_probs.shape[4])
        self._cat_layer.log_probs = cat_log_probs
        # sum_weight: (B, S', 1, R) -> (1, B * S', 1, R)
        sum_weight = circuit_params['sum_weight']
        sum_weight = sum_weight.view(1, -1, 1, sum_weight.shape[3])
        self._sum_layer.weight = sum_weight

        # Compute the conditional log-likelihoods
        # yy: (B * S', 1, H)
        # log_probs: (B * S', 1, 1)
        log_probs = self.circuit(yy)

        # The loss is the negated average conditional log-likelihood
        loss = -log_probs.mean()

        return None, loss
