import torch

from torch import Tensor
from cirkit.backend.torch.queries import SamplingQuery

from nanogpt.models.circuit import CircuitCP, MultiTokenHead
from nanogpt.models.gpt import GPTEncoder
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
    def __init__(self, lm_encoder: GPTEncoder, lm_head: MultiTokenHead, circuit: CircuitCP):
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
        self.sampler = SamplingQuery(self.circuit.circuit)

    def forward(self, xx: Tensor, yy: Tensor | None = None) -> Tensor | None:
        # If we pass a target yy, we are in training mode
        # In training mode we use teacher forcing to make model(xx[:i]) predict yy[i:i+H]
        # If we are not in training mode, we want to predict the tokens that should follow xx
        training_mode = (yy is not None)

        # xx: (B, S, D)
        xx = self.lm_encoder(xx)
        if training_mode:
            # At training time, we want to learn to predict the next H tokens
            # xx: (B, S', D), where S' = S - H + 1
            xx: xx[:, :-self.lm_head.n_token + 1]
        else:
            # At generation time, we want to do future token prediction
            # so we only condition on last output
            # xx: (B, S', D), where S' = 1
            xx = xx[:, [-1]]

        # Obtain dictionary of circuit parameters
        circuit_params = self.lm_head(xx)
        # cat_logits: (H, B, S', R, V)
        cat_log_probs = circuit_params['cat_log_probs']
        # sum_weight: (B, S', 1, R)
        sum_weight = circuit_params['sum_weight']

        # cat_log_probs: (H, B * S', R, V)
        cat_log_probs = cat_log_probs.view(cat_log_probs.shape[0], -1, cat_log_probs.shape[3], cat_log_probs.shape[4])
        # sum_weight: (1, B * S', 1, R)
        sum_weight = sum_weight.view(1, -1, 1, sum_weight.shape[3])

        # Set the parameters of the circuit
        self._cat_layer.log_probs = cat_log_probs
        self._sum_layer.weight = sum_weight

        loss: Tensor | None = None
        if training_mode:
            # Compute sliding windows indices
            # from yy: (B, S) to yy: (B, S', H)
            # where S' = S - H + 1
            yy = yy.unfold(dimension=1, size=self.lm_head.n_token, step=1)
            # Note that we unsqueeze a channel dimension, as required by cirkit
            # yy: (B, S', H) -> (B * S', 1, H)
            yy = yy.reshape(-1, 1, yy.shape[2])

            # Compute the conditional log-likelihoods
            # yy: (B * S', 1, H)
            # log_probs: (B * S', 1, 1)
            log_probs = self.circuit(yy)

            # The loss is the negated average conditional log-likelihood
            loss = -log_probs.mean()

        return loss

    @torch.no_grad()
    def generate(self, inputs: torch.Tensor):

        self.forward(inputs)

        sample, mix_samples = self.sampler(1)
        # Remove extraneous channel dimension
        sample = sample.squeeze(dim=1)
        return sample
