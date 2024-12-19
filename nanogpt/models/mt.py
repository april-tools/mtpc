import torch
from cirkit.templates.logic.sdd import sliding_window

from torch import Tensor

from nanogpt.models.layers import TorchBatchedCategoricalLayer


class MultiTokenLM(torch.nn.Module):
    """ A MultiTokenLM comprises three parts:

    1. A LM encoder, which can be the encoder (i.e. arch without lm_head)
    of any pretrained LLM. The encoder provides contextual embeddings for
    tokens.

    2. A lm_head, which expands the contextual embeddings into parameters
    for the (circuit) output layer.

    3. A circuit which models the output tokens and encodes their dependencies.
    """
    def __init__(self, lm_encoder, lm_head, circuit):
        super().__init__()

        self.lm_encoder = lm_encoder
        self.lm_head = lm_head
        self.circuit = circuit

        self._cat_layer: TorchBatchedCategoricalLayer = next(self.circuit.circuit.input_layers)
        assert isinstance(self._cat_layer, TorchBatchedCategoricalLayer)
        # self._sum_layer = self.circuit.circuit.layers[2]

    def forward(self, xx: Tensor, yy: Tensor, return_logits=False):
        # (B, S, D)
        xx = self.lm_encoder(xx)

        # Obtain dict of params
        circuit_params = self.lm_head(xx)

        # (H, B * S', R, V)
        cat_logits = circuit_params['categoricals']
        self._cat_layer.probs = torch.softmax(cat_logits, dim=-1)

        # TODO: Parametrise the sum weights
        # sum_weight = circuit_params['sum_weight']  # (1, B * S', 1, R)
        # self._sum_layer.weight = lambda: torch.softmax(sum_weight, dim=-1)

        # Extract sliding windows
        # from yy: (B, S) to yy: (B * S', H)
        # S' = S - H + 1
        # unsqueeze channel dimension -> (B * S', 1, H)
        tokens_idx = torch.arange(yy.shape[0], device=yy.device)
        sliding_window_idx = tokens_idx.unfold(0, self.lm_head.n_token, 1)
        yy = yy[:, sliding_window_idx].view(-1, 1, self.lm_head.n_token)

        # Compute the conditional log-likelihoods
        log_probs = self.circuit(yy)  # (B * S', 1, 1)
        loss = -log_probs.mean()

        return None, loss
