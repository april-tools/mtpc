import torch

from torch import Tensor


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

        # self._layers_to_parameterize: Dict[str, TorchLayer]
        self._cat_layer = self.circuit.circuit.layers[0]
        self._sum_layer = self.circuit.circuit.layers[2]

    def forward(self, xx: Tensor, yy: Tensor, return_logits=True):

        # (B, S, D)
        xx = self.lm_encoder(xx)

        circuit_params = self.lm_head(logits)

        cat_logits = circuit_params['categorical']  # (H, B * S', K, V)
        sum_weight = circuit_params['sum_weight']  # (1, B * S', 1, K)

        self._cat_layer.probs = lambda: torch.softmax(cat_logits, dim=-1)
        self._sum_layer.weight = lambda: torch.softmax(sum_weight, dim=-1)

        # Extract sliding windows? from yy: (B, S) to yy: (B * S', H)

        # yy: (B * S', H)
        # unsqueeze channel dimension -> (B * S', 1, H)
        yy = yy.unsqueeze(dim=1)
        log_probs = self.circuit(yy)  # (B * S', 1)

        return log_probs
