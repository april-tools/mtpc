import torch

from torch import Tensor
from cirkit.backend.torch.queries import SamplingQuery

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
        self._sum_layer = layers[2]
        assert isinstance(self._sum_layer, TorchBatchedSumLayer)

        self.sampler = SamplingQuery(self.circuit._circuit)

    # TODO: Replace xx and yy with input_ids
    def forward(self, xx: Tensor, yy: Tensor = None, return_logits: bool = False):

        # If we pass a target yy, we are in training mode
        # In training mode we use teacher forcing to make model(xx[:i]) predict yy[i:i+H]
        # If we are not in training mode, we want to predict the tokens that should follow xx
        training_mode = (yy is not None)

        # (B, S, D)
        xx = self.lm_encoder(xx)

        if not training_mode:
            # At generation time, we want to do future token prediction
            # so we only condition on last output
            xx = xx[:, [-1], :]

        # Obtain dict of circuit parameters
        circuit_params = self.lm_head(xx)

        # Index the categorical logits and the sum weight accordingly
        # cat_logits: (H, B, S, R, V) -> (H, B * S', R, V)
        cat_log_probs = circuit_params['cat_log_probs']

        # sum_weight: (B, S, 1, R) -> (1, B * S', 1, R)
        sum_weight = circuit_params['sum_weight']

        if training_mode:
            # S' = S - H + 1
            crop_sentence_len = xx.shape[1] - self.lm_head.n_token + 1

            cat_log_probs = cat_log_probs[:, :, :crop_sentence_len]
            sum_weight = sum_weight[:, :crop_sentence_len]

        cat_log_probs = cat_log_probs.reshape(cat_log_probs.shape[0], -1, cat_log_probs.shape[3], cat_log_probs.shape[4])
        sum_weight = sum_weight.reshape(1, -1, 1, sum_weight.shape[3])

        self._cat_layer.log_probs = cat_log_probs
        self._sum_layer.weight = sum_weight

        mtp_loss = None

        if training_mode:

            # Compute sliding windows indices
            # from yy: (B, S) to yy: (B * S', 1, H)
            # unsqueeze channel dimension -> (B * S', 1, H)
            tokens_idx = torch.arange(yy.shape[1], device=yy.device)                 # (S,)
            # last two arguments are window size and step, correspondingly
            sliding_window_idx = tokens_idx.unfold(0, self.lm_head.n_token, 1)       # (S', H)
            yy = yy[:, sliding_window_idx].view(-1, 1, sliding_window_idx.shape[1])  # (B * S', 1, H)

            # Compute the conditional log-likelihoods
            # yy: (B * S', 1, H)
            # log_probs: (B * S', 1, 1)
            log_probs = self.circuit(yy)

            # The loss is the negated average conditional log-likelihood
            mtp_loss = -log_probs.mean()

            # Next token prediction - equivalent to marginalising out future tokens
            # See https://arxiv.org/pdf/2410.17765, eq. 11
            # (1, B * S', 1, V)
            next_token_cats = torch.exp(self._cat_layer.log_probs[0, :, :, :])
            # (B * S', V)
            next_token_probs = (self._sum_layer.weight @ next_token_cats).squeeze()

            # We keep track of next token prediction loss too, in order to discern
            # how good the model would be for just next token prediction
            bs_idxs = torch.arange(yy.shape[0], device=yy.device)
            stp_probs = next_token_probs[bs_idxs, yy[:, :, 0].ravel()]
            stp_loss = -torch.log(stp_probs).mean()

        return dict(loss=mtp_loss, stp_loss=stp_loss, mtp_loss=mtp_loss)

    @torch.no_grad()
    def generate(self, inputs: torch.Tensor):

        self.forward(inputs)

        sample, mix_samples = self.sampler(1)
        # Remove Extraneous Channel Dimension
        sample = sample.squeeze(dim=1)
        return sample
