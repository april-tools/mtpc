import torch
from torch import Tensor

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers import TorchHadamardLayer, TorchKroneckerLayer
from cirkit.pipeline import PipelineContext
from cirkit.utils.scope import Scope
from cirkit.templates import utils, tensor_factorizations, pgms
from cirkit.backend.torch.queries import SamplingQuery, IntegrateQuery

from mtp.models.circuit_layers import TorchBatchedCategoricalLayer, TorchBatchedSumLayer

# from cirkit.templates import tensor_factorizations, utils
from .pipeline import setup_pipeline_context


class ParametersConfig:
    def __init__(self):
        self._sum_layers = []
        self._categorical_layers = []
        self._sum_weights_shapes = []
        self._categorical_log_probs_shapes = []

    @property
    def sum_layers(self) -> list:
        return self._sum_layers
    
    @property
    def categorical_layers(self) -> list:
        return self._categorical_layers

    @property
    def sum_weights_shapes(self) -> list:
        return self._sum_weights_shapes
    
    @property
    def categorical_log_probs_shapes(self) -> list:
        return self._categorical_log_probs_shapes

    def register_sum_layer(self, layer: TorchBatchedSumLayer):
        self._sum_layers.append(layer)
        shape = (layer.num_folds, layer.num_output_units, layer.num_input_units)
        self._sum_weights_shapes.append(shape)

    def register_categorical_layer(self, layer: TorchBatchedCategoricalLayer):
        self._categorical_layers.append(layer)
        shape = (layer.num_folds, layer.num_output_units, layer.num_categories)
        self._categorical_log_probs_shapes.append(shape)


class CircuitModel(torch.nn.Module):
    def __init__(self, vocab_size: int, n_token: int, n_component: int, *, kind: str = 'cp'):
        assert vocab_size > 1
        assert n_token > 1
        assert n_component > 0
        assert kind in ['cp', 'hmm']
        super().__init__()

        self.vocab_size = vocab_size  # V
        self.n_token = n_token  # H
        self.n_component = n_component  # R
        self.kind = kind

        if kind == 'cp':
            if self.n_component == 1:
                # Instantiate a fully-factorized model, i.e., rank-1 CP
                # Instantiate a symbolic circuit encoding a fully-factorized distribution
                symb_circuit = pgms.fully_factorized(
                    self.n_token,
                    input_layer='categorical',
                    input_params={'logits': utils.Parameterization()},
                    input_layer_kwargs={'num_categories': self.vocab_size}
                )
            else:  # self.n_component > 1
                # Instantiate a symbolic circuit encoding the CP decomposition
                symb_circuit = tensor_factorizations.cp(
                    (self.vocab_size,) * self.n_token,
                    rank=self.n_component,
                    input_layer="categorical",
                    input_params={'logits': utils.Parameterization()},
                    weight_param=utils.Parameterization()
                )
        elif kind == 'hmm':
            assert self.n_component > 1, "An HMM model requires n_component > 1"
            # Instantiate an HMM model
            symb_circuit = pgms.hmm(
                list(range(n_token)),
                input_layer='categorical',
                num_latent_states=self.n_component,
                input_params={'logits': utils.Parameterization()},
                input_layer_kwargs={'num_categories': self.vocab_size}
            )
        else:
            assert False, f"Unknown model kind called {kind}"

        # Build the compilation context and compile the circuit
        self._ctx: PipelineContext = setup_pipeline_context()
        self._circuit: TorchCircuit = self._ctx.compile(symb_circuit)

        # Retrieve the circuit layers to parameterize
        self._parameters_config = ParametersConfig()
        for i, layer in enumerate(self._circuit.topological_ordering()):
            if isinstance(layer, (TorchHadamardLayer, TorchKroneckerLayer)):
                continue
            if isinstance(layer, TorchBatchedCategoricalLayer):
                self._parameters_config.register_categorical_layer(layer)
                continue
            if isinstance(layer, TorchBatchedSumLayer):
                self._parameters_config.register_sum_layer(layer)
                continue
            assert False, f"Unknown layer to parameterize, {type(layer)}"

        # We currently support only one folded categorical layer whose folds are sorted based on the token ids
        assert len(self._parameters_config.categorical_layers) == 1

        # Initialize the sampler and the marginalizer objects
        self.sampler = SamplingQuery(self._circuit)
        self.marginalizer = IntegrateQuery(self._circuit)

        # Cache some constants used in self-speculative decoding
        # Masks for marginalising all tokens after position t
        mar_scopes = list(
            reversed(
                [
                    Scope(self.n_token - i - 1 for i in range(t))
                    for t in range(self.n_token)
                ]
            )
        )
        self.register_buffer(
            "_autoregressive_mar_mask",
            IntegrateQuery.scopes_to_mask(self._circuit, mar_scopes),
        )

        # Masks for marginalising all but one token
        one_hot_mar_scopes = [
            Scope(i for i in range(self.n_token) if i != t)
            for t in range(self.n_token)
        ]
        self.register_buffer(
            "_univariate_mar_mask",
            IntegrateQuery.scopes_to_mask(self._circuit, one_hot_mar_scopes),
        )

    @property
    def circuit(self) -> TorchCircuit:
        return self._circuit

    @property
    def parameters_config(self) -> dict:
        return self._parameters_config

    def parameterize(self, parameters: dict):
        # Free previous tensors before we produce new ones
        # this is important, since parameters of Categorical layers can be very large
        for layer in self._parameters_config.sum_layers:
            layer.weight = None
        for layer in self._parameters_config.categorical_layers:
            layer.log_probs = None

        # Set the parameters of the circuit
        for layer, log_probs in zip(self._parameters_config.categorical_layers, parameters['categorical']):
            # log_probs: (F, B, S', R, V) -> (F, B * S', R, V)
            layer.log_probs = log_probs.flatten(1, 2)
        for layer, weight in zip(self._parameters_config.sum_layers, parameters['sum']):
            # weight: (F, B, S', K1, K2) -> (F, B * S', K1, K2)
            layer.weight = weight.flatten(1, 2)

    @torch._dynamo.disable
    def forward(self, yy):
        return self._circuit(yy).ravel()

    @property
    def _batch_size(self) -> int:
        # Hack to get the batch size of the parameters in the circuit
        return self._parameters_config.categorical_layers[0].log_probs.shape[1]
    
    @property
    def _device(self) -> torch.device:
        # Hack to get the device of the circuit
        return self._parameters_config.categorical_layers[0].log_probs.device

    @torch._dynamo.disable
    def univariate_marginal_at_k(self, k: int, yy: Tensor | None = None, with_logits: bool = False):
        assert 0 <= k <= self.n_token
        if with_logits:
            if yy is not None:
                raise ValueError('Expected yy=None, got: %s' % yy)
            BS = self._batch_size
            yy = torch.zeros(BS, self.vocab_size, device=self._device)
            yy[:, k] = -1
        else:
            assert len(yy.shape) == 2
            BS, H = yy.shape
            assert H == self.n_token
        log_probs = self.marginalizer(
            yy, integrate_vars=self._univariate_mar_mask[k]
        )
        if with_logits is True:
            log_probs = log_probs.reshape(BS, self.vocab_size)
        else:
            log_probs = log_probs.reshape(BS)
        # BS, V if with_logits else BS
        return log_probs

    @torch._dynamo.disable
    def autoregressive_marginal_at_k(self, k: int, yy: Tensor, with_logits: bool = False):
        # Marginalises out future tokens
        assert len(yy.shape) == 2
        BS, H = yy.shape
        assert H == self.n_token
        assert 0 <= k <= self.n_token
        if with_logits:
            # In the circuit implementation if we see -1 for a categorical
            # we expand to all possible realisations of that random variable
            yy = yy.clone()
            yy[:, k] = -1
        log_probs = self.marginalizer(
            yy, integrate_vars=self._autoregressive_mar_mask[k]
        )
        if with_logits:
            log_probs = log_probs.reshape(BS, self.vocab_size)
        else:
            log_probs = log_probs.reshape(BS)
        # BS, V if with_logits else BS
        return log_probs

    def autoregressive_conditionals(self, yy: Tensor, with_logits: bool = False):
        BS, H = yy.shape
        assert H == self.n_token

        # NOTE: Below can be computed in parallel
        marginals = []
        for k in range(H):
            # Compute P(x_{t+1}, x_{t+2}, .. , x_{t+k} | x_{<=t})
            # BS x V if with_logits else BS x 1
            marginal = self.autoregressive_marginal_at_k(
                k, yy=yy, with_logits=with_logits
            )
            marginals.append(marginal)
        marginals = torch.stack(marginals)
        # Go in reverse to avoid overwriting useful info.
        # Stop at 1, since conditional for ntp is just marginal
        for k in reversed(range(1, H)):
            # P(x_{t+k} | x_{t+1}, x_{t+2}, .. , x_{t+k-1}, x_{<=t}) =
            # P(x_{t+1}, x_{t+2}, .. , x_{t+k} | x_{<=t}) /
            # P(x_{t+1}, x_{t+2}, .. , x_{t+k-1} | x_{<=t})
            # we subtract since these are logprobs
            if with_logits is False:
                marginals[k] = marginals[k] - marginals[k - 1]
            else:
                # NOTE: if we use with_logits, we have a slight complication:
                # both marginals we use in the division have been evaluated for
                # all possible settings of the last categorical variable,
                # x_{t+k}, and x_{t+k-1}, respectively.
                # Therefore, for the prev_marginal we need to pick the value
                # that we condition on. Below we pick this value:
                prev_marginals = marginals[k - 1][
                    torch.arange(BS, device=yy.device), yy[:, k - 1].ravel()
                ]
                # Unsqueeze to broadcast
                marginals[k] = marginals[k] - prev_marginals.unsqueeze(-1)
        # H, BS, V if with_logits else H, BS
        return marginals

    def sample(self, num_samples=1):
        return self.sampler(num_samples=num_samples)
