import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, SumLayer, CategoricalLayer
from cirkit.utils.scope import Scope
from cirkit.backend.torch.queries import SamplingQuery, IntegrateQuery

# from cirkit.templates import tensor_factorizations, utils
from .pipeline import setup_pipeline_context


class CircuitCP(torch.nn.Module):
    def __init__(self, vocab_size, n_token, n_component):
        super().__init__()
        self.vocab_size = vocab_size  # V
        self.n_token = n_token  # H
        self.n_component = n_component  # R
        if self.n_token > 1:
            # TODO: when we will be able to set custom input layers (e.g., CategoricalLayer) in tensor factorizations
            #  (which is very soon)
            # NOTE: Below will not work for case n_token=1, because an arity check fails.
            # self.symb_circuit = tensor_factorizations.cp(
            #     (self.vocab_size,) * self.n_token,
            #     rank=self.n_component,
            #     factor_param=utils.Parameterization(initialization='normal', activation='none'),
            #     weight_param=utils.Parameterization(initialization='normal', activation='none')
            # )
            cats = [
                CategoricalLayer(
                    scope=Scope([i]),
                    num_output_units=self.n_component,
                    num_channels=1,
                    num_categories=self.vocab_size,
                )
                for i in range(self.n_token)
            ]
            hadamard = HadamardLayer(self.n_component, arity=self.n_token)
            out = SumLayer(
                num_input_units=self.n_component, num_output_units=1, arity=1
            )
            self.symb_circuit = Circuit(
                num_channels=1,
                layers=cats + [hadamard, out],
                in_layers={out: [hadamard], hadamard: cats},
                outputs=[out],
            )
            self.cat_layer_idx = 0
            self.sum_layer_idx = 2
        elif self.n_token == 1:
            cat = CategoricalLayer(
                scope=Scope([0]),
                num_output_units=self.n_component,
                num_channels=1,
                num_categories=self.vocab_size,
            )
            out = SumLayer(
                num_input_units=self.n_component, num_output_units=1, arity=1
            )
            self.symb_circuit = Circuit(
                num_channels=1,
                layers=[cat, out],
                in_layers={out: [cat]},
                outputs=[out],
            )
            # We have no HadamardLayer in between
            self.cat_layer_idx = 0
            self.sum_layer_idx = 1
        else:
            raise ValueError("n_token must be > 0, got %d" % self.n_token)

        self._ctx: PipelineContext = setup_pipeline_context()
        self._circuit: TorchCircuit = self._ctx.compile(self.symb_circuit)

        # Initializer the sampler and the marginalizer objects
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
            Scope([i for i in range(self.n_token) if i != t])
            for t in range(self.n_token)
        ]
        self.register_buffer(
            "_univariate_mar_mask",
            IntegrateQuery.scopes_to_mask(self._circuit, one_hot_mar_scopes),
        )

    @property
    def circuit(self) -> TorchCircuit:
        return self._circuit

    @torch._dynamo.disable
    def forward(self, yy):
        return self._circuit(yy).ravel()

    @torch._dynamo.disable
    def univariate_marginal_at_k(self, k, yy=None, with_logits=False):
        assert 0 <= k <= self.n_token
        if with_logits:
            if yy is not None:
                raise ValueError('Expected yy=None, got: %s' % yy)
            log_probs = self._circuit.layers[self.cat_layer_idx].log_probs
            H, BS, R, V = log_probs.shape
            yy = torch.zeros(BS, 1, V, device=log_probs.device)
            yy[:, :, k] = -1
        else:
            assert len(yy.shape) == 3
            BS, C, H = yy.shape
            assert C == 1
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
    def autoregressive_marginal_at_k(self, k, yy, with_logits=False):
        # Marginalises out future tokens
        assert len(yy.shape) == 3
        BS, C, H = yy.shape
        assert C == 1
        assert H == self.n_token
        assert 0 <= k <= self.n_token
        if with_logits:
            # In the circuit implementation if we see -1 for a categorical
            # we expand to all possible realisations of that random variable
            yy = yy.clone()
            yy[:, :, k] = -1
        log_probs = self.marginalizer(
            yy, integrate_vars=self._autoregressive_mar_mask[k]
        )
        if with_logits is True:
            log_probs = log_probs.reshape(BS, self.vocab_size)
        else:
            log_probs = log_probs.reshape(BS)
        # BS, V if with_logits else BS
        return log_probs

    def autoregressive_conditionals(self, yy, with_logits=False):
        assert len(yy.shape) == 3
        BS, C, H = yy.shape
        assert C == 1
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
                    torch.arange(BS, device=yy.device), yy[:, :, k - 1].ravel()
                ]
                # Unsqueeze to broadcast
                marginals[k] = marginals[k] - prev_marginals.unsqueeze(-1)
        # H, BS, V if with_logits else H, BS
        return marginals

    def sample(self, num_samples=1):
        return self.sampler(num_samples=num_samples)
