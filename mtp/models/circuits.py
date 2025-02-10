import torch

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.layers import HadamardLayer, SumLayer, CategoricalLayer
from cirkit.utils.scope import Scope

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
                num_channels=1, layers=[cat, out], in_layers={out: [cat]}, outputs=[out]
            )
            # We have no HadamardLayer in between
            self.cat_layer_idx = 0
            self.sum_layer_idx = 1
        else:
            raise ValueError("n_token must be > 0, got %d" % self.n_token)

        self._ctx: PipelineContext = setup_pipeline_context()
        self._circuit: TorchCircuit = self._ctx.compile(self.symb_circuit)

    @property
    def circuit(self) -> TorchCircuit:
        return self._circuit

    @torch._dynamo.disable
    def forward(self, yy):
        return self._circuit(yy)
