import torch

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.pipeline import PipelineContext
from cirkit.symbolic.layers import CategoricalLayer, SumLayer
from .circuit_layers import TorchBatchedCategoricalLayer, TorchBatchedSumLayer


def setup_pipeline_context() -> PipelineContext:
    # Initialize a cirkit pipeline compilation context,
    # where we overwrite the compilation rules of categorical and sum layers
    # as to allow parameters having an extra batch dimension
    ctx = PipelineContext(
        backend="torch",
        semiring="lse-sum",
        fold=True,
        optimize=False,
    )
    ctx.add_layer_compilation_rule(compile_batched_categorical_layer)
    ctx.add_layer_compilation_rule(compile_batched_sum_layer)
    return ctx


def compile_batched_categorical_layer(
    compiler: TorchCompiler, sl: CategoricalLayer
) -> TorchBatchedCategoricalLayer:
    return TorchBatchedCategoricalLayer(
        torch.tensor(tuple(sl.scope)),
        sl.num_output_units,
        num_channels=sl.num_channels,
        num_categories=sl.num_categories,
        semiring=compiler.semiring,
    )


def compile_batched_sum_layer(
    compiler: TorchCompiler, sl: SumLayer
) -> TorchBatchedSumLayer:
    return TorchBatchedSumLayer(
        sl.num_input_units,
        sl.num_output_units,
        arity=sl.arity,
        semiring=compiler.semiring,
    )
