import torch
import functools

from abc import ABC
from collections.abc import Mapping
from collections.abc import Iterable
from typing import Any
from torch import Tensor, distributions

from cirkit.backend.torch.layers import TorchExpFamilyLayer, TorchInnerLayer
from cirkit.backend.torch.layers import TorchInputLayer, TorchLayer
from cirkit.backend.torch.semiring import Semiring, LSESumSemiring
from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.utils.scope import Scope

from mtp.models.loss import IGNORE_TOKEN_ID


def sanitize_input(yy: Tensor) -> Tensor:
    """
    Sanitizes the input tensor, yy, by replacing IGNORE_TOKEN_ID values
    which cannot be processed by cirkit.

    Args:
        yy (Tensor): The input tensor containing values to be sanitized.

    Returns:
        Tensor: The sanitized tensor with invalid values corrected.
    """
    yyc = yy.clone()
    # 0 is just a placeholder value - the random variables we are setting
    # will be marginalised out via marg_mask, so the value does not matter.
    NO_ERROR_PLACEHOLDER = 0
    yyc[yy == IGNORE_TOKEN_ID] = NO_ERROR_PLACEHOLDER
    return yyc


class TorchBatchedCategoricalLayer(TorchExpFamilyLayer):
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_categories: int = 2,
        semiring: Semiring | None = None,
    ) -> None:
        """Initialize a Categorical layer.

        Args:
            scope_idx: A tensor of shape $(F, D)$, where $F$ is the number of folds, and
                $D$ is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape $(D,)$ can be specified, which will be interpreted
                as a tensor of shape $(1, D)$, i.e., with $F = 1$.
            num_output_units: The number of output units.
            num_categories: The number of categories for Categorical distribution.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].

        Raises:
            ValueError: If the scope contains more than one variable.
            ValueError: If the number of categories is negative.
        """
        num_variables = scope_idx.shape[-1]
        if num_variables != 1:
            raise ValueError(
                "The batched Categorical layer encodes a univariate distribution"
            )
        if num_categories <= 0:
            raise ValueError(
                "The number of categories for Categorical distribution must be positive"
            )
        super().__init__(
            scope_idx,
            num_output_units,
            semiring=semiring,
        )
        self.num_categories = num_categories
        self._log_probs: Tensor | None = None

    @property
    def log_probs(self) -> Tensor:
        if self._log_probs is None:
            raise ValueError("No log probs have been set")
        return self._log_probs

    @log_probs.setter
    def log_probs(self, log_probs: Tensor | None):
        if log_probs is not None:
            if (
                len(log_probs.shape) != 4
                or log_probs.shape[0] != self.num_folds
                or log_probs.shape[2] != self.num_output_units
                or log_probs.shape[3] != self.num_categories
            ):
                raise ValueError(
                    f"Expected log probs of shape ({self.num_folds}, B, {self.num_output_units}, {self.num_categories}), "
                    f"but found {log_probs.shape}"
                )
        self._log_probs = log_probs

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_output_units": self.num_output_units,
            "num_categories": self.num_categories,
        }
    
    @property
    def fold_settings(self) -> tuple[Any, ...]:
        return self.num_variables, *self.config.items()

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Categorical should be discrete

        # NOTE: Below is because we use -100 for tokens that should not be
        # predicted. While we will marginalise those out, cirkit chokes on -100
        # so replace it with a placeholder value (it does not matter which value).
        x = sanitize_input(x)

        # x: (F, B, 1) -> (F, B)
        x = x.squeeze(dim=2)
        F, B = x.shape
        V = self.num_categories

        # log_probs: (F, B, K, V)
        log_probs = self.log_probs
        idx_fold = torch.arange(F, device=log_probs.device)
        idx_batch = torch.arange(B, device=log_probs.device)
        # y: (F, B, K)
        if log_probs.shape[1] != B:
            log_probs = log_probs.broadcast_to(-1, B, -1, -1)

        # While expensive, we can compute the probability for all realisations
        # of a single categorical variable. We need this for some losses,
        # such as KL, where we need the whole categorical distribution.
        # If we want to do this, we set the value of x for all entries in that
        # fold to -1. Let's check if we have fold that is all negative ones.
        expand_logits = torch.all(x == -1, dim=-1)
        if torch.any(expand_logits):
            assert expand_logits.sum() == 1
            # Expand idx batch
            idx_batch = torch.repeat_interleave(idx_batch, V, dim=0)
            # Repeat batch dimension
            x = torch.repeat_interleave(x, V, dim=1)
            # Replace the -1 with torch.arange(V).num_categories)
            x[expand_logits] = torch.tile(torch.arange(V, device=log_probs.device), (B,))
        y = log_probs[idx_fold[:, None], idx_batch[None, :], :, x]
        return self.semiring.map_from(y, LSESumSemiring)

    def log_partition_function(self) -> Tensor:
        return torch.zeros(
            size=(self.num_folds, 1, self.num_output_units), device=self.log_probs.device
        )

    def sample(self, num_samples: int = 1) -> Tensor:
        # log_probs: (F, B, K, V)
        log_probs = self.log_probs
        probs = torch.exp(log_probs)
        dist = distributions.Categorical(probs=probs)
        # samples: (num_samples, F, B, K)
        samples = dist.sample((num_samples,))
        # samples: (F, K, num_samples, B) -> (F, K, num_samples * B)
        samples = samples.permute(1, 3, 0, 2)
        samples = samples.flatten(start_dim=2)
        return samples


class TorchBatchedSumLayer(TorchInnerLayer):
    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        *,
        semiring: Semiring | None = None,
        num_folds: int = 1,
    ):
        r"""Initialize a sum layer.

        Args:
            num_input_units: The number of input units.
            num_output_units: The number of output units.
            arity: The arity of the layer.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].
            num_folds: The number of channels.

        Raises:
            ValueError: If the arity is not a positive integer.
        """
        if arity < 1:
            raise ValueError("The arity must be a positive integer")
        super().__init__(
            num_input_units,
            num_output_units,
            arity=arity,
            semiring=semiring,
            num_folds=num_folds,
        )
        self._weight: Tensor | None = None

    @property
    def weight(self) -> Tensor:
        if self._weight is None:
            raise ValueError("No weight have been set")
        return self._weight

    @weight.setter
    def weight(self, weight: Tensor | None):
        if weight is not None:
            if (
                len(weight.shape) != 4
                or weight.shape[0] != self.num_folds
                or weight.shape[2] != self.num_output_units
                or weight.shape[3] != self.arity * self.num_input_units
            ):
                raise ValueError(
                    f"Expected probs of shape ({self.num_folds}, B, {self.num_output_units}, {self.arity * self.num_input_units}), "
                    f"but found {weight.shape}"
                )
        self._weight = weight

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
        }

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        return *self.config.items(),

    def forward(self, x: Tensor) -> Tensor:
        # x: (F, H, B, Ki) -> (F, B, H * Ki)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        # weight: (F, B, Ko, H * Ki)
        # If we expanded logits in the categorical
        # we need to expand the weights along the B axis here too
        if self.weight.shape[1] != x.shape[1]:
            V = x.shape[1] // self.weight.shape[1]
            weight = torch.repeat_interleave(self.weight, V, dim=1)
        else:
            weight = self.weight
        return self.semiring.einsum(
            "fbi,fboi->fbo", inputs=(x,), operands=(weight,), dim=-1, keepdim=True
        )  # shape (F, B, Ko).

    def sample(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # weight: (F, B, Ko, H * Ki)
        weight = self.weight
        negative = torch.any(weight < 0.0)
        normalized = torch.allclose(
            torch.sum(weight, dim=-1), torch.ones(1, device=weight.device)
        )
        if negative or not normalized:
            raise TypeError(
                "Sampling in sum layers only works with positive weights summing to 1"
            )

        # x: (F, H, Ki, num_samples * B, D) -> (F, H * Ki, num_samples * B, D)
        num_samples = x.shape[3] // weight.shape[1]
        x = x.flatten(1, 2)

        # mixing_distribution: (F, B, Ko, H * Ki)
        mixing_distribution = torch.distributions.Categorical(probs=weight)

        # mixing_samples: (num_samples, F, B, Ko) -> (F, Ko, num_samples, B) -> (F, Ko, num_samples * B)
        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_samples = mixing_samples.permute(1, 3, 0, 2)
        mixing_samples = mixing_samples.flatten(start_dim=2)

        # Choose the sample that was chosen by the sum layer
        # This is done by selecting the corresponding index using gather
        # mixing_indices: (F, Ko, num_samples * B, 1) -> (F, Ko, num_samples * B, D)
        mixing_indices = mixing_samples.unsqueeze(dim=-1)
        mixing_indices = mixing_indices.broadcast_to(
            mixing_samples.shape[0],
            mixing_samples.shape[1],
            mixing_samples.shape[2],
            x.shape[3],
        )

        # x: (F, Ko, num_samples * B, D)
        x = torch.gather(x, dim=1, index=mixing_indices)
        return x, mixing_samples


class Query(ABC):
    """An object used to run queries of circuits compiled using the torch backend."""

    def __init__(self) -> None:
        ...


class IntegrateQuery(Query):
    """The integration query object allows marginalising out variables.

    Computes output in two forward passes:
        a) The normal circuit forward pass for input x
        b) The integration forward pass where all variables are marginalised

    A mask over random variables is computed based on the scopes passed as
    input. This determines whether the integrated or normal circuit result
    is returned for each variable.
    """

    def __init__(self, circuit: TorchCircuit) -> None:
        """Initialize an integration query object.

        Args:
            circuit: The circuit to integrate over.

        Raises:
            ValueError: If the circuit to integrate is not smooth or not decomposable.
        """
        if not circuit.properties.smooth or not circuit.properties.decomposable:
            raise ValueError(
                f"The circuit to integrate must be smooth and decomposable, "
                f"but found {circuit.properties}"
            )
        super().__init__()
        self._circuit = circuit

    def __call__(self, x: Tensor, *, integrate_vars: Tensor | Scope | Iterable[Scope]) -> Tensor:
        """Solve an integration query, given an input batch and the variables to integrate.

        Args:
            x: An input batch of shape $(B, D)$, where $B$ is the batch size,
                and $D$ is the number of variables.
            integrate_vars: The variables to integrate. It must be a subset of the variables on
                which the circuit given in the constructor is defined on.
                The format can be one of the following three:
                    1. Tensor of shape (B, D) where B is the batch size and D is the number of
                        variables in the scope of the circuit. Its dtype should be torch.bool
                        and have True in the positions of random variables that should be
                        marginalised out and False elsewhere.
                    2. Scope, in this case the same integration mask is applied for all entries
                        of the batch
                    3. List of Scopes, where the length of the list must be either 1 or B. If
                        the list has length 1, behaves as above.
        Returns:
            The result of the integration query, given as a tensor of shape $(B, O, K)$,
                where $B$ is the batch size, $O$ is the number of output vectors of the circuit, and
                $K$ is the number of units in each output vector.
        """
        if isinstance(integrate_vars, Tensor):
            # Check type of tensor is boolean
            if integrate_vars.dtype != torch.bool:
                raise ValueError(
                    f"Expected dtype of tensor to be torch.bool, got {integrate_vars.dtype}"
                )
            # If single dimensional tensor, assume batch size = 1
            if len(integrate_vars.shape) == 1:
                integrate_vars = torch.unsqueeze(integrate_vars, 0)
            # If the scope is correct, proceed, otherwise error
            num_vars = max(self._circuit.scope) + 1
            if integrate_vars.shape[1] == num_vars:
                integrate_vars_mask = integrate_vars
            else:
                raise ValueError(
                    f"Circuit scope has {num_vars} variables but integrate_vars "
                    f"was defined over {integrate_vars.shape[1]} != {num_vars} variables"
                )
        else:
            # Convert list of scopes to a boolean mask of dimension (B, N) where
            # N is the number of variables in the circuit's scope.
            integrate_vars_mask = IntegrateQuery.scopes_to_mask(self._circuit, integrate_vars)
            integrate_vars_mask = integrate_vars_mask.to(x.device)

        # Check batch sizes of input x and mask are compatible
        if integrate_vars_mask.shape[0] not in (1, x.shape[0]):
            raise ValueError(
                "The number of scopes to integrate over must "
                "either match the batch size of x, or be 1 if you "
                "want to broadcast. Found #inputs = "
                f"{x.shape[0]} != {integrate_vars_mask.shape[0]} = len(integrate_vars)"
            )

        output = self._circuit.evaluate(
            x,
            module_fn=functools.partial(
                IntegrateQuery._layer_fn, integrate_vars_mask=integrate_vars_mask
            ),
        )  # (O, B, K)
        return output.transpose(0, 1)  # (B, O, K)

    @staticmethod
    def _layer_fn(layer: TorchLayer, x: Tensor, *, integrate_vars_mask: Tensor) -> Tensor:
        # Evaluate a layer: if it is not an input layer, then evaluate it in the usual
        # feed-forward way. Otherwise, use the variables to integrate to solve the marginal
        # queries on the input layers.
        output = layer(x)  # (F, B, Ko)
        if not isinstance(layer, TorchInputLayer):
            return output
        if layer.num_variables > 1:
            raise NotImplementedError("Integration of multivariate input layers is not supported")
        # integrate_vars_mask is a boolean tensor of dim (B, N)
        # where N is the number of variables in the scope of the whole circuit.
        #
        # layer.scope_idx contains a subset of the variable_idxs of the scope
        # but may be a reshaped tensor; the shape and order of the variables may be different.
        #
        # as such, we need to use the idxs in layer.scope_idx to lookup the values from
        # the integrate_vars_mask - this will return the correct shape and values.
        #
        # if integrate_vars_mask was a vector, we could do integrate_vars_mask[layer.scope_idx]
        # the vmap below applies the above across the B dimension

        # integration_mask has dimension (B, F, Ko)
        integration_mask = torch.vmap(lambda x: x[layer.scope_idx])(integrate_vars_mask)
        # permute to match integration_output: integration_mask has dimension (F, B, Ko)
        integration_mask = integration_mask.permute([1, 0, 2])

        if not torch.any(integration_mask).item():
            return output

        integration_output = layer.integrate()

        # NOTE: Below is our hack to support `with_logits` for circuits
        # Problem: when `with_logits` is true, outputs has dim: (F, B * V, Ko)
        # while integration_output always has dim  (F, 1, Ko).
        # Now, if integration_mask has dim (F, 1, Ko) - broadcasting works.
        # However, if we set a batched integration mask (F, B, Ko)
        # broadcasting no longer works. Below we fix this case by
        # repeating the integration mask to match the logits
        mask_bs, output_bs = integration_mask.shape[1], output.shape[1]
        if mask_bs != output_bs:
            vocab_size = output_bs // mask_bs
            integration_mask = torch.repeat_interleave(integration_mask, vocab_size, dim=1)

        # Use the integration mask to select which output should be the result of
        # an integration operation, and which should not be
        # This is done in parallel for all folds, and regardless of whether the
        # circuit is folded or unfolded
        return torch.where(integration_mask, integration_output, output)

    @staticmethod
    def scopes_to_mask(circuit: TorchCircuit, batch_integrate_vars: Scope | list[Scope]):
        """Accepts a batch of scopes and returns a boolean mask as a tensor with
        True in positions of specified scope indices and False otherwise.
        """
        # If we passed a single scope, assume B = 1
        if isinstance(batch_integrate_vars, Scope):
            batch_integrate_vars = [batch_integrate_vars]

        batch_size = len(tuple(batch_integrate_vars))
        # There are cases where the circuit.scope may change,
        # e.g. we may marginalise out X_1 and the length of the scope may be smaller
        # but the actual scope will not have been shifted.
        num_rvs = max(circuit.scope) + 1
        num_idxs = sum(len(s) for s in batch_integrate_vars)

        # TODO: Maybe consider using a sparse tensor
        mask = torch.zeros((batch_size, num_rvs), dtype=torch.bool)

        # Catch case of only empty scopes where the following command will fail
        if num_idxs == 0:
            return mask

        batch_idxs, rv_idxs = zip(
            *((i, idx) for i, idxs in enumerate(batch_integrate_vars) for idx in idxs if idxs)
        )

        # Check that we have not asked to marginalise variables that are not defined
        invalid_idxs = Scope(rv_idxs) - circuit.scope
        if invalid_idxs:
            raise ValueError(
                "The variables to marginalize must be a subset of "
                "the circuit scope. Invalid variables "
                f"not in scope: {list(invalid_idxs)} "
            )

        mask[batch_idxs, rv_idxs] = True

        return mask


class SamplingQuery(Query):
    """The sampling query object."""

    def __init__(self, circuit: TorchCircuit) -> None:
        """Initialize a sampling query object. Currently, only sampling from the joint distribution
            is supported, i.e., sampling won't work in the case of circuits obtained by
            marginalization, or by observing evidence. Conditional sampling is currently not
            implemented.

        Args:
            circuit: The circuit to sample from.

        Raises:
            ValueError: If the circuit to sample from is not normalised.
        """
        if not circuit.properties.smooth or not circuit.properties.decomposable:
            raise ValueError(
                f"The circuit to sample from must be smooth and decomposable, "
                f"but found {circuit.properties}"
            )
        # TODO: add a check to verify the circuit is monotonic and normalized?
        super().__init__()
        self._circuit = circuit

    def __call__(self, num_samples: int = 1) -> tuple[Tensor, list[Tensor]]:
        """Sample a number of data points.

        Args:
            num_samples: The number of samples to return.

        Return:
            A pair (samples, mixture_samples), consisting of (i) an assignment to the observed
            variables the circuit is defined on, and (ii) the samples of the finitely-discrete
            latent variables associated to the sum units. The samples (i) are returned as a
            tensor of shape (num_samples, num_variables).

        Raises:
            ValueError: if the number of samples is not a positive number.
        """
        if num_samples <= 0:
            raise ValueError("The number of samples must be a positive number")

        mixture_samples: list[Tensor] = []
        # samples: (O, K, num_samples, D)
        samples = self._circuit.evaluate(
            module_fn=functools.partial(
                self._layer_fn,
                num_samples=num_samples,
                mixture_samples=mixture_samples,
            ),
        )
        # samples: (num_samples, O, K, D)
        samples = samples.permute(2, 0, 1, 3)
        # TODO: fix for the case of multi-output circuits, i.e., O != 1 or K != 1
        samples = samples[:, 0, 0]  # (num_samples, D)
        return samples, mixture_samples

    def _layer_fn(
        self, layer: TorchLayer, *inputs: Tensor, num_samples: int, mixture_samples: list[Tensor]
    ) -> Tensor:
        # Sample from an input layer
        if not inputs:
            assert isinstance(layer, TorchInputLayer)
            samples = layer.sample(num_samples)
            samples = self._pad_samples(samples, layer.scope_idx)
            mixture_samples.append(samples)
            return samples

        # Sample through an inner layer
        assert isinstance(layer, TorchInnerLayer)
        samples, mix_samples = layer.sample(*inputs)
        if mix_samples is not None:
            mixture_samples.append(mix_samples)
        return samples

    def _pad_samples(self, samples: Tensor, scope_idx: Tensor) -> Tensor:
        """Pads univariate samples to the size of the scope of the circuit (output dimension)
        according to scope for compatibility in downstream inner nodes.
        """
        if scope_idx.shape[1] != 1:
            raise NotImplementedError("Padding is only implemented for univariate samples")

        # padded_samples: (F, K, num_samples, D)
        padded_samples = torch.zeros(
            (*samples.shape, len(self._circuit.scope)), device=samples.device, dtype=samples.dtype
        )
        fold_idx = torch.arange(samples.shape[0], device=samples.device)
        padded_samples[fold_idx, :, :, scope_idx.squeeze(dim=1)] = samples
        return padded_samples
