import torch

from collections.abc import Mapping
from typing import Any
from torch import Tensor, distributions

from cirkit.backend.torch.layers import TorchExpFamilyLayer, TorchInnerLayer
from cirkit.backend.torch.semiring import Semiring, LSESumSemiring

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

    def mode(self) -> tuple[Tensor, Tensor]:
        log_probs = self.log_probs
        max_log_probs, max_values = torch.max(log_probs, dim=3)
        max_values = max_values.permute(0, 2, 1)
        return max_log_probs, max_values


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
        probs = weight

        # x: (F, H, Ki, num_samples * B, D) -> (F, H * Ki, num_samples * B, D)
        num_samples = x.shape[3] // weight.shape[1]
        x = x.flatten(1, 2)

        # mixing_distribution: (F, B, Ko, H * Ki)
        mixing_distribution = torch.distributions.Categorical(probs=probs)

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

    def mode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        assert self.semiring == LSESumSemiring
        # x: (F, H, B, Ki) -> (F, B, H * Ki)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        # weight: (F, B, Ko, H * Ki)
        weight = torch.log(self.weight) + x.unsqueeze(dim=2)
        max_probs, max_values = torch.max(weight, dim=3)
        return max_probs, max_values

    def argmax(self, x: Tensor, max_vals: Tensor) -> Tensor:
        assert self.semiring == LSESumSemiring
        # x: (F, H, Ki, num_argmax * B, D) -> (F, H * Ki, num_argmax * B, D)
        num_argmax = x.shape[3] // self.weight.shape[1]
        x = x.flatten(start_dim=1, end_dim=2)
        # max_vals: (F, B, Ko) -> (F, Ko, B) -> (F, Ko, 1, B, 1)
        max_vals = max_vals.permute(0, 2, 1).unsqueeze(dim=2).unsqueeze(dim=-1)
        # max_vals: (F, Ko, num_argmax * B, D)
        max_vals = max_vals.broadcast_to(
            max_vals.shape[0],
            max_vals.shape[1],
            num_argmax,
            max_vals.shape[3],
            x.shape[3]
        )
        max_vals = max_vals.flatten(start_dim=2, end_dim=3)
        return torch.gather(x, dim=1, index=max_vals)
