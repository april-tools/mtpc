from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, distributions

from cirkit.backend.torch.layers import TorchExpFamilyLayer, TorchInnerLayer
from cirkit.backend.torch.semiring import Semiring, SumProductSemiring


class TorchBatchedCategoricalLayer(TorchExpFamilyLayer):
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        num_channels: int = 1,
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
            num_channels: The number of channels.
            num_categories: The number of categories for Categorical distribution.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].

        Raises:
            ValueError: If the scope contains more than one variable.
            ValueError: If the number of categories is negative.
        """
        if num_channels != 1:
            raise NotImplementedError(
                "The batched categorical layer requires num_channels=1"
            )
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
            num_channels=num_channels,
            semiring=semiring,
        )
        self.num_categories = num_categories
        self._probs: Tensor | None = None

    @property
    def probs(self) -> Tensor:
        if self._probs is None:
            raise ValueError("No probs have been set")
        return self._probs

    @probs.setter
    def probs(self, probs: Tensor | None):
        if probs is not None:
            if (
                len(probs.shape) != 4
                or probs.shape[0] != self.num_folds
                or probs.shape[2] != self.num_output_units
                or probs.shape[3] != self.num_categories
            ):
                raise ValueError(
                    f"Expected probs of shape ({self.num_folds}, -1, {self.num_output_units}, {self.num_categories}), "
                    f"but found {probs.shape}"
                )
        self._probs = probs

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_output_units": self.num_output_units,
            "num_channels": self.num_channels,
            "num_categories": self.num_categories,
        }

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Categorical should be discrete
        # x: (F, C, B, 1) -> (F, B)
        x = x.squeeze(dim=3).squeeze(dim=1)
        # probs: (F, B, K, N)
        probs = self.probs
        idx_fold = torch.arange(self.num_folds, device=probs.device)
        idx_batch = torch.arange(x.shape[1], device=probs.device)
        # y: (F, B, K)
        y = probs[idx_fold[:, None], idx_batch[None, :], :, x]
        return self.semiring.map_from(y, SumProductSemiring)

    def log_partition_function(self) -> Tensor:
        return torch.zeros(
            size=(self.num_folds, 1, self.num_output_units), device=self.probs.device
        )

    def sample(self, num_samples: int = 1) -> Tensor:
        # probs: (F, B, K, N)
        probs = self.probs
        dist = distributions.Categorical(probs=probs)
        # samples: (num_samples, F, B, K)
        samples = dist.sample((num_samples,))
        # samples: (F, K, num_samples, B) -> (F, K, num_samples * B)
        samples = samples.permute(1, 3, 0, 2)
        return samples.flatten(start_dim=2)


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
                    f"Expected probs of shape ({self.num_folds}, -1, {self.num_output_units}, {self.arity * self.num_input_units}), "
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

    def forward(self, x: Tensor) -> Tensor:
        # x: (F, H, B, Ki) -> (F, B, H * Ki)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        # weight: (F, B, Ko, H * Ki)
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

        # x: (F, H, C, Ki, num_samples * B, D) -> (F, C, H * Ki, num_samples * B, D)
        x = x.permute(0, 2, 1, 3, 4, 5).flatten(2, 3)
        num_samples = x.shape[3]

        # mixing_distribution: (F, B, Ko, H * Ki)
        mixing_distribution = torch.distributions.Categorical(probs=weight)

        # mixing_samples: (num_samples, B, F, Ko) -> (F, Ko, num_samples, B) -> (F, Ko, num_samples * B)
        mixing_samples = mixing_distribution.sample((num_samples,))
        mixing_samples = mixing_samples.permute(2, 3, 0, 1)
        mixing_samples = mixing_samples.flatten(start_dim=2)

        # mixing_indices: (F, 1, Ko, num_samples * B, 1) -> (F, C, Ko, num_samples * B, D)
        mixing_indices = mixing_samples.unsqueeze(dim=1).unsqueeze(dim=-1)
        mixing_indices = mixing_indices.broadcast_to(
            mixing_samples.shape[0],
            x.shape[1],
            mixing_samples.shape[2],
            mixing_samples.shape[3],
            x.shape[4],
        )

        # x: (F, C, Ko, num_samples * B, D)
        x = torch.gather(x, dim=2, index=mixing_indices)
        return x, mixing_samples
