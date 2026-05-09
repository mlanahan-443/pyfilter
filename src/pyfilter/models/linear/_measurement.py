from typing import override

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from pyfilter.config import FDTYPE_ as FDYTPE
from pyfilter.hints import FloatArray, IntArr
from pyfilter.types import Covariance, GaussianRV, RandomVariable

from ._base import LinearTransformBase


class SelectionTransform[State: RandomVariable](LinearTransformBase[State]):
    """Selects a subset of state components by integer indices.

    Equivalent to left-multiplication by a selection matrix S of shape
    (m, n), where row i is e_{indices[i]}^T. Bypasses both the
    materialization of S and any matmul by indexing the state's
    component axis directly.
    """

    def __init__(
        self, indices: IntArr | slice, input_dim: int, dtype: DTypeLike = FDYTPE
    ) -> None:
        super().__init__(dtype=dtype)
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self._index_or_slice = indices
        self._input_dim = input_dim

    @property
    def indices(self) -> IntArr:
        if isinstance(self._index_or_slice, slice):
            return np.arange(
                self._index_or_slice.start,
                self._index_or_slice.stop,
                self._index_or_slice.step,
            )

        return np.asarray(self._index_or_slice).astype(np.int32)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return int(self.indices.size)

    @property
    @override
    def matrix(self) -> FloatArray:
        M = np.zeros((self.output_dim, self._input_dim), dtype=self.dtype)
        M[np.arange(self.output_dim), self.indices] = 1.0
        return M

    @override
    def transform(self, x: State) -> State:
        """Selection matrix transform."""
        return x[..., self._index_or_slice]

    def transform_array[arrT: ArrayLike](self, x: arrT) -> arrT:
        """Selection into an array."""
        return x[..., self._index_or_slice]

    def transform_covariance[covT: Covariance](self, cov: covT) -> covT:
        return cov[..., self._index_or_slice, self._index_or_slice]


class GaussianSelectionTransform(SelectionTransform[GaussianRV]):
    """Selection for gausssian random variables."""

    @override
    def transform(self, x: GaussianRV) -> GaussianRV:
        """Selection for gaussian random variables."""
        return x.marginal(self._index_or_slice)
