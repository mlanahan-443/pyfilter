from typing import override

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike

from pyfilter.hints.jax_hints import JaxFloatArray, JaxIntArray
from pyfilter.types import Covariance, GaussianRV, RandomVariable
import equinox as eqx
from ._base import LinearTransformBase


class SelectionTransform[State: RandomVariable](LinearTransformBase[State]):
    """Selects a subset of state components by integer indices.

    Equivalent to left-multiplication by a selection matrix S of shape
    (m, n), where row i is e_{indices[i]}^T. Bypasses both the
    materialization of S and any matmul by indexing the state's
    component axis directly.
    """

    key: JaxIntArray | slice = eqx.field(static = True)
    input_dim: int

    @property
    def indices(self) -> JaxIntArray:
        if isinstance(self.key, slice):
            return jnp.arange(
                self.key.start,
                self.key.stop,
                self.key.step,
            )

        return jnp.asarray(self.key).astype(jnp.integer)

    @property
    def output_dim(self) -> int:
        return int(self.indices.size)

    @property
    @override
    def matrix(self) -> JaxFloatArray:
        """Form the selection matrix explicitly."""
        return jax.nn.one_hot(self.indices, self.input_dim)

    @override
    def transform(self, x: State) -> State:
        """Selection matrix transform."""
        return x[..., self.key]

    def transform_array[arrT: ArrayLike](self, x: arrT) -> arrT:
        """Selection into an array."""
        return x[..., self.key]

    def transform_covariance[covT: Covariance](self, cov: covT) -> covT:
        return cov[..., self.key, self.key]


class GaussianSelectionTransform(SelectionTransform[GaussianRV]):
    """Selection for gausssian random variables."""

    @override
    def transform(self, x: GaussianRV) -> GaussianRV:
        """Selection for gaussian random variables."""
        return x.marginal(self.key)
