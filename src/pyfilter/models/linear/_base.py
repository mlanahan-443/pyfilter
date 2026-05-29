from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jax import numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from pyfilter.config import FDTYPE_ as FDTYPE
from pyfilter.hints.jax_hints import JaxFloatArray
from pyfilter.types import Covariance, RandomVariable


@dataclass(frozen=True)
class LinearTransformBase[State: RandomVariable](ABC):
    dtype: DTypeLike = FDTYPE

    @property
    @abstractmethod
    def matrix(self) -> JaxFloatArray:
        """The matrix implied by the transform."""

    @abstractmethod
    def transform(self, x: State) -> State:
        """Transform the state."""

    @abstractmethod
    def transform_array[arrT: ArrayLike](self, x: arrT) -> arrT:
        """Transform an array."""

    @abstractmethod
    def transform_covariance[covT: Covariance](self, cov: covT) -> covT:
        """Transform a covariance."""

    def __matmul__(self, x: State) -> State:
        return self.transform(x)


class GenericLinearTransform[State: RandomVariable](LinearTransformBase[State]):
    def __init__(self, A: JaxFloatArray):
        self._A = A
        self.__setattr__("dtype", self._A.dtype)

    @property
    def matrix(self) -> JaxFloatArray:
        return self._A

    def transform(self, x: State) -> State:
        return self._A @ x

    def transform_array[arrT: ArrayLike](self, x: arrT) -> arrT:
        return self._A @ x

    @abstractmethod
    def transform_covariance[covT: Covariance](self, cov: covT) -> covT:
        """Transform a covariance."""
        if isinstance(cov, jnp.ndarray):
            return jnp.einsum(
                "...ij,...jk,...lk->...il", self._A, cov, self._A, optimize=True
            )

        return cov.quadratic_form(self._A)


@dataclass(frozen=True)
class LinearTransitionBase[State: RandomVariable](ABC):
    """Base linear transition."""

    dtype: DTypeLike = FDTYPE

    @abstractmethod
    def transform(self, x: State, dt: JaxFloatArray) -> State:
        """Transform the state x(k) -> x(k+1)"""


@runtime_checkable
class HasMatrix(Protocol):
    """Transition has an explicit matrix."""

    def matrix(self, dt: JaxFloatArray) -> JaxFloatArray: ...


@runtime_checkable
class HasInverse(Protocol):
    """Transition has an explicit inverse matrix."""

    def inverse(self, dt: JaxFloatArray) -> JaxFloatArray: ...


@runtime_checkable
class HasInverseTransform[State: RandomVariable](Protocol):
    """Transition is invertible."""

    def inverse_transform(self, x: State, dt: JaxFloatArray) -> State: ...


class LTI_Transition[State: RandomVariable](LinearTransitionBase[State]):
    def __init__(self, A: JaxFloatArray) -> None:
        super().__init__(dtype=A.dtype)
        self._A = A

    def matrix(self, dt: JaxFloatArray) -> JaxFloatArray:
        return self._A

    def transform(self, x: State, dt: JaxFloatArray) -> State:
        return self._A @ x

    def inverse(self, dt: JaxFloatArray) -> JaxFloatArray:
        return jnp.linalg.inv(self._A)
