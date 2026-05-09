from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from pyfilter.config import FDTYPE_ as FDTYPE
from pyfilter.hints import FloatArray
from pyfilter.types import Covariance, RandomVariable


@dataclass(frozen=True)
class LinearTransformBase[State: RandomVariable](ABC):
    dtype: DTypeLike = FDTYPE

    @property
    @abstractmethod
    def matrix(self) -> FloatArray:
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
    def __init__(self, A: FloatArray):
        self._A = A
        self.__setattr__("dtype", self._A.dtype)

    @property
    def matrix(self) -> FloatArray:
        return self._A

    def transform(self, x: State) -> State:
        return self._A @ x

    def transform_array[arrT: ArrayLike](self, x: arrT) -> arrT:
        return self._A @ x

    @abstractmethod
    def transform_covariance[covT: Covariance](self, cov: covT) -> covT:
        """Transform a covariance."""
        if isinstance(cov, np.ndarray):
            return np.einsum(
                "...ij,...jk,...lk->...il", self._A, cov, self._A, optimize=True
            )

        return cov.quadratic_form(self._A)


@dataclass(frozen=True)
class LinearTransitionBase[State: RandomVariable](ABC):
    """Base linear transition."""

    dtype: DTypeLike = FDTYPE

    @abstractmethod
    def transform(self, x: State, dt: FloatArray) -> State:
        """Transform the state x(k) -> x(k+1)"""


@runtime_checkable
class HasMatrix(Protocol):
    """Transition has an explicit matrix."""

    def matrix(self, dt: FloatArray) -> FloatArray: ...


@runtime_checkable
class HasInverse(Protocol):
    """Transition has an explicit inverse matrix."""

    def inverse(self, dt: FloatArray) -> FloatArray: ...


@runtime_checkable
class HasInverseTransform[State: RandomVariable](Protocol):
    """Transition is invertible."""

    def inverse_transform(self, x: State, dt: FloatArray) -> State: ...


class LTI_Transition[State: RandomVariable](LinearTransitionBase[State]):
    def __init__(self, A: FloatArray) -> None:
        super().__init__(dtype=A.dtype)
        self._A = A

    def matrix(self, dt: FloatArray) -> FloatArray:
        return self._A

    def transform(self, x: State, dt: FloatArray) -> State:
        return self._A @ x

    def inverse(self, dt: FloatArray) -> FloatArray:
        return np.linalg.inv(self._A)
