from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from pyfilter.hints import FloatArray
from pyfilter.types import RandomVariable


class LinearTransformBase[State: RandomVariable](ABC):
    @property
    @abstractmethod
    def matrix(self) -> FloatArray:
        """The matrix implied by the transform."""

    @abstractmethod
    def transform(self, x: State) -> State:
        """Transform the state."""

    def __matmul__(self, x: State) -> State:
        return self.transform(x)


class InvertibleLinearTransform[State: RandomVariable](LinearTransformBase[State]):
    @abstractmethod
    def inverse(self) -> FloatArray:
        """The inverse of the transform."""


class GenericLinearTransform[State: RandomVariable](LinearTransformBase[State]):
    def __init__(self, A: FloatArray) -> None:
        super().__init__()
        self._A = A

    @property
    def matrix(self) -> FloatArray:
        return self._A

    def transform(self, x: State) -> State:
        return self._A @ x


class LinearTransitionBase[State: RandomVariable, Time: FloatArray](ABC):
    """Base linear transition."""

    @abstractmethod
    def transform(self, x: State, dt: Time) -> State:
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
class HasInverseTransform[State: RandomVariable, Time: FloatArray](Protocol):
    """Transition is invertible."""

    def inverse_transform(self, x: State, dt: Time) -> State: ...


class LTI_Transition[State: RandomVariable, Time: FloatArray](
    LinearTransitionBase[State, Time]
):
    def __init__(self, A: FloatArray) -> None:
        super().__init__()
        self._A = A

    def matrix(self, dt: FloatArray) -> FloatArray:
        return self._A

    def transform(self, x: State, dt: Time) -> State:
        return self._A @ x

    def inverse(self, dt: Time) -> FloatArray:
        return np.linalg.inv(self._A)
