from __future__ import annotations
from abc import ABC, abstractmethod
from pyfilter.types.random_variables import GaussianRV
from pyfilter.hints import FloatArray

type Variable = GaussianRV | FloatArray


class LinearTransformBase[State: Variable](ABC):
    @property
    @abstractmethod
    def matrix(self) -> FloatArray:
        """The matrix implied by the transform."""

    @abstractmethod
    def transform(self, x: State) -> State:
        """Transform the state."""

    def __matmul__(self, x: State) -> State:
        return self.transform(x)


class InvertibleLinearTransform[State: Variable](LinearTransformBase[State]):
    @abstractmethod
    def inverse(self) -> FloatArray:
        """The inverse of the transform."""


class GenericLinearTransform[State: Variable](LinearTransformBase):
    def __init__(self, A: FloatArray) -> None:
        super().__init__()
        self._A = A

    @property
    def matrix(self) -> FloatArray:
        return self._A

    def transform(self, x: State) -> State:
        return self._A @ x
