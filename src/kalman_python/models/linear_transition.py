from __future__ import annotations
from abc import ABC, abstractmethod
from kalman_python.types.random_variables import GaussianRV
from kalman_python.hints import FloatArray
import numpy as np

type Variable = GaussianRV[GaussianRV] | GaussianRV[FloatArray] | FloatArray


class LinearTransitionBase[State: Variable, Time: FloatArray](ABC):
    
    @abstractmethod
    def matrix(self, dt: FloatArray) -> FloatArray:
        """The matrix implied by the transform."""

    @abstractmethod
    def transform(self, x: State, dt: Time) -> State:
        """Transform the state."""

    @abstractmethod
    def inverse(self, dt: Time) -> FloatArray:
        """The inverse of the transform."""


class LTI_Transition[State: Variable, Time: FloatArray](
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
