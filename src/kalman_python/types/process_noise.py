from abc import ABC, abstractmethod
from ..hints import FloatArray
from .covariance import CovarianceBase

type Covariance = FloatArray | CovarianceBase


class ProcessNoise(ABC):
    def __init__(self, shape: tuple):
        self.shape = shape

    @abstractmethod
    def covariance(self, dt: FloatArray) -> Covariance:
        pass

    def __call__(self, dt: FloatArray) -> Covariance:
        return self.covariance(dt)
