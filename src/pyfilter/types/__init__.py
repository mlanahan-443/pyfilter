from typing import Any

from ..hints import FloatArray
from .covariance import CovarianceBase
from .random_variables import GaussianRV

type RandomVariable = GaussianRV[Any] | FloatArray
type Covariance = CovarianceBase | FloatArray

__all__ = [
    "CovarianceBase",
    "GaussianRV",
    "Covariance",
    "RandomVariable",
]
