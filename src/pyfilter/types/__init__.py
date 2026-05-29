from typing import Any

from ..hints import JaxFloatArray
from .covariance import CovarianceBase
from .random_variables import GaussianRV

type RandomVariable = GaussianRV[Any] | JaxFloatArray
type Covariance = CovarianceBase | JaxFloatArray

__all__ = [
    "CovarianceBase",
    "GaussianRV",
    "Covariance",
    "RandomVariable",
]
