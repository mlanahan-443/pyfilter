from typing import Any

from ..hints.jax_hints import JaxFloatArray
from .covariance import (
    CholeskyFactorCovariance,
    CovarianceBase,
    DiagonalCovariance,
    InformationCovariance,
)
from .linear_solve import solve_symmetric
from .random_variables import GaussianRV

type RandomVariable = GaussianRV[Any] | JaxFloatArray
type Covariance = CovarianceBase | JaxFloatArray

__all__ = [
    "CovarianceBase",
    "CholeskyFactorCovariance",
    "DiagonalCovariance",
    "InformationCovariance",
    "GaussianRV",
    "Covariance",
    "RandomVariable",
    "solve_symmetric",
]
