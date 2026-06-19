from __future__ import annotations

from jax import numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky

from pyfilter.hints.jax_hints import JaxFloatArray
from pyfilter.types.covariance import (
    CholeskyFactorCovariance,
    CovarianceBase,
    DiagonalCovariance,
    solve_cholesky_covariance,
    solve_diagonal_covariance,
)

def solve_symmetric_dense_array(
    A: JaxFloatArray, B: JaxFloatArray, overwrite_b: bool = False
) -> JaxFloatArray:
    """
    Solve a symmetric system of equations using cholesky's
    decomposition

    Parameters
    ----------
    A: NDArray[DTYPE]
        nxn coefficient matrix
    B: NDArray[DTYPE]
        n x m data matrix/vector

    Returns
    ----------
    NDArray[DTYPE]
        n x m result

    NOTE: this in theory should be more efficient (though less robust) than
          jax.scipy's linalg.solve(*,assume_a = 'symmetric'), which uses LAPACK's
          'SYSV'. This uses the Bunch-Kaufman diagonal pivoting algorithm
          which can handle indefinite matrices, however we are gaurenteed that
          our matrix is PD
    """

    L = cholesky(A, overwrite_a=overwrite_b, lower=True)
    result: JaxFloatArray = cho_solve((L, True), B)
    return result


def solve_symmetric(
    A: JaxFloatArray | CovarianceBase | DiagonalCovariance | CholeskyFactorCovariance,
    B: JaxFloatArray,
    overwrite_b: bool = False,
) -> JaxFloatArray:
    """
    Solve a symmetric system of equations using cholesky's
    decomposition

    Args:
        A: nxn coefficient matrix.
        B: The result of the linear transformation.

    Returns
    ----------
        The variable.
    """

    _ALLOWED_TYPES = "FloatingArr,CholeskyFactorCovariance,DiagonalCovariance"
    if isinstance(A, jnp.ndarray):
        return solve_symmetric_dense_array(A, B, overwrite_b=overwrite_b)
    elif isinstance(A, CholeskyFactorCovariance):
        return solve_cholesky_covariance(A, B, overwrite_b=overwrite_b)
    elif isinstance(A, DiagonalCovariance):
        return solve_diagonal_covariance(A, B)
    else:
        msg = f"A must be one of: {_ALLOWED_TYPES}, not {type(A)}"
        raise TypeError(msg)
