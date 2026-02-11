from __future__ import annotations
from scipy.linalg import cholesky, cho_solve
from kalman_python.types.covariance import (
    solve_cholesky_covariance,
    solve_diagonal_covariance,
    DiagonalCovariance,
    CholeskyFactorCovariance,
    CovarianceBase,
)
from kalman_python.hints import FloatArray
import numpy as np


def solve_symmetric_cholesky_dense_array(
    A: FloatArray, B: FloatArray, overwrite_b: bool = False
) -> FloatArray:
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
          scipy's linalg.solve(*,assume_a = 'symmetric'), which uses LAPACK's
          'SYSV'. This uses the Bunch-Kaufman diagonal pivoting algorithm
          which can handle indefinite matrices, however we are gaurenteed that
          our matrix is PD
    """

    L = cholesky(A, overwrite_a=overwrite_b, lower=True)
    return cho_solve((L, True), B)


def solve_symmetric_cholesky(
    A: FloatArray | CovarianceBase | DiagonalCovariance | CholeskyFactorCovariance,
    B: FloatArray,
    overwrite_b: bool = False,
) -> FloatArray:
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
    if isinstance(A, np.ndarray):
        return solve_symmetric_cholesky_dense_array(A, B, overwrite_b=overwrite_b)
    elif isinstance(A, CholeskyFactorCovariance):
        return solve_cholesky_covariance(A, B, overwrite_b=overwrite_b)
    elif isinstance(A, DiagonalCovariance):
        return solve_diagonal_covariance(A, B)
    else:
        msg = f"A must be one of: {_ALLOWED_TYPES}, not {type(A)}"
        raise TypeError(msg)
