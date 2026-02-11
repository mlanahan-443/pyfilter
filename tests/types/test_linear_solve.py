from pyfilter.types.covariance import CholeskyFactorCovariance, DiagonalCovariance
from pyfilter.linear_solve import (
    solve_cholesky_covariance,
    solve_symmetric_cholesky,
    solve_diagonal_covariance,
    solve_symmetric_cholesky_dense_array,
)
from scipy.linalg import cholesky
import numpy as np

np.random.seed(45)


def test_solve_symmetric_cholesky_dense_array():
    """Test that the cholesky solver returns the correct result.

    Construct a PSD matrix, and test that this is the same result
    as returned using the LU decomposition in np.linalg.solve."""
    batch_shape = (100, 10)
    mat_shape = 40
    A_ = np.random.random(size=batch_shape + (mat_shape, mat_shape))
    A = A_ + A_.transpose((0, 1, 3, 2))

    # Force PSD-ness.
    A[:, :, np.diag_indices(mat_shape)] += (
        10 * np.eye(mat_shape)[np.newaxis, np.newaxis, ...]
    )
    b = np.random.random(batch_shape + (mat_shape, mat_shape))

    x_compare = np.linalg.solve(A, b)

    x_cholesky = solve_symmetric_cholesky_dense_array(A, b)

    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using symmetric solver differs from LU decomposition solve",
    )


def test_solve_cholesky_covariance():
    """Test that the cholesky solver returns the correct result.

    Construct a PSD matrix, convert to a cholesky covariance object,
    test that what is returned using the LU decomposition in
    np.linalg.solve is the same."""
    batch_shape = (100, 10)
    mat_shape = 40
    A_ = np.random.random(size=batch_shape + (mat_shape, mat_shape))
    A = A_ + A_.transpose((0, 1, 3, 2))
    # Force PSD-ness.
    A[:, :, np.diag_indices(mat_shape)] += (
        10 * np.eye(mat_shape)[np.newaxis, np.newaxis, ...]
    )
    L = cholesky(A, lower=True)

    b = np.random.random(batch_shape + (mat_shape, mat_shape))

    x_compare = np.linalg.solve(A, b)

    cov = CholeskyFactorCovariance(L)

    x_cholesky = solve_cholesky_covariance(cov, b)

    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using cholesky factor covariance differs from LU decomposition solve.",
    )


def test_solve_diagonal_covariance():
    """Test that the solution using diagonal covariance returns the correct result.

    Construct a Diagonal matrix, convert to a diagonal covariance object,
    test that what is returned using the solver vs. division is the same."""
    batch_shape = (100, 10)
    mat_shape = 40
    d = np.random.random(size=batch_shape + (mat_shape,))

    b = np.random.random(batch_shape + (mat_shape, mat_shape))

    x_compare = b / d[..., np.newaxis]

    cov = DiagonalCovariance(d**0.5)

    x_cholesky = solve_diagonal_covariance(cov, b)
    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using cholesky factor covariance differs from division.",
    )


def test_solve_symmetric_cholesky():
    """Test that the solver dispatcher returns the intended result."""
    batch_shape = (100, 10)
    mat_shape = 40
    A_ = np.random.random(size=batch_shape + (mat_shape, mat_shape))
    A = A_ + A_.transpose((0, 1, 3, 2))

    # Force PSD-ness.
    A[:, :, np.diag_indices(mat_shape)] += (
        10 * np.eye(mat_shape)[np.newaxis, np.newaxis, ...]
    )
    b = np.random.random(batch_shape + (mat_shape, mat_shape))

    x_compare = np.linalg.solve(A, b)

    x_cholesky = solve_symmetric_cholesky(A, b)

    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using symmetric cholesky solver dispatcher with dense array differs from LU decomposition solve",
    )

    L = cholesky(A, lower=True)
    cov = CholeskyFactorCovariance(L)
    x_cholesky = solve_symmetric_cholesky(cov, b)

    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using symmetric cholesky solver with CholeskyFactorCovariance differs from LU decomposition solve.",
    )

    batch_shape = (100, 10)
    mat_shape = 40
    d = np.random.random(size=batch_shape + (mat_shape,))

    b = np.random.random(batch_shape + (mat_shape, mat_shape))

    x_compare = b / d[..., np.newaxis]

    cov = DiagonalCovariance(d**0.5)

    x_cholesky = solve_symmetric_cholesky(cov, b)
    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using symmetric cholesky solver with DiagonalCovariance differs from division.",
    )


if __name__ == "__main__":
    test_solve_diagonal_covariance()
