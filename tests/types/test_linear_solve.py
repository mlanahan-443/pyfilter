import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.linalg import cholesky

from pyfilter.linear_solve import (
    solve_cholesky_covariance,
    solve_diagonal_covariance,
    solve_symmetric,
    solve_symmetric_dense_array,
)
from pyfilter.types.covariance import CholeskyFactorCovariance, DiagonalCovariance

jax.config.update("jax_enable_x64", True)


def test_solve_symmetric_dense_array():
    """Test that the cholesky solver returns the correct result.

    Construct a PSD matrix, and test that this is the same result
    as returned using the LU decomposition in jnp.linalg.solve."""
    batch_shape = (100, 10)
    mat_shape = 40
    seed = 1701
    key = jax.random.key(seed)
    _, subkey = jax.random.split(key)

    A_ = jax.random.uniform(subkey, shape=batch_shape + (mat_shape, mat_shape))
    A = A_ + A_.mT

    # Force PSD-ness.
    diag = jnp.diag_indices(mat_shape)
    A = A.at[..., diag[0], diag[1]].add(10.0)
    _, subkey = jax.random.split(key)
    b = jax.random.uniform(subkey, batch_shape + (mat_shape, mat_shape))

    x_compare = jnp.linalg.solve(A, b)

    x_cholesky = solve_symmetric_dense_array(A, b)

    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using symmetric solver differs from LU decomposition solve",
    )


def test_solve_cholesky_covariance():
    """Test that the cholesky solver returns the correct result.

    Construct a PSD matrix, convert to a cholesky covariance object,
    test that what is returned using the LU decomposition in
    jnp.linalg.solve is the same."""
    batch_shape = (100, 10)
    mat_shape = 40
    seed = 170144
    key = jax.random.key(seed)
    _, subkey = jax.random.split(key)

    A_ = jax.random.uniform(subkey, shape=batch_shape + (mat_shape, mat_shape))
    A = A_ + A_.transpose((0, 1, 3, 2))
    diag = jnp.diag_indices(mat_shape)
    A = A.at[..., diag[0], diag[1]].add(10.0)
    L = cholesky(A, lower=True)
    _, subkey = jax.random.split(key)

    b = jax.random.uniform(subkey, shape=batch_shape + (mat_shape, mat_shape))

    x_compare = jnp.linalg.solve(A, b)

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
    seed = 170144
    key = jax.random.key(seed)
    _, subkey = jax.random.split(key)

    d = jax.random.uniform(key=subkey, shape=batch_shape + (mat_shape,))

    _, subkey = jax.random.split(key)

    b = jax.random.uniform(key=subkey, shape=batch_shape + (mat_shape, mat_shape))

    x_compare = b / d[..., jnp.newaxis]

    cov = DiagonalCovariance(d**0.5)

    x_cholesky = solve_diagonal_covariance(cov, b)
    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using cholesky factor covariance differs from division.",
    )


def test_solve_symmetric():
    """Test that the solver dispatcher returns the intended result."""
    batch_shape = (100, 10)
    mat_shape = 40
    seed = 170144
    key = jax.random.key(seed)
    _, subkey = jax.random.split(key)

    A_ = jax.random.uniform(key=subkey, shape=batch_shape + (mat_shape, mat_shape))
    A = A_ + A_.transpose((0, 1, 3, 2))

    # Force PSD-ness.
    diag = jnp.diag_indices(mat_shape)
    A = A.at[..., diag[0], diag[1]].add(10.0)
    _, subkey = jax.random.split(key)
    b = jax.random.uniform(key=subkey, shape=batch_shape + (mat_shape, mat_shape))

    x_compare = jnp.linalg.solve(A, b)

    x_cholesky = solve_symmetric(A, b)

    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using symmetric cholesky solver dispatcher with dense array differs from LU decomposition solve",
    )

    L = cholesky(A, lower=True)
    cov = CholeskyFactorCovariance(L)
    x_cholesky = solve_symmetric(cov, b)

    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using symmetric cholesky solver with CholeskyFactorCovariance differs from LU decomposition solve.",
    )

    batch_shape = (100, 10)
    mat_shape = 40
    _, subkey = jax.random.split(key)
    d = jax.random.uniform(key=subkey, shape=batch_shape + (mat_shape,))
    _, subkey = jax.random.split(key)
    b = jax.random.uniform(key=subkey, shape=batch_shape + (mat_shape, mat_shape))

    x_compare = b / d[..., jnp.newaxis]

    cov = DiagonalCovariance(d**0.5)

    x_cholesky = solve_symmetric(cov, b)
    np.testing.assert_allclose(
        x_compare,
        x_cholesky,
        err_msg="X using symmetric cholesky solver with DiagonalCovariance differs from division.",
    )


if __name__ == "__main__":
    test_solve_diagonal_covariance()
