import pytest
from pyfilter.types.covariance import (
    CholeskyFactorCovariance,
    DiagonalCovariance,
    InformationCovariance,
)
from jax import numpy as jnp
import jax


@pytest.fixture
def dim() -> int:
    """Fixed matrix size (e.g. 4x4) for all tests."""
    return 4


@pytest.fixture(params=[2, 3, 4])
def ndim(request) -> int:
    """
    Parameterizes the rank of the array.
    2 = Single Matrix (N, N)
    3 = Batch of Matrices (B, N, N)
    4 = Batch of Batches (B1, B2, N, N)
    """
    return request.param


@pytest.fixture
def batch_shape(ndim: int) -> tuple[int, ...]:
    """
    Derives the batch shape from the total ndim.
    """
    # Define some arbitrary batch sizes for testing
    # Using different sizes (5, 3) helps catch broadcasting bugs
    full_batch_sizes = (5, 3, 2)

    # Return the slice corresponding to the extra dimensions
    # ndim=2 -> (), ndim=3 -> (5,), ndim=4 -> (5, 3)
    return full_batch_sizes[: ndim - 2]


@pytest.fixture
def P_full(dim: int, batch_shape: tuple[int, ...]) -> jnp.ndarray:
    """Returns random, positive-definite matrices with correct batch shape."""
    # Shape becomes (*batch_shape, dim, dim)
    full_shape = batch_shape + (dim, dim)

    key = jax.random.key(44)
    A = jax.random.uniform(key=key, shape=full_shape)

    # Make positive definite: A @ A.T + I
    # We use swapaxes to transpose only the last two dimensions for the batch
    A_T = jnp.swapaxes(A, -1, -2)
    P = A @ A_T + dim * jnp.eye(dim)
    return P


@pytest.fixture
def L_factor(P_full: jnp.ndarray) -> jnp.ndarray:
    """Returns the true lower-triangular Cholesky factor of P."""
    # NOTE: We use jnp.linalg.cholesky here because it supports
    # batch dimensions natively, whereas jax.scipy.linalg.cho_factor does not.
    return jnp.linalg.cholesky(P_full)


@pytest.fixture
def chol_cov(L_factor: jnp.ndarray) -> CholeskyFactorCovariance:
    return CholeskyFactorCovariance(L_factor.copy())


@pytest.fixture
def diag_std(dim: int, batch_shape: tuple[int, ...]) -> jnp.ndarray:
    """Returns random standard deviations with correct batch shape."""
    # Shape becomes (*batch_shape, dim)
    return jax.random.uniform(key=jax.random.key(44), shape=batch_shape + (dim,)) + 0.5


@pytest.fixture
def diag_cov(diag_std: jnp.ndarray) -> DiagonalCovariance:
    return DiagonalCovariance(diag_std.copy())


@pytest.fixture
def info_cov(P_full: jnp.ndarray) -> InformationCovariance:
    """Create an InformationCovariance from Lambda = Sigma^{-1}"""
    Lambda = jnp.linalg.inv(P_full)
    return InformationCovariance(Lambda.copy())
