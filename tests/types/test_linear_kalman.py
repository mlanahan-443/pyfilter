from kalman_python.filter.linear import (
    LinearGuassianKalman,
    LinearPredictor,
    GaussianRV,
    square_root_quadratic_update,
)
from kalman_python.types.process_noise import ProcessNoise
import numpy as np
from numpy.typing import NDArray
from numpy.random import default_rng
import pytest
from kalman_python.types.covariance import CholeskyFactorCovariance


@pytest.fixture
def dim() -> int:
    """Fixed matrix size (e.g. 4x4) for all tests."""
    return 4


@pytest.fixture(params=[2, 3, 4])
def ndim(request) -> int:
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
def P_full1(dim: int, batch_shape: tuple[int, ...]) -> np.ndarray:
    full_shape = batch_shape + (dim, dim)

    A = np.random.rand(*full_shape)
    A_T = np.swapaxes(A, -1, -2)
    P = A @ A_T + dim * np.eye(dim)
    return P


@pytest.fixture
def P_full2(dim: int, batch_shape: tuple[int, ...]) -> np.ndarray:
    """Returns random, positive-definite matrices with correct batch shape."""
    full_shape = batch_shape + (dim, dim)

    A = np.random.rand(*full_shape)
    A_T = np.swapaxes(A, -1, -2)
    P = A @ A_T + dim * np.eye(dim)
    return P


@pytest.fixture
def L_factor1(P_full1: np.ndarray) -> np.ndarray:
    """Returns the true lower-triangular Cholesky factor of P."""
    return np.linalg.cholesky(P_full1)


@pytest.fixture
def L_factor2(P_full2: np.ndarray) -> np.ndarray:
    """Returns the true lower-triangular Cholesky factor of P."""
    return np.linalg.cholesky(P_full2)


@pytest.fixture
def chol_cov1(L_factor1: np.ndarray) -> CholeskyFactorCovariance:
    return CholeskyFactorCovariance(L_factor1.copy())


@pytest.fixture
def chol_cov2(L_factor2: np.ndarray) -> CholeskyFactorCovariance:
    return CholeskyFactorCovariance(L_factor2.copy())


@pytest.fixture
def A1(dim: int, batch_shape: tuple[int, ...]) -> np.ndarray:
    """Returns a random transformation matrix A."""
    # We make A batched as well to test full batch-on-batch operations
    return np.random.rand(*(batch_shape + (dim, dim))) + 0.1


@pytest.fixture
def A2(dim: int, batch_shape: tuple[int, ...]) -> np.ndarray:
    """Returns a random transformation matrix A."""
    # We make A batched as well to test full batch-on-batch operations
    return np.random.rand(*(batch_shape + (dim, dim))) + 0.1


class ProcessNoiseClass(ProcessNoise):
    def covariance(self, dt: NDArray[np.float64]) -> NDArray[np.float64]:
        cov = np.zeros(self.shape)
        np.fill_diagonal(cov, 1e-2)
        return cov


def test_square_root_quadratic_update(
    A1: np.ndarray,
    chol_cov1: CholeskyFactorCovariance,
    A2: np.ndarray,
    chol_cov2: CholeskyFactorCovariance,
):
    result = square_root_quadratic_update(chol_cov1, A1, chol_cov2, A2)

    check = A1 @ chol_cov1.full() @ A1.swapaxes(
        -2, -1
    ) + A2 @ chol_cov2.full() @ A2.swapaxes(-2, -1)

    np.testing.assert_allclose(check, result.full())


def test_linear_guassian_kalman():
    init_mean = np.zeros(4)
    init_cov = np.zeros((4, 4))
    init_state = GaussianRV(init_mean, init_cov)

    A = np.zeros((4, 4))
    np.fill_diagonal(init_cov, 1e-2)
    np.fill_diagonal(A, 1)
    H = np.zeros((2, 4))
    H[0, 0] = 1
    H[1, 1] = 1
    process_noise = ProcessNoiseClass(init_cov.shape)
    predictor = LinearPredictor(A, process_noise)
    linear_kalman = LinearGuassianKalman(predictor, H)

    generator = default_rng()
    measurements = generator.random((10, 2))
    dt = np.ones((10, 1)) * 1e-1
    prediction = linear_kalman.predict(init_state, np.array([0.0]))
    for i in range(measurements.shape[0]):
        measurement_residual = linear_kalman.innovation(prediction, measurements[i])
        updated = linear_kalman.update(prediction, measurement_residual)
        prediction = linear_kalman.predict(updated, dt[i])


if __name__ == "__main__":
    test_linear_guassian_kalman()
