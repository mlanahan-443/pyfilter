import numpy as np
import pytest
from numpy.random import default_rng
from numpy.typing import NDArray

from pyfilter.filter.linear import (
    LinearGaussianKalman,
    square_root_quadratic_update,
)
from pyfilter.models.linear_transform import GenericLinearTransform
from pyfilter.models.linear_transition import LTI_Transition
from pyfilter.types.covariance import CholeskyFactorCovariance
from pyfilter.types.process_noise import ProcessNoise
from pyfilter.types.random_variables import GaussianRV


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


class SimpleProcessNoise(ProcessNoise):
    """Simple constant process noise for testing."""

    def __init__(self, shape: tuple):
        super().__init__(shape)
        self._cov = np.zeros(shape)
        np.fill_diagonal(self._cov, 1e-2)

    def covariance(self, dt: NDArray[np.float64]) -> GaussianRV:
        return GaussianRV.zero_mean(self._cov)


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


def test_linear_gaussian_kalman_basic():
    """Test basic LinearGaussianKalman functionality with LTI_Transition."""
    # Initialize state: 4D state vector
    init_mean = np.zeros(4)
    init_cov = np.eye(4) * 1e-2
    init_state = GaussianRV(init_mean, init_cov)

    # Transition model: identity (state doesn't change)
    A = np.eye(4)
    transition_model = LTI_Transition(A)

    # Process noise
    process_noise = SimpleProcessNoise((4, 4))

    # Measurement model: observe first 2 components
    H = np.zeros((2, 4))
    H[0, 0] = 1
    H[1, 1] = 1
    measurement_model = GenericLinearTransform(H)

    # Create filter
    kalman_filter = LinearGaussianKalman(
        transition_model, process_noise, measurement_model
    )

    # Generate random measurements
    generator = default_rng(seed=42)
    num_steps = 10
    dt = np.array(0.1)

    state = init_state
    for _i in range(num_steps):
        # Predict
        prediction = kalman_filter.predict(state, dt)

        # Generate measurement
        meas_val = generator.random(2)
        meas_cov = np.eye(2) * 0.1
        measurement = GaussianRV(meas_val, meas_cov)

        # Update
        innovation = kalman_filter.innovation(prediction, measurement)
        state = kalman_filter.update(prediction, innovation)

        # Verify state is valid
        assert state.mean.shape == (4,)
        assert state.covariance.shape == (4, 4)
        assert np.all(np.isfinite(state.mean))


def test_lti_transition():
    """Test LTI_Transition class."""
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    transition = LTI_Transition(A)

    # Test matrix method
    dt = np.array(1.0)
    assert np.allclose(transition.matrix(dt), A)

    # Test transform method
    state = GaussianRV(np.array([1.0, 2.0]), np.eye(2))
    transformed = transition.transform(state, dt)

    expected_mean = A @ state.mean
    assert np.allclose(transformed.mean, expected_mean)

    # Test inverse
    A_inv = transition.inverse(dt)
    assert np.allclose(A @ A_inv, np.eye(2))


def test_generic_linear_transform():
    """Test GenericLinearTransform class."""
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3x2 matrix
    transform = GenericLinearTransform(H)

    # Test matrix property
    assert np.allclose(transform.matrix, H)

    # Test transform method
    state = GaussianRV(np.array([2.0, 3.0]), np.eye(2) * 0.5)
    transformed = transform.transform(state)

    expected_mean = H @ state.mean
    assert np.allclose(transformed.mean, expected_mean)
    assert transformed.mean.shape == (3,)
