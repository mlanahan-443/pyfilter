import numpy as np
import pytest
from numpy.random import default_rng

from pyfilter.filter.linear import (
    LinearGaussianKalman,
    SquareRootLinearGuassianKalman,
)
from pyfilter.hints import FloatArray
from pyfilter.models.linear import GenericLinearTransform, LTI_Transition
from pyfilter.types.covariance import CholeskyFactorCovariance, DiagonalCovariance
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

    def covariance(self, dt: FloatArray) -> GaussianRV:
        return GaussianRV.zero_mean(self._cov)


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

        # Update directly from measurement
        state = kalman_filter.update(prediction, measurement)

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


def test_square_root_kalman_basic():
    """Test basic SquareRootLinearGaussianKalman functionality."""
    # Initialize state: 4D state vector with Cholesky covariance
    init_mean = np.zeros(4)
    init_L = np.eye(4) * 0.1  # Cholesky factor
    init_state = GaussianRV(init_mean, CholeskyFactorCovariance(init_L))

    # Transition model: identity
    A = np.eye(4)
    transition_model = LTI_Transition(A)

    # Process noise
    process_noise = SimpleProcessNoise((4, 4))

    # Measurement model: observe first 2 components
    H = np.zeros((2, 4))
    H[0, 0] = 1
    H[1, 1] = 1
    measurement_model = GenericLinearTransform(H)

    # Create square root filter
    sq_kalman_filter = SquareRootLinearGuassianKalman(
        transition_model, process_noise, measurement_model
    )

    # Generate random measurements
    generator = default_rng(seed=42)
    num_steps = 10
    dt = np.array(0.1)

    state = init_state
    for _i in range(num_steps):
        # Predict
        prediction = sq_kalman_filter.predict(state, dt)

        # Generate measurement
        meas_val = generator.random(2)
        meas_cov = DiagonalCovariance(np.ones(2) * 0.1)
        measurement = GaussianRV(meas_val, meas_cov)

        # Update directly from measurement
        state = sq_kalman_filter.update(prediction, measurement)

        # Verify state is valid
        assert state.mean.shape == (4,)
        assert isinstance(state.covariance, CholeskyFactorCovariance)
        assert state.covariance.cholesky_factor.shape == (4, 4)
        assert np.all(np.isfinite(state.mean))
        assert np.all(np.isfinite(state.covariance.cholesky_factor))


def test_square_root_kalman_vs_standard():
    """Test that SquareRootLinearGaussianKalman produces same results as LinearGaussianKalman."""
    # Initialize state
    init_mean = np.array([1.0, 2.0, 3.0, 4.0])
    init_cov = np.eye(4) * 0.5
    init_L = np.linalg.cholesky(init_cov)

    # Standard filter state
    state_standard = GaussianRV(init_mean.copy(), init_cov.copy())
    # Square root filter state
    state_sq = GaussianRV(init_mean.copy(), CholeskyFactorCovariance(init_L.copy()))

    # Transition model
    A = np.array(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    transition_model = LTI_Transition(A)

    # Process noise
    process_noise = SimpleProcessNoise((4, 4))

    # Measurement model: observe all components
    H = np.eye(4)
    measurement_model = GenericLinearTransform(H)

    # Create both filters
    standard_filter = LinearGaussianKalman(
        transition_model, process_noise, measurement_model
    )
    sq_filter = SquareRootLinearGuassianKalman(
        transition_model, process_noise, measurement_model
    )

    # Run both filters
    generator = default_rng(seed=123)
    num_steps = 5
    dt = np.array(0.1)

    for _i in range(num_steps):
        # Predict
        pred_standard = standard_filter.predict(state_standard, dt)
        pred_sq = sq_filter.predict(state_sq, dt)

        # Verify predictions match
        np.testing.assert_allclose(pred_standard.mean, pred_sq.mean, rtol=1e-10)
        np.testing.assert_allclose(
            pred_standard.covariance, pred_sq.covariance.full(), rtol=1e-10
        )

        # Generate measurement
        meas_val = generator.random(4)
        meas_cov = DiagonalCovariance(np.ones(4) * 0.2)
        measurement_standard = GaussianRV(meas_val.copy(), meas_cov.copy())
        measurement_sq = GaussianRV(meas_val.copy(), meas_cov.copy())

        # Update
        state_standard = standard_filter.update(pred_standard, measurement_standard)
        state_sq = sq_filter.update(pred_sq, measurement_sq)

        # Verify updates match
        np.testing.assert_allclose(state_standard.mean, state_sq.mean, rtol=1e-8)
        np.testing.assert_allclose(
            state_standard.covariance, state_sq.covariance.full(), rtol=1e-8
        )


def test_square_root_kalman_innovation():
    """Test innovation computation for SquareRootLinearGaussianKalman."""
    # Initialize state
    init_mean = np.array([1.0, 2.0])
    init_L = np.eye(2) * 0.1
    init_state = GaussianRV(init_mean, CholeskyFactorCovariance(init_L))

    # Simple transition (identity)
    transition_model = LTI_Transition(np.eye(2))
    process_noise = SimpleProcessNoise((2, 2))

    # Measurement model
    H = np.eye(2)
    measurement_model = GenericLinearTransform(H)

    # Create filter
    sq_filter = SquareRootLinearGuassianKalman(
        transition_model, process_noise, measurement_model
    )

    # Predict
    dt = np.array(0.1)
    prediction = sq_filter.predict(init_state, dt)

    # Create measurement
    meas_val = np.array([1.5, 2.5])
    meas_cov = DiagonalCovariance(np.ones(2) * 0.1)
    measurement = GaussianRV(meas_val, meas_cov)

    # Compute innovation
    innovation = sq_filter.innovation(prediction, measurement)

    # Expected innovation mean: z - H @ x_pred
    expected_innov_mean = meas_val - H @ prediction.mean

    np.testing.assert_allclose(innovation.mean, expected_innov_mean)
    assert innovation.covariance.shape == (2, 2)


def test_innovation_consistency():
    """Test that innovation method is consistent between LinearGaussianKalman and SquareRootLinearGaussianKalman."""
    # Initialize states
    init_mean = np.array([1.0, 2.0, 3.0])
    init_cov = np.diag([0.5, 0.3, 0.2])
    init_L = np.linalg.cholesky(init_cov)

    state_standard = GaussianRV(init_mean.copy(), init_cov.copy())
    state_sq = GaussianRV(init_mean.copy(), CholeskyFactorCovariance(init_L.copy()))

    # Models
    A = np.eye(3)
    transition_model = LTI_Transition(A)
    process_noise = SimpleProcessNoise((3, 3))
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    measurement_model = GenericLinearTransform(H)

    # Create filters
    standard_filter = LinearGaussianKalman(
        transition_model, process_noise, measurement_model
    )
    sq_filter = SquareRootLinearGuassianKalman(
        transition_model, process_noise, measurement_model
    )

    # Predict
    dt = np.array(0.1)
    pred_standard = standard_filter.predict(state_standard, dt)
    pred_sq = sq_filter.predict(state_sq, dt)

    # Measurement
    meas_val = np.array([1.2, 2.3])
    meas_cov = DiagonalCovariance(np.ones(2) * 0.15)
    measurement = GaussianRV(meas_val, meas_cov)

    # Compute innovations
    innov_standard = standard_filter.innovation(pred_standard, measurement)
    innov_sq = sq_filter.innovation(pred_sq, measurement)

    # Innovations should match
    np.testing.assert_allclose(innov_standard.mean, innov_sq.mean, rtol=1e-10)
    # Compare covariance values (handle different representations)
    innov_sq_cov = (
        innov_sq.covariance
        if isinstance(innov_sq.covariance, np.ndarray)
        else innov_sq.covariance.full()
    )
    innov_standard_cov = (
        innov_standard.covariance
        if isinstance(innov_standard.covariance, np.ndarray)
        else innov_standard.covariance.full()
    )

    np.testing.assert_allclose(innov_standard_cov, innov_sq_cov, rtol=1e-10)
