import numpy as np
from pyfilter.filter.linear import LinearGuassianKalman
from pyfilter.models.linear_transform import LinearTransformBase
from pyfilter.models.linear_transition import LinearTransitionBase
from pyfilter.hints import FloatArray
from pyfilter.types.random_variables import GaussianRV
from pyfilter.types.process_noise import ProcessNoise
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent / "data" / Path(__file__).stem


class TransitionModel(LinearTransitionBase):
    def matrix(self, dt: FloatArray) -> FloatArray:
        A = np.zeros(dt.shape + (6, 6))
        A[..., np.diag_indices(6)] = 1
        A[..., 0, 1] = A[..., 1, 2] = A[..., 3, 4] = A[..., 4, 5] = dt
        A[..., 0, 2] = A[..., 3, 5] = 0.5 * dt**2

        return A

    def inverse(self, dt: FloatArray):
        return np.linalg.inv(self.matrix(dt))

    def transform(self, x: GaussianRV, dt: FloatArray) -> GaussianRV:
        return self.matrix(dt) @ x


class MeasurementModel(LinearTransformBase):
    def transform(self, x: GaussianRV) -> GaussianRV:
        return x.marginal(np.array([0, 3]))

    @property
    def matrix(self) -> FloatArray:
        H = np.zeros((2, 6))
        H[0, 0] = 1
        H[1, 3] = 1
        return H


class ProcessNoiseModel(ProcessNoise):
    def __init__(self, intensity: float, shape: tuple):
        super().__init__(shape)
        self._intensity = intensity

    def covariance(self, dt: FloatArray) -> GaussianRV:
        block = np.empty(dt.shape + (3, 3))
        block[..., 0, 0] = 0.25 * dt**4
        block[..., 0, 1] = block[..., 1, 0] = 0.5 * dt**3
        block[..., 0, 2] = block[..., 2, 0] = 0.5 * dt**2
        block[..., 1, 2] = block[..., 2, 1] = dt
        block[..., 2, 2] = np.ones_like(dt)

        zeros = np.zeros_like(block)
        mat = self._intensity * np.block([[block, zeros], [zeros, block]])
        return GaussianRV.zero_mean(mat)


@pytest.fixture
def variance() -> float:
    """Variance."""
    return 0.2**2


@pytest.fixture
def dt() -> FloatArray:
    return np.array(1.0)


@pytest.fixture
def measurement_model() -> MeasurementModel:
    """An instance of the measurement model."""
    return MeasurementModel()


@pytest.fixture
def process_model(variance: float) -> ProcessNoiseModel:
    """The process noise model."""
    return ProcessNoiseModel(variance, (6, 6))


@pytest.fixture
def transition_model() -> TransitionModel:
    """The transition model."""
    return TransitionModel()


@pytest.fixture
def linear_filter(
    transition_model: TransitionModel,
    process_model: ProcessNoiseModel,
    measurement_model: MeasurementModel,
) -> LinearGuassianKalman:
    """The filter to test."""
    return LinearGuassianKalman(transition_model, process_model, measurement_model)


@pytest.fixture
def meas_variance() -> float:
    return 3.0**2


@pytest.fixture
def measurement_covariance(meas_variance: float) -> FloatArray:
    return np.array([[meas_variance, 0], [0, meas_variance]])


def test_linear_filter(
    linear_filter: LinearGuassianKalman,
    measurement_covariance: FloatArray,
    dt: FloatArray,
    data_path: Path,
):
    """Test the linear filter against known output."""
    measurement_means = pd.read_csv(
        data_path / "test_linear_filter_measurements.csv", index_col=0
    ).to_numpy()

    measurements = GaussianRV(
        measurement_means,
        np.repeat(
            measurement_covariance[np.newaxis, ...], len(measurement_means), axis=0
        ),
    )
    state = GaussianRV(np.ones(6), np.diag(np.ones(6)) * 10)

    for i in range(len(measurements)):
        prediction = linear_filter.predict(state, dt)
        innovation = linear_filter.innovation(prediction, measurements[i])
        state = linear_filter.update(prediction, innovation)
        print(state.mean)

    print(state.mean)
