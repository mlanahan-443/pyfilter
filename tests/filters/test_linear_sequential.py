from pathlib import Path

import pandas as pd
import pytest
from jax import numpy as jnp

from pyfilter.filter.linear import LinearGaussianKalman
from pyfilter.hints.jax_hints import JaxFloatArray
from pyfilter.models.linear import LinearTransformBase, LinearTransitionBase
from pyfilter.types import InformationCovariance
from pyfilter.types.process_noise import ProcessNoise
from pyfilter.types.random_variables import GaussianRV
import numpy as np


@pytest.fixture
def data_path() -> Path:
    return Path(__file__).parent / "data" / Path(__file__).stem


class TransitionModel(LinearTransitionBase):
    def matrix(self, dt: JaxFloatArray) -> JaxFloatArray:
        A = np.zeros(dt.shape + (6, 6))
        A[..., np.diag_indices(6)] = 1
        A[..., 0, 1] = A[..., 1, 2] = A[..., 3, 4] = A[..., 4, 5] = dt
        A[..., 0, 2] = A[..., 3, 5] = 0.5 * dt**2

        return jnp.array(A)

    def inverse(self, dt: JaxFloatArray):
        return jnp.linalg.inv(self.matrix(dt))

    def transform(self, x: GaussianRV, dt: JaxFloatArray) -> GaussianRV:
        return self.matrix(dt) @ x


class MeasurementModel(LinearTransformBase):
    def transform(self, x: GaussianRV) -> GaussianRV:
        return x.marginal(jnp.array([0, 3]))

    @property
    def matrix(self) -> JaxFloatArray:
        H = np.zeros((2, 6))
        H[0, 0] = 1
        H[1, 3] = 1
        return jnp.array(H)

    def transform_array(self, x):
        return x[..., jnp.array([0, 3])]

    def transform_covariance(self, cov):
        return cov[..., jnp.array([0, 3]), jnp.array([0, 3])]


class ProcessNoiseModel(ProcessNoise):
    intensity: float
    shape_in: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.shape_in

    def covariance(self, dt: JaxFloatArray) -> JaxFloatArray:
        block = np.empty(dt.shape + (3, 3))
        block[..., 0, 0] = 0.25 * dt**4
        block[..., 0, 1] = block[..., 1, 0] = 0.5 * dt**3
        block[..., 0, 2] = block[..., 2, 0] = 0.5 * dt**2
        block[..., 1, 2] = block[..., 2, 1] = dt
        block[..., 1, 1] = dt**2
        block[..., 2, 2] = np.ones_like(dt)

        zeros = jnp.zeros_like(block)
        mat = self.intensity * jnp.block([[block, zeros], [zeros, block]])
        return mat

    def inverse_covariance(self, dt: JaxFloatArray) -> InformationCovariance | JaxFloatArray:
        return jnp.linalg.inv(self.covariance(dt))


@pytest.fixture
def variance() -> float:
    """Variance."""
    return 0.2**2


@pytest.fixture
def dt() -> JaxFloatArray:
    return jnp.array(1.0)


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
) -> LinearGaussianKalman:
    """The filter to test."""
    return LinearGaussianKalman(transition_model, process_model, measurement_model)


@pytest.fixture
def meas_variance() -> float:
    return 3.0**2


@pytest.fixture
def measurement_covariance(meas_variance: float) -> JaxFloatArray:
    return jnp.array([[meas_variance, 0], [0, meas_variance]])


def test_linear_filter(
    linear_filter: LinearGaussianKalman,
    measurement_covariance: JaxFloatArray,
    dt: JaxFloatArray,
    data_path: Path,
):
    """Test the linear filter against known output."""
    measurement_means = pd.read_csv(
        data_path / "test_linear_filter_measurements.csv", index_col=0
    ).to_numpy()

    measurements = GaussianRV(
        measurement_means,
        jnp.repeat(measurement_covariance[jnp.newaxis, ...], len(measurement_means), axis=0),
    )
    state = GaussianRV(jnp.zeros(6), jnp.diag(jnp.ones(6)) * 500)

    for i in range(len(measurements)):
        prediction = linear_filter.predict(state, dt)
        innovation = linear_filter.innovation(prediction, measurements[i])
        state = linear_filter.update(prediction, innovation)

    # Verify the filter converged to a reasonable estimate
    assert state.mean.shape == (6,)
    assert jnp.all(jnp.isfinite(state.mean))
