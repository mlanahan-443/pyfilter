from pyfilter.types.process_noise import ProcessNoise
from pyfilter.models.linear import LinearTransformBase, LinearTransitionBase
from pyfilter.types import GaussianRV, InformationCovariance
from pyfilter.hints.jax_hints import JaxFloatArray
import numpy as np
from jax import numpy as jnp


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


class SimpleProcessNoise(ProcessNoise):
    """Simple constant process noise for testing."""

    _shape: int

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._shape,)

    def covariance(self, dt: JaxFloatArray) -> JaxFloatArray:
        return jnp.diag(jnp.ones(self._shape) * 1e-2)

    def inverse_covariance(self, dt: JaxFloatArray) -> InformationCovariance:
        return InformationCovariance(jnp.diag(jnp.ones(self._shape) * 100))


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

    def transform_array(self, x: JaxFloatArray, dt: JaxFloatArray) -> JaxFloatArray:
        return self.matrix(dt) @ x
