"""Development of a fixed point kalman smoother using jax."""

from dataclasses import dataclass

import jax
from jax import numpy as jnp

from pyfilter.filter.linear import SquareRootLinearGuassianKalman
from pyfilter.hints.jax_hints import JaxFloatArray
from pyfilter.models.linear import GaussianSelectionTransform, IntegratorChainTransition
from pyfilter.types.covariance import CholeskyFactorCovariance, DiagonalCovariance
from pyfilter.types.process_noise import WeinerProcessNoise
from pyfilter.types.random_variables import GaussianRV


def square_root_filter_factory(intensity: JaxFloatArray) -> SquareRootLinearGuassianKalman:
    return SquareRootLinearGuassianKalman(
        IntegratorChainTransition(n=1, p=2),
        WeinerProcessNoise(1, 2, intensity),
        GaussianSelectionTransform(slice(0, 1), 2),
    )


@dataclass
class Simulation:
    """Simulation container."""

    n: int
    dt: float
    sigma: float

    @property
    def generating_transition(self) -> IntegratorChainTransition:
        return IntegratorChainTransition(n=1, p=2)

    @property
    def time_step(self) -> jnp.ndarray:
        return jnp.array(self.dt)

    def scan(self, x: jnp.ndarray, xs) -> tuple[jnp.ndarray, jnp.ndarray]:
        xnew = self.generating_transition.transform(x, self.time_step)
        return xnew, xnew

    def __call__(
        self, x0: jnp.ndarray, key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Do the simulation."""
        dt = jnp.repeat(self.time_step, self.n + 1)
        time = jnp.cumsum(dt)
        _, x = jax.lax.scan(self.scan, x0, length=len(time))
        noise = jax.random.normal(key, shape=(self.n + 1, 2)) * self.sigma

        return time, x, x + noise


@dataclass
class FilterScan:
    filter: SquareRootLinearGuassianKalman

    def step(self, state, xs):
        measurement, dt = xs
        prediction = self.filter.predict(state, dt)
        update = self.filter.update(prediction, measurement)
        return update, (prediction, update)

    def __call__(
        self, init_state: GaussianRV, measurements: GaussianRV, time_steps: jnp.ndarray
    ) -> tuple[GaussianRV, GaussianRV]:
        _, (pred_sq, estimate_sq) = jax.lax.scan(self.step, init_state, (measurements, time_steps))
        return pred_sq, estimate_sq


def linear_pred_filter(
    x_init: JaxFloatArray,
    P_init: JaxFloatArray,
    intensity: float,
    z: JaxFloatArray,
    R: JaxFloatArray,
    dt: float,
) -> tuple[JaxFloatArray, JaxFloatArray]:
    square_root_filter = square_root_filter_factory(jnp.array(intensity))
    L = jnp.linalg.cholesky(P_init)
    state = GaussianRV(x_init.squeeze(), CholeskyFactorCovariance(L))
    time_steps = jnp.repeat(jnp.array(dt), len(z))
    measurements = GaussianRV(z, DiagonalCovariance(R))
    fscan = FilterScan(square_root_filter)
    _, estimate_sq = fscan(state, measurements, time_steps)
    state_pred = square_root_filter.transition_model.transform(estimate_sq[-1:], jnp.array(dt))
    return state_pred.mean, state_pred.covariance.full()


def linear_filter_scan(
    x_init: JaxFloatArray,
    P_init: JaxFloatArray,
    intensity: float,
    z: JaxFloatArray,
    R: JaxFloatArray,
    dt: float,
) -> tuple[tuple[JaxFloatArray, JaxFloatArray], tuple[JaxFloatArray, JaxFloatArray]]:
    square_root_filter = square_root_filter_factory(jnp.array(intensity))
    L = jnp.linalg.cholesky(P_init)
    state = GaussianRV(x_init.squeeze(), CholeskyFactorCovariance(L))
    time_steps = jnp.repeat(jnp.array(dt), len(z))
    measurements = GaussianRV(z, DiagonalCovariance(R))
    fscan = FilterScan(square_root_filter)
    pred_sq, estimate_sq = fscan(state, measurements, time_steps)
    return (
        (estimate_sq.mean, estimate_sq.covariance.full()),
        (pred_sq.mean, pred_sq.covariance.full()),
    )
