"""Development of fixed lag smoothing."""

from typing import NamedTuple

import jax
import matplotlib.pyplot as plt
import numpy as np
from common import Simulation, linear_pred_filter
from jax import numpy as jnp
from jax import scipy as jscipy

from pyfilter.hints.jax_hints import JaxFloatArray
from pyfilter.models.linear import (
    GaussianSelectionTransform,
)
from pyfilter.types.process_noise import WeinerProcessNoise
from pyfilter.types.random_variables import GaussianRV


class _FLCarry(NamedTuple):
    """State entering the step for z_k. Slot i is lag i."""

    x: JaxFloatArray  # (N+2, n)     x[i] = x̂_{k-i, k-1}
    P: JaxFloatArray  # (N+2, n, n)  P[i] = P_k^{i,i}
    Sig: JaxFloatArray  # (N+2, n, n)  Sig[i] = P_k^{0,i}


def smooth_fl(
    z: JaxFloatArray,
    R: JaxFloatArray,
    x_init: JaxFloatArray,
    P_init: JaxFloatArray,
    H: JaxFloatArray,
    F: JaxFloatArray,
    Q: JaxFloatArray,
    lag: int,
) -> tuple[JaxFloatArray, JaxFloatArray]:
    """Fixed lag smoothing of x.

    Args:
        z: Measurements[j] at time[j], j = 0,...,N
        R: The measurement covariance.
        x_init: The initial filter mean estimate.
        P_init: The initial filter covariance estimate.
        H: The measurement equation.
        F: The state dynamic equation.
        Q: The process noise of the state prediction.

    Returns:
        A tuple containing series for measurements j = 0,..,N-1:
            1. The smoothed mean estimate
            2. The state apriori mean estimate
            3. The state apriori covariance estimate
            4. The state covariance smoothed estimate
    """

    def _step(
        c: _FLCarry, inputs: tuple[JaxFloatArray, JaxFloatArray]
    ) -> tuple[_FLCarry, tuple[JaxFloatArray, JaxFloatArray]]:
        z_k, R_k = inputs

        S = H @ c.Sig[0] @ H.mT + R_k
        cf = jscipy.linalg.cho_factor(0.5 * (S + S.mT))
        innov = z_k - H @ c.x[0]

        prev = c.Sig[:-1]  # P_k^{0,i-1}
        L = jax.vmap(lambda G: jscipy.linalg.cho_solve(cf, H @ G.mT).mT)(prev)  # L_{k,i}

        M = F - (F @ L[0]) @ H  # F - L_{k,0} H

        x_t = c.x[:-1] + jnp.einsum("inm,m->in", L, innov)
        P_t = c.P[:-1] - (prev @ H.mT) @ jnp.swapaxes(L, -1, -2)
        Sig_t = prev @ M.mT

        x0 = F @ x_t[0]  # x̂(k+1|k)
        P0 = F @ P_t[0] @ F.mT + Q

        nxt = _FLCarry(
            jnp.concatenate([x0[None], x_t]),
            jnp.concatenate([P0[None], P_t]),
            jnp.concatenate([P0[None], Sig_t]),
        )
        return nxt, (nxt.x[lag + 1], nxt.P[lag + 1])

    n = x_init.shape[0]
    init = _FLCarry(
        jnp.broadcast_to(x_init, (lag + 2, n)),
        jnp.broadcast_to(P_init, (lag + 2, n, n)),
        jnp.broadcast_to(P_init, (lag + 2, n, n)),
    )
    _, out = jax.lax.scan(_step, init, (z, R))
    return out


def main():
    meas_model = GaussianSelectionTransform(slice(0, 1), 2)
    H = meas_model.matrix
    x0 = jnp.array([0.0, -1.0])
    simulation = Simulation(35, 0.1, 10.0)
    key = jax.random.key(45)
    n_sim = 100

    estimatation_error = []

    fs_break = 5
    for iter_key in jax.random.split(key, n_sim):
        time, truth, x = simulation(x0, iter_key)

        # Prior for the initial x.

        # Sample from the initial prior
        init_key = jax.random.split(iter_key)[0]
        intensity = simulation.sigma**2
        P_init = jnp.eye(x0.shape[0]) * intensity
        x_init = jax.random.multivariate_normal(init_key, x0, P_init)

        # Setup the measurements
        measurements = jnp.einsum("ij,...j->...i", H, x)
        R = jnp.repeat(jnp.array([[simulation.sigma**2]]), len(measurements), axis=0)

        cov_improvement: list[JaxFloatArray] = []

        z_to_filter, z_to_smooth = measurements[:fs_break], measurements[fs_break:]
        R_to_filter, R_to_smooth = R[:fs_break], R[fs_break:]

        # Run KF until breakpoint
        x_filter, P_filter = linear_pred_filter(
            x_init, P_init, intensity, z_to_filter, R_to_filter, simulation.dt
        )
        # Get smoothed estimate
        meas_rv = GaussianRV(z_to_smooth, R_to_smooth)
        x_init_rv = GaussianRV(x_filter.squeeze(axis=0), P_filter.squeeze(axis=0))
        jnp.repeat(simulation.dt, len(meas_rv.mean))
        process_noise_model = WeinerProcessNoise(1, 2, intensity)
        x_s, P_s = smooth_fl(
            meas_rv.mean,
            meas_rv.covariance,
            x_init_rv.mean,
            x_init_rv.covariance,
            meas_model.matrix,
            simulation.generating_transition.matrix(jnp.array(simulation.dt)),
            process_noise_model.covariance(jnp.array(simulation.dt)),
            30,
        )
        error = jnp.linalg.norm(truth[fs_break : fs_break + 1] - x_s, axis=-1) ** 2
        estimatation_error.append(error[jnp.newaxis, ...])
        cov_improvement = jnp.linalg.trace(P_filter - P_s) / jnp.linalg.trace(P_filter)
        tr_P = jnp.linalg.trace(P_s)

    error_arr = jnp.concatenate(estimatation_error, axis=0)
    error_mean = error_arr.mean(axis=0)
    print(error_mean.mean())
    error_std = error_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    steps = np.arange(len(error_mean))
    upper = error_mean + 2 * error_std
    lower = jnp.clip(error_mean - 2 * error_std, min=0)
    ax.plot(
        steps,
        error_mean / error_mean[0],
        lw=1.5,
        color="red",
        label=r"$||\hat{\varepsilon_s}||_2^2$",
    )
    ax.plot(steps, upper / error_mean[0], lw=0.75, color="red")
    ax.plot(steps, lower / error_mean[0], lw=0.75, color="red")
    ax.fill_between(steps, lower / error_mean[0], upper / error_mean[0], alpha=0.2, color="red")
    ax.plot(steps, tr_P / tr_P[0], lw=1.5, color="k", label=r"$tr(P)$")
    ax.plot(steps, cov_improvement, lw=1.5, color="k", ls="--", label=r"$tr(P - P_{smoothed})/tr(P)$")

    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("Normalized Estimate Errors (Actual, Estimated)", fontsize=12)
    ax.legend(fontsize=12)

    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
