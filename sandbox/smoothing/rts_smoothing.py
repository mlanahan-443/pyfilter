"""Development of fixed lag smoothing."""

from typing import NamedTuple

import jax
import matplotlib.pyplot as plt
import numpy as np
from common import Simulation, linear_filter_scan
from jax import numpy as jnp

from pyfilter.hints.jax_hints import JaxFloatArray
from pyfilter.models.linear import (
    GaussianSelectionTransform,
)
from pyfilter.types import solve_symmetric


class _RtsCarry(NamedTuple):
    x_s_next: JaxFloatArray
    P_s_next: JaxFloatArray


def diagnostic(
    x_pred: JaxFloatArray,
    P_pred: JaxFloatArray,
    x_update: JaxFloatArray,
    P_update: JaxFloatArray,
    F: JaxFloatArray,
):

    N = len(x_pred)

    k = N // 2
    W = jnp.linalg.solve(P_pred[k + 1], F @ P_update[k]).T
    print(jnp.abs(jnp.linalg.eigvals(W)))  # expect < 1

    print(jnp.linalg.eigvalsh(P_pred[k] - P_update[k]))  # expect >= 0

    Q_implied = P_pred[k + 1] - F @ P_update[k] @ F.T
    print(Q_implied)  # expect your Q_d

    print(jnp.abs(jnp.linalg.eigvals(F)))


@jax.jit
def smooth_rts(
    x_pred: JaxFloatArray,
    P_pred: JaxFloatArray,
    x_update: JaxFloatArray,
    P_update: JaxFloatArray,
    F: JaxFloatArray,
) -> tuple[JaxFloatArray, JaxFloatArray]:
    def _step(
        carry: _RtsCarry,
        inputs: tuple[JaxFloatArray, JaxFloatArray, JaxFloatArray, JaxFloatArray],
    ) -> tuple[_RtsCarry, tuple[JaxFloatArray, JaxFloatArray]]:
        x_p_next, P_p_next, x_u, P_u = inputs

        W_T = solve_symmetric(P_p_next, F @ P_u)
        W = W_T.mT
        P_s = P_u - W @ (P_p_next - carry.P_s_next) @ W_T
        P_s = 0.5 * (P_s + P_s.mT)
        x_s = x_u + W @ (carry.x_s_next - x_p_next)

        return _RtsCarry(x_s_next=x_s, P_s_next=P_s), (x_s, P_s)

    init = _RtsCarry(x_s_next=x_update[-1], P_s_next=P_update[-1])
    _, (x_s, P_s) = jax.lax.scan(
        _step,
        init,
        xs=(x_pred[1:], P_pred[1:], x_update[:-1], P_update[:-1]),
        reverse=True,
    )

    return (
        jnp.concatenate([x_s, x_update[-1:]], axis=0),
        jnp.concatenate([P_s, P_update[-1:]], axis=0),
    )


def main():

    meas_model = GaussianSelectionTransform(slice(0, 1), 2)
    H = meas_model.matrix
    x0 = jnp.array([0.0, -1.0])
    n_init = 20
    simulation = Simulation(50, 0.1, 1.0)
    key = jax.random.key(45)
    n_sim = 5

    estimatation_error_s = []
    estimatation_error_f = []
    trace_cov_s = []
    trace_cov_f = []

    for iter_key in jax.random.split(key, n_sim):
        time, truth, x = simulation(x0, iter_key)

        # Sample from the initial prior
        init_key = jax.random.split(iter_key)[0]
        intensity = simulation.sigma**2
        P_init = jnp.eye(x0.shape[0]) * intensity
        x_init = jax.random.multivariate_normal(init_key, x0, P_init)

        # Setup the measurements
        measurements = jnp.einsum("ij,...j->...i", H, x)
        R = jnp.repeat(jnp.array([[simulation.sigma**2]]), len(measurements), axis=0)

        F = simulation.generating_transition.matrix(jnp.array(simulation.dt))

        # Run KF until breakpoint
        (x_filter, P_filter), (x_pred, P_pred) = linear_filter_scan(
            x_init, P_init, intensity, measurements, R, simulation.dt
        )
        x_filter, P_filter = x_filter[n_init:], P_filter[n_init:]
        x_pred, P_pred = x_pred[n_init:], P_pred[n_init:]
        truth = truth[n_init:]
        x_s, P_s = smooth_rts(x_pred, P_pred, x_filter, P_filter, F)

        error_s = jnp.linalg.norm(truth - x_s, axis=-1) ** 2
        error_f = jnp.linalg.norm(truth - x_filter, axis=-1) ** 2

        estimatation_error_s.append(error_s[jnp.newaxis, ...])
        estimatation_error_f.append(error_f[jnp.newaxis, ...])
        tr_P_f = jnp.linalg.trace(P_filter)
        tr_P_s = jnp.linalg.trace(P_s)
        trace_cov_s.append(tr_P_s[jnp.newaxis, ...])
        trace_cov_f.append(tr_P_f[jnp.newaxis, ...])

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r"$||\hat{\varepsilon_s}||_2^2$", r"$||\hat{\varepsilon_f}||_2^2$"]
    for estimation_error, trace_cov, color, ls, label in zip(
        [estimatation_error_s, estimatation_error_f],
        [trace_cov_s, trace_cov_f],
        ["red", "black"],
        ["-", "--"],
        labels, strict=False,
    ):
        error_arr = jnp.concatenate(estimation_error, axis=0)
        tr_arr = jnp.concatenate(trace_cov, axis=0)
        error_mean = error_arr.mean(axis=0)
        tr_mean = tr_arr.mean(axis=0)

        steps = np.arange(len(error_mean))
        ax.plot(steps, error_mean, lw=1.5, ls=ls, color=color, label=label)

        ax.plot(steps, tr_mean, lw=1.5, ls="-.", color=color, label=r"$tr(P)$")

    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("Estimate Errors", fontsize=12)
    ax.legend(fontsize=12)

    # fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
