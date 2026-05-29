"""This compares the runtimes of a simple relatively naive implementation of a linear gaussian kalman filter
using:

1. Numpy, with python recursive loop.
2. Numpy, with numba compiled recursion.
3. Jax, with jax.lax.scan.

The results are for CPU: 8 core AMD Ryzen 7 8840HS w/ Radeon 780M Graphics machine. We force single threading due to problem size.

The results demonstrate that either numba or Jax acceleration yield a significant speedup
- n = 100 measurements: ~5.1x
- n = 1000 measurement: ~ 5.5x
- n = 10000 measurement: ~5.9x

The numba kernel produces less jitter than the jax kernel.
"""

import jax
import numba
import numpy as np
import rich
from jax import numpy as jnp
from numpy.random import default_rng
from numpy.typing import NDArray

from pyfilter.gutil import LineProfiler
from pyfilter.models.linear import GaussianSelectionTransform, IntegratorChainTransition
from pyfilter.types.process_noise import WeinerProcessNoise

jax.config.update("jax_enable_x64", True)

import time


def numpy_kalman_step(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    eye: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Predict
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    # Update (Joseph form)
    HP = H @ P_pred
    S = HP @ H.T + R
    gain = np.linalg.solve(S, HP).T
    residual = z - H @ x_pred
    x_new = x_pred + gain @ residual
    A = eye - gain @ H
    P_new = A @ P_pred @ A.T + gain @ R @ gain.T
    return x_new, P_new


def numpy_filter[Arr: NDArray[np.floating]](
    x0: Arr,
    P0: Arr,
    z: Arr,
    F: Arr,
    H: Arr,
    R: Arr,
    Q: Arr,
) -> tuple[Arr, Arr]:

    eye = np.eye(x0.shape[-1])
    x_out = np.empty((z.shape[0] + 1, x0.shape[-1]), dtype=x0.dtype)
    P_out = np.empty((z.shape[0] + 1, *P0.shape[-2:]), dtype=P0.dtype)
    x_out[0] = x0
    P_out[0] = P0
    for i in range(z.shape[0]):
        x_out[i + 1, :], P_out[i + 1, :] = numpy_kalman_step(
            x_out[i, :], P_out[i, :], z[i], F, H, R, Q, eye
        )

    return x_out, P_out


@numba.njit(cache=True, fastmath=False, boundscheck=False)
def numbda_kalman_step(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    eye: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Predict
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    # Update (Joseph form)
    HP = H @ P_pred
    S = HP @ H.T + R
    gain = np.linalg.solve(S, HP).T
    residual = z - H @ x_pred
    x_new = x_pred + gain @ residual
    A = eye - gain @ H
    P_new = A @ P_pred @ A.T + gain @ R @ gain.T
    return x_new, P_new


@numba.njit(cache=True, fastmath=False, boundscheck=False)
def numbda_filter(
    x0: np.ndarray,
    P0: np.ndarray,
    z: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eye = np.eye(x0.shape[-1])
    x_out = np.empty((z.shape[0] + 1, x0.shape[-1]), dtype=x0.dtype)
    P_out = np.empty((z.shape[0] + 1, *P0.shape[-2:]), dtype=P0.dtype)
    x_out[0] = x0
    P_out[0] = P0
    for i in range(z.shape[0]):
        x_out[i + 1, :], P_out[i + 1, :] = numbda_kalman_step(
            x_out[i, :], P_out[i, :], z[i], F, H, R, Q, eye
        )

    return x_out, P_out


@jax.jit
def jax_filter(
    x0: jax.Array,
    P0: jax.Array,
    z: jax.Array,
    F: jax.Array,
    H: jax.Array,
    R: jax.Array,
    Q: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    eye = jnp.eye(x0.shape[-1], dtype=x0.dtype)

    def step(state, z_k):
        x, P = state
        x_pred = F @ x
        P_pred = F @ P @ F.mT + Q
        HP = H @ P_pred
        S = HP @ H.mT + R
        gain = jnp.linalg.solve(S, HP).mT
        residual = z_k - H @ x_pred
        x_new = x_pred + gain @ residual
        A = eye - gain @ H
        P_new = A @ P_pred @ A.mT + gain @ R @ gain.mT
        return (x_new, P_new), (x_new, P_new)

    _, (xs, Ps) = jax.lax.scan(step, (x0, P0), z)
    return jnp.concatenate([x0[jnp.newaxis, ...], xs]), jnp.concatenate([P0[jnp.newaxis, ...], Ps])


def generate_data(
    x0: NDArray[np.floating], n: int, F: NDArray[np.floating]
) -> NDArray[np.floating]:
    x = np.zeros((n + 1, 9), dtype=x0.dtype)
    x[0] = x0.copy()
    for i in range(1, n + 1):
        x[i] = F @ x[i - 1]

    return x


def main():

    # Common Setup
    x0 = np.array([1, -10, 1, -0.15, 0.03, 1, 0.001, 0.01, -0.2], dtype=np.float64)
    F = np.array(IntegratorChainTransition(n=3, p=3).matrix(jnp.array(0.5)))
    H = np.array(GaussianSelectionTransform(slice(0, 3), 9).matrix)
    Q = np.array(WeinerProcessNoise(n=3, p=3, intensity=jnp.array(0.1)).covariance(jnp.array(0.5)))

    n = 100
    rng = default_rng(seed=45)
    noise = rng.normal(scale=0.1, size=(n + 1, 3)).astype(x0.dtype)
    z = generate_data(x0, n, F)[:, :3] + noise
    R = np.eye(3) * 0.1
    P0 = np.eye(9) * 10

    # NumPy
    numpy_profiler = LineProfiler("NumPy")
    numpy_profiler.timeit(lambda: numpy_filter(x0, P0, z, F, H, R, Q), number=100, repeat=5)

    rich.print(numpy_profiler)

    # Jax
    # Convert to jax arrays.
    args = tuple(jnp.asarray(a) for a in [x0, P0, z, F, H, R, Q])

    # Check compile time.
    start = time.time()
    jax_filter(*args)
    end = time.time()
    print(f"Jax Compile Time: {round((end - start) * 1e3)} ms")

    # Profile
    jax_profiler = LineProfiler("Jax")
    jax_profiler.timeit(lambda: jax.block_until_ready(jax_filter(*args)), number=100, repeat=5)
    rich.print(jax_profiler)

    # Numba
    # Numba is picky about datatypes
    args = tuple(arr.astype(np.float64) for arr in [x0, P0, z, F, H, R, Q])

    # Check compile time
    start = time.time()
    numbda_filter(*args)
    end = time.time()
    print(f"Numba Compile Time: {round((end - start) * 1e3)} ms")

    # Profile.
    numba_profiler = LineProfiler("Numba")
    numba_profiler.timeit(lambda: numbda_filter(*args), number=100, repeat=5)

    rich.print(numba_profiler)

    # Check correctness against NumPy reference.
    x_np, P_np = numpy_filter(x0, P0, z, F, H, R, Q)

    x_jax, P_jax = (
        np.array(x) for x in jax_filter(*tuple(jnp.asarray(a) for a in [x0, P0, z, F, H, R, Q]))
    )

    x_numba, P_numba = numbda_filter(
        *tuple(arr.astype(np.float64) for arr in [x0, P0, z, F, H, R, Q])
    )

    for name, arrs in zip(("Jax", "Numba"), ((x_jax, P_jax), (x_numba, P_numba))):
        np.testing.assert_almost_equal(
            arrs[0], x_np, err_msg=f"Mean estimate differs for {name} filter"
        )
        np.testing.assert_almost_equal(
            arrs[1], P_np, err_msg=f"Covariance estimate differs for {name} filter"
        )


if __name__ == "__main__":
    main()
