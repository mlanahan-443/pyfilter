import jax.scipy
from jax import numpy as jnp

from pyfilter.hints.jax_hints import JaxFloatArray


def expm_discretizer(A: JaxFloatArray, dt: JaxFloatArray) -> JaxFloatArray:
    """Exact discretization: Phi = expm(A * dt)."""
    return jax.scipy.linalg.expm(A * dt)


def euler_discretizer(A: JaxFloatArray, dt: JaxFloatArray) -> JaxFloatArray:
    """First-order Euler: Phi = I + A * dt."""
    eye = jnp.eye(A.shape[-1], dtype=A.dtype)
    return eye + A * dt
