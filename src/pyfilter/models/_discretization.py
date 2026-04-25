import numpy as np
import scipy

from ..hints import FloatArray


def expm_discretizer(A: FloatArray, dt: FloatArray) -> FloatArray:
    """Exact discretization: Phi = expm(A * dt)."""
    return scipy.linalg.expm(A * dt)


def euler_discretizer(A: FloatArray, dt: FloatArray) -> FloatArray:
    """First-order Euler: Phi = I + A * dt."""
    eye = np.eye(A.shape[-1])
    return eye + A * dt
