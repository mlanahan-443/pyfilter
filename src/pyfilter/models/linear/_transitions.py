# _transitions.py
from dataclasses import dataclass
from functools import cached_property

import jax
import jax.scipy.special
from jax import numpy as jnp

from pyfilter.hints.jax_hints import JaxBoolArray, JaxFloatArray
from pyfilter.types import RandomVariable

from ._base import LinearTransitionBase


@dataclass
class IntegratorChainTransition[State: RandomVariable](LinearTransitionBase[State]):
    r"""Integrator chain transition for p integrators in n spatial dimensions.

    Models the continuous-time system $x \in \mathbb{R}^n$ driven by white
    noise $w(t)$ on the $p$-th derivative:

    .. math::
        \frac{d^p x}{dt^p} = w(t)

    The state vector is laid out in derivative-major order:

    .. math::
        \mathbf{x} = [x_1, \ldots, x_n,
                      \dot{x}_1, \ldots, \dot{x}_n,
                      \ldots,
                      x^{(p-1)}_1, \ldots, x^{(p-1)}_n]^\top

    The discrete-time transition matrix is computed in closed form: since
    the continuous generator $A$ is nilpotent of index $p$, the matrix
    exponential $e^{A \Delta t}$ terminates as a finite polynomial.
    Equivalently, $\Phi(\Delta t) = T(\Delta t) \otimes I_n$, where $T$ is
    $p \times p$ upper-triangular with $T_{ij} = \Delta t^{j-i} / (j-i)!$
    for $j \geq i$.

    Args:
        n: Spatial dimension (e.g., 3 for 3-D position).
        p: Number of kinematic levels tracked (1 = position only,
           2 = position + velocity, 3 = position + velocity + acceleration).

    Examples:
        2-D nearly-constant-velocity (position + velocity, p=2):

        >>> cv2d = IntegratorChainTransition(n=2, p=2)
        >>> cv2d.state_dim
        4

        3-D nearly-constant-acceleration (position + velocity + acceleration, p=3):

        >>> ca3d = IntegratorChainTransition(n=3, p=3)
        >>> ca3d.state_dim
        9
    """

    n: int
    p: int

    def __post_init__(self) -> None:
        if self.n is None:
            raise ValueError("n must not be None")
        if self.p is None:
            raise ValueError("p must not be None")

        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.p < 1:
            raise ValueError(f"p must be >= 1, got {self.p}")

    @property
    def state_dim(self) -> int:
        """Total dimension of the state vector ($n \\cdot p$)."""
        return self.n * self.p

    @cached_property
    def _temporal_factors(self) -> tuple[JaxBoolArray, JaxFloatArray, JaxFloatArray]:
        """Precompute index structure of the temporal matrix T.

        Returns:
            valid: ``(p, p)`` upper-triangular mask.
            exponent: ``(p, p)`` array of $j - i$ values (clipped to 0
                outside the upper triangle so ``dt ** exponent`` is safe).
            inv_factorial: ``(p, p)`` array of $1 / (j - i)!$ values
                (also masked-safe).
        """
        i_idx, j_idx = jnp.indices([self.p, self.p])
        lag = j_idx - i_idx  # (p, p), in [-(p-1), p-1]
        valid = lag >= 0
        lag_safe = jnp.where(valid, lag, 0)

        factorials = jax.scipy.special.factorial(jnp.arange(self.p))
        inv_factorial = 1.0 / factorials[lag_safe]
        return valid, lag_safe, inv_factorial

    @cached_property
    def _eye_n(self) -> JaxFloatArray:
        """Cached ``np.eye(n)`` for the Kronecker product."""
        return jnp.eye(self.n)

    @property
    def A(self) -> JaxFloatArray:
        """Just identity matrix."""
        return jnp.eye(self.state_dim, k=self.n)

    def matrix(self, dt: JaxFloatArray) -> JaxFloatArray:
        """Discrete-time transition matrix $\\Phi(\\Delta t)$.

        Args:
            dt: Timestep(s). Scalar or array of arbitrary leading batch shape.

        Returns:
            Array of shape ``(*dt.shape, state_dim, state_dim)``.
        """
        dt_arr = jnp.asarray(dt)
        valid, exponent, inv_factorial = self._temporal_factors

        # Temporal matrix: T[..., i, j] = dt^(j-i) / (j-i)! for j >= i.
        # Broadcast dt over the (p, p) grid.
        dt_b = dt_arr[..., jnp.newaxis, jnp.newaxis]
        T = jnp.where(valid, dt_b**exponent * inv_factorial, 0.0)  # (*batch, p, p)

        # Kronecker with I_n via einsum: Phi[..., i*n+a, j*n+b] = T[..., i, j] * I[a, b].
        Phi = jnp.einsum("...ij,ab->...iajb", T, self._eye_n)

        return Phi.reshape(*dt_arr.shape, self.state_dim, self.state_dim)

    def inverse(self, dt: JaxFloatArray) -> JaxFloatArray:
        """Inverse transition: $\\Phi(\\Delta t)^{-1} = \\Phi(-\\Delta t)$.

        For an integrator chain, the inverse has a clean closed form
        and does not require a matrix inversion.
        """
        dt_arr = jnp.asarray(dt)
        return self.matrix(-dt_arr)

    def transform(self, x: State, dt: JaxFloatArray) -> State:
        """Push a state forward by ``dt`` under the discrete-time dynamics.

        Args:
            x: State (deterministic vector or random variable).
            dt: Timestep, broadcastable with any leading batch dims of ``x``.

        Returns:
            The propagated state.
        """
        return self.matrix(dt) @ x
