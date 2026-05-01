# _transitions.py
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.special

from pyfilter.config import FDTYPE_ as FDTYPE
from pyfilter.hints import BoolArray, FloatArray
from pyfilter.types import RandomVariable

from ._base import LinearTransitionBase


@dataclass(frozen=True)
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
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.p < 1:
            raise ValueError(f"p must be >= 1, got {self.p}")

    @property
    def state_dim(self) -> int:
        """Total dimension of the state vector ($n \\cdot p$)."""
        return self.n * self.p

    @cached_property
    def _temporal_factors(self) -> tuple[BoolArray, FloatArray, FloatArray]:
        """Precompute index structure of the temporal matrix T.

        Returns:
            valid: ``(p, p)`` upper-triangular mask.
            exponent: ``(p, p)`` array of $j - i$ values (clipped to 0
                outside the upper triangle so ``dt ** exponent`` is safe).
            inv_factorial: ``(p, p)`` array of $1 / (j - i)!$ values
                (also masked-safe).
        """
        i_idx, j_idx = np.indices((self.p, self.p))
        lag = j_idx - i_idx  # (p, p), in [-(p-1), p-1]
        valid = lag >= 0
        lag_safe = np.where(valid, lag, 0)

        factorials = scipy.special.factorial(np.arange(self.p)).astype(FDTYPE)
        inv_factorial = 1.0 / factorials[lag_safe]
        return valid, lag_safe.astype(FDTYPE), inv_factorial

    @cached_property
    def _eye_n(self) -> FloatArray:
        """Cached ``np.eye(n)`` for the Kronecker product."""
        return np.eye(self.n)

    @property
    def A(self) -> FloatArray:
        """Continuous-time generator matrix.

        Block-bidiagonal with $I_n$ on the first block super-diagonal:
        $A_{ij} = I_n$ if $j = i + 1$, else $0$ (in block form).
        """
        d = self.state_dim
        A = np.zeros((d, d))
        if self.p > 1:
            # Place I_n on each block super-diagonal position (i, i+1).
            block_diag = np.einsum("i,ab->iab", np.ones(self.p - 1), self._eye_n)
            for i in range(self.p - 1):
                A[
                    i * self.n : (i + 1) * self.n, (i + 1) * self.n : (i + 2) * self.n
                ] = block_diag[i]
        return A

    def matrix(self, dt: FloatArray) -> FloatArray:
        """Discrete-time transition matrix $\\Phi(\\Delta t)$.

        Args:
            dt: Timestep(s). Scalar or array of arbitrary leading batch shape.

        Returns:
            Array of shape ``(*dt.shape, state_dim, state_dim)``.
        """
        dt_arr = np.asarray(dt, dtype=FDTYPE)
        valid, exponent, inv_factorial = self._temporal_factors

        # Temporal matrix: T[..., i, j] = dt^(j-i) / (j-i)! for j >= i.
        # Broadcast dt over the (p, p) grid.
        dt_b = dt_arr[..., np.newaxis, np.newaxis]
        T = np.where(valid, dt_b**exponent * inv_factorial, 0.0)  # (*batch, p, p)

        # Kronecker with I_n via einsum: Phi[..., i*n+a, j*n+b] = T[..., i, j] * I[a, b].
        Phi = np.einsum("...ij,ab->...iajb", T, self._eye_n)

        return Phi.reshape(*dt_arr.shape, self.state_dim, self.state_dim)

    def inverse(self, dt: FloatArray) -> FloatArray:
        """Inverse transition: $\\Phi(\\Delta t)^{-1} = \\Phi(-\\Delta t)$.

        For an integrator chain, the inverse has a clean closed form
        and does not require a matrix inversion.
        """
        return self.matrix(-np.asarray(dt, dtype=FDTYPE))

    def transform(self, x: State, dt: FloatArray) -> State:
        """Push a state forward by ``dt`` under the discrete-time dynamics.

        Args:
            x: State (deterministic vector or random variable).
            dt: Timestep, broadcastable with any leading batch dims of ``x``.

        Returns:
            The propagated state.
        """
        return self.matrix(dt) @ x
