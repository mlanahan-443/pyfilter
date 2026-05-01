from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy

from pyfilter.config import FDTYPE_ as FDTYPE
from pyfilter.types import Covariance

from ..hints import FloatArray


class ProcessNoise(ABC):
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    @abstractmethod
    def covariance(self, dt: FloatArray) -> Covariance:
        pass

    def __call__(self, dt: FloatArray) -> Covariance:
        return self.covariance(dt)


def _full_intensity_matrix(intensity: FloatArray, n: int) -> FloatArray:
    """Normalize ``intensity`` to a full $(..., n, n)$ matrix."""
    Q = np.asarray(intensity, dtype=FDTYPE)
    if Q.ndim == 0:
        return Q * np.eye(n)
    if Q.ndim == 1:
        if Q.shape[0] != n:
            raise ValueError(f"1-D intensity must have length {n}, got {Q.shape}")
        return np.diag(Q)
    if Q.shape[-2:] != (n, n):
        raise ValueError(
            f"Intensity matrix must have trailing shape ({n}, {n}), got {Q.shape}"
        )
    return Q


@dataclass
class WeinerProcessNoise(ProcessNoise):
    r"""
    Generate a discrete process noise model for the continuous time
    system $x \in \mathbb{R}^n$ driven by white noise $w(t)$ on the
    $p$-th derivative:

    .. math::
        \frac{d^p x}{dt^p} = w(t) \qquad \mathbb{E}[w(t)w^T(\tau)] = \tilde{Q}\delta(t - \tau)

    Where:
    .. math::
        Q_d = \int_{0}^{\Delta t} e^{A \tau} \tilde{Q} e^{A^T \tau} d\tau

    Args:
        n: Spatial dimension.
        p: Number of kinematic levels.
        intensity: Continuous-time noise intensity $\tilde{Q}$ on the
            highest derivative. Either a scalar (interpreted as $\sigma^2 I_n$),
            a length-$n$ vector (diagonal), or an $(n, n)$ matrix.
            Optional leading batch dimensions are supported and broadcast
            with ``dt``.

    Examples:
        # 2-D nearly-constant-velocity (position + velocity, p=2), with isotropic covariance:
        >>> wpn_cv_2d = WeinerProcessNoise(2,2, np.array(0.1))
        >>> wpn_cv_2d(np.array(0.1))
        ... array([[3.33333333e-05, 0.00000000e+00, 5.00000000e-04, 0.00000000e+00],
        ...        [0.00000000e+00, 3.33333333e-05, 0.00000000e+00, 5.00000000e-04],
        ...        [5.00000000e-04, 0.00000000e+00, 1.00000000e-02, 0.00000000e+00],
        ...        [0.00000000e+00, 5.00000000e-04, 0.00000000e+00, 1.00000000e-02]])


        # 2-D nearly-constant-velocity (position + velocity, p=2), with anisotropic covariance:
        >>> wpn_cv_2d = WeinerProcessNoise(2,2, np.array([0.1,0.2]))
        >>> wpn_cv_2d(np.array([0.5,1.]))
        ... array([[[0.00416667, 0.        , 0.0125    , 0.        ],
        ...         [0.        , 0.00833333, 0.        , 0.025     ],
        ...         [0.0125    , 0.        , 0.05      , 0.        ],
        ...         [0.        , 0.025     , 0.        , 0.1       ]],
        ...
        ...       [[0.03333333, 0.        , 0.05      , 0.        ],
        ...        [0.        , 0.06666667, 0.        , 0.1       ],
        ...        [0.05      , 0.        , 0.1       , 0.        ],
        ...        [0.        , 0.1       , 0.        , 0.2       ]]])

        # 3-D nearly-constant-acceleration (position + velocity + acceleration, p=3) with anisotropic covariance:
        >>> wpn_ca_3d = WeinerProcessNoise(3,3, np.array([0.01,0.02,0.014]))
        >>> wpn_ca_3d(np.array([0.01,0.03])).shape
        ... (2,9,9)

    """

    n: int
    p: int
    intensity: FloatArray

    def __post_init__(self) -> None:
        if self.n < 1 or self.p < 1:
            raise ValueError("n and p must be >= 1")

        super().__init__((*self.intensity.shape[:-2], self.state_dim, self.state_dim))

    @property
    def state_dim(self) -> int:
        return self.n * self.p

    @cached_property
    def _temporal_factors(self) -> tuple[FloatArray, FloatArray]:
        """Precompute the (p, p) exponent and 1/coefficient grids.

        Returns ``(exponents, coeffs)`` such that
        ``tau[i, j] = dt ** exponents[i, j] * coeffs[i, j]``.
        """
        # a_i = p - 1 - i
        a = (self.p - 1) - np.arange(self.p)  # (p,)
        a_i = a[:, None]  # (p, 1)
        a_j = a[None, :]  # (1, p)

        exponents = (a_i + a_j + 1).astype(FDTYPE)  # (p, p)

        factorials = scipy.special.factorial(np.arange(self.p)).astype(FDTYPE)
        denom = factorials[a_i] * factorials[a_j] * (a_i + a_j + 1)
        coeffs = 1.0 / denom  # (p, p)
        return exponents, coeffs

    @cached_property
    def _intensity_matrix(self) -> FloatArray:
        return _full_intensity_matrix(self.intensity, self.n)

    def covariance(self, dt: FloatArray) -> FloatArray:
        """Discrete process noise covariance $Q_d(\\Delta t)$.

        Args:
            dt: Timestep(s); scalar or arbitrary leading batch shape.

        Returns:
            Array of shape ``(*broadcast_shape, state_dim, state_dim)`` where
            ``broadcast_shape`` is the broadcast of ``dt.shape`` and the
            leading batch dims of ``intensity``.
        """
        exponents, coeffs = self._temporal_factors
        Q_tilde = self._intensity_matrix  # (..., n, n)

        # tau[..., i, j] = dt^exponents[i, j] * coeffs[i, j]
        dt_b = dt[..., np.newaxis, np.newaxis]
        tau = dt_b**exponents * coeffs  # (*dt_batch, p, p)

        # Q_d = tau ⊗ Q_tilde, batched over the leading dims of both.
        # Result block (i, j) is tau[..., i, j] * Q_tilde[..., :, :].
        Qd: FloatArray = np.einsum("...ij,...ab->...iajb", tau, Q_tilde)

        # Determine final batch shape via broadcasting of dt and intensity batches.
        intensity_batch = Q_tilde.shape[:-2]
        out_batch = np.broadcast_shapes(dt.shape, intensity_batch)
        return Qd.reshape((*out_batch, self.state_dim, self.state_dim))


@dataclass
class VanLoanProcessNoise(ProcessNoise):
    r"""Obtain discrete process noise numerically from continuous transition and process noise covariances.

    This is useful if the process noise is difficult to determine exactly. This method exploits the fact that:


    .. math::
        M = \begin{bmatrix}
            -A & Q_c \\
            0 & A^T
        \end{bmatrix} \Delta t

    Then:
    .. math::
        e^{M} = \begin{bmatrix}
            e^{-A \Delta t} & \int_{0}^1 e^{A (s - 1)} Q_{c} e^{As} ds \\
            0 & e^{A \Delta t}
        \end{bmatrix}

    Where the upper right hand block is the F^{-1} Q_d, which can be solved explicitly.

    Args:
        A: continuous time system matrix.
        Qc: continuous time process noise.

    # 2-D nearly-constant-velocity (position + velocity, p=2), with isotropic covariance:
    >>> A = np.array([[0,0,1,0],
                      [0,0,0,1],
                      [0,0,0,0],
                      [0,0,0,0]])
    >>> Q_c = np.zeros_like(A)
    >>> Q_c[2:,2:] = np.eye(2)*0.1
    >>> vlpn_cv_2d = VanLoanProcessNoise(A,Q_c)
    >>> wpn_cv_2d(np.array(0.1))
    ... array([[3.33333333e-05, 0.00000000e+00, 5.00000000e-04, 0.00000000e+00],
    ...        [0.00000000e+00, 3.33333333e-05, 0.00000000e+00, 5.00000000e-04],
    ...        [5.00000000e-04, 0.00000000e+00, 1.00000000e-02, 0.00000000e+00],
    ...        [0.00000000e+00, 5.00000000e-04, 0.00000000e+00, 1.00000000e-02]])
    """

    A: FloatArray
    Qc: FloatArray

    def __post_init__(self):
        shape = np.broadcast_shapes(self.A.shape, self.Qc.shape)
        super().__init__(shape)

    @property
    def n(self):
        return self.A.shape[-1]

    @cached_property
    def coeff(self) -> FloatArray:
        r"""The coefficient $C$ for the matrix $M  = C \Delta t$
        \begin{align*}
        M = \begin{bmatrix}
            -A & Q_c \\
            0 & A^T
        \end{bmatrix} \Delta t \\
        M = C \Delta t
        \end{align*}
        """
        Q = self.Qc
        A = self.A
        n = self.n
        coeff = np.zeros((*A.shape[:-2], 2 * n, 2 * n), dtype=np.result_type(Q, A))
        coeff[..., :n, :n] = -A
        coeff[..., :n, n:] = Q
        coeff[..., n:, n:] = A.mT
        return coeff

    def covariance(self, dt: FloatArray) -> FloatArray:
        """Compute covariance using van-loans discritization."""
        # compute matrix exponential.
        exp_M = scipy.linalg.expm(self.coeff * dt[..., np.newaxis, np.newaxis])
        n = self.n

        # e^{A^T dt} = F
        Phi = exp_M[..., n:, n:].mT

        # Van loans formulae
        Q_d = Phi @ exp_M[..., :n, n:]

        # Symmetrize, the matrix exponential has some low level (~1e-15) noise.
        return 0.5 * (Q_d + Q_d.mT)
