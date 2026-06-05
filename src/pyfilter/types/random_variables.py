from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

import equinox as eqx
from jax import numpy as jnp

from pyfilter.hints.jax_hints import ArrayIndex, JaxFloatArray
from pyfilter.linear_solve import solve_symmetric_cholesky
from pyfilter.types.covariance import (
    CovarianceBase,
    cholesky_factor,
    linear_cross_covariance,
)

CHOLESK_SYMN_ = {"chofactor", "cho", "cholesky", "square-root"}
ARRAY_SYMN_ = {"array", "np.ndarray", "JaxFloatArray", "Array"}
COV_SYMN_ = CHOLESK_SYMN_.union(ARRAY_SYMN_)

type CovarianceType = CovarianceBase | JaxFloatArray
type Variable = GaussianRV[Any] | JaxFloatArray | CovarianceBase | float


class GaussianRV[Covariance: CovarianceType](eqx.Module):
    mean: JaxFloatArray
    covariance: Covariance

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the mean array."""
        return self.mean.shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Returns the batch shape of the mean array."""
        return self.mean.shape[:-1]

    @property
    def length(self) -> int:
        """The dimension of the random variable (last dimension of mean)."""
        return self.mean.shape[-1]

    def __len__(self) -> int:
        """Return the dimension of the random variable (last dimension of mean)."""
        return self.length

    def _check_compatible(self, other: Variable) -> None:
        """Check if operations with 'other' are valid."""
        if isinstance(other, GaussianRV):
            if self.shape != other.shape:
                raise ValueError(f"Incompatible shapes: {self.shape} and {other.shape}")
        elif isinstance(other, jnp.ndarray):
            # For array operations, check if broadcasting is valid
            try:
                jnp.broadcast_shapes(self.mean.shape, other.shape)
            except ValueError:
                raise ValueError(f"Cannot broadcast shapes {self.mean.shape} and {other.shape}")
        elif isinstance(other, CovarianceBase):
            if other.matrix_shape != self.covariance.shape[:-2]:
                raise ValueError(
                    f"Covariance with shape: {other.shape} not compatible with {self.covariance.shape}"
                )

    def __add__(self, other: Variable) -> GaussianRV[Any]:
        """Add a GaussianRV, constant array, or scalar to this GaussianRV."""
        self._check_compatible(other)

        if isinstance(other, GaussianRV):
            # When adding covariances, ensure Covariance objects are on the left
            # to properly invoke their __add__ methods
            if isinstance(self.covariance, CovarianceBase):
                new_cov = self.covariance + other.covariance  # type: ignore[operator]
            elif isinstance(other.covariance, CovarianceBase):
                new_cov = other.covariance + self.covariance  # type: ignore[operator]
            else:
                new_cov = self.covariance + other.covariance

            return GaussianRV(self.mean + other.mean, new_cov)
        elif isinstance(other, CovarianceBase):
            return GaussianRV(self.mean, self.covariance + other)  # type: ignore[operator]
        else:  # scalar or array constant
            return GaussianRV(self.mean + other, self.covariance.copy())

    def __radd__(self, other: Variable) -> GaussianRV[Any]:
        """Right addition (for scalar/array + GaussianRV)."""
        return self.__add__(other)

    def __sub__(self, other: Variable) -> GaussianRV[Any]:
        """Subtract a GaussianRV, constant array, or scalar from this GaussianRV."""
        self._check_compatible(other)

        if isinstance(other, GaussianRV):
            # Ensure Covariance objects are on the left for proper method dispatch
            if isinstance(self.covariance, CovarianceBase):
                new_cov = self.covariance + other.covariance  # type: ignore[operator]
            elif isinstance(other.covariance, CovarianceBase):
                new_cov = other.covariance + self.covariance  # type: ignore[operator]
            else:
                new_cov = self.covariance + other.covariance

            return GaussianRV(self.mean - other.mean, new_cov)
        else:  # scalar or array constant
            return GaussianRV(self.mean - other, self.covariance.copy())  # type: ignore[operator]

    def __rsub__(self, other: Variable) -> GaussianRV[Any]:
        """Right subtraction (for scalar/array - GaussianRV)."""
        self._check_compatible(other)
        if isinstance(other, GaussianRV):
            # Ensure Covariance objects are on the left for proper method dispatch
            if isinstance(other.covariance, CovarianceBase):
                new_cov = other.covariance + self.covariance  # type: ignore[operator]
            elif isinstance(self.covariance, CovarianceBase):
                new_cov = self.covariance + other.covariance  # type: ignore[operator]
            else:
                new_cov = other.covariance + self.covariance

            return GaussianRV(other.mean - self.mean, new_cov)

        else:
            return GaussianRV(other - self.mean, self.covariance.copy())  # type: ignore[operator]

    def __mul__(self, other: JaxFloatArray | float) -> GaussianRV[Any]:
        """Multiply GaussianRV by a deterministic matrix or scalar.

        For scalar a: Y = aX -> mean_Y = a*mean_X, cov_Y = a²*cov_X
        For matrix A: Y = AX -> mean_Y = A*mean_X, cov_Y = A*cov_X*A^T
        """
        other = jnp.asarray(other, dtype=self.mean.dtype)

        if other.ndim == 0:  # scalar
            return GaussianRV(self.mean * other, self.covariance * (other**2))  # type: ignore[operator]
        elif other.ndim == 1:  # element-wise multiplication
            # Treat as diagonal matrix multiplication
            self._check_compatible(other)
            # Broadcasting for batch dimensions
            if isinstance(self.covariance, jnp.ndarray):
                cov_scale = other[..., :, jnp.newaxis] * other[..., jnp.newaxis, :]
                return GaussianRV(self.mean * other, self.covariance * cov_scale)
            else:
                raise TypeError(
                    "If gaussian random variable has a Covariance class covariance, then only elementwise multiplication with scalrs is allowed."
                )

        else:  # matrix multiplication
            # other has shape (..., m, n) where n matches last dim of mean
            if other.shape[-1] != len(self):
                raise ValueError(f"Matrix dimension mismatch: {other.shape} @ {self.mean.shape}")

            # Batch matrix multiply for mean: (..., m, n) @ (..., n) -> (..., m)
            new_mean = jnp.einsum("...ij,...j->...i", other, self.mean)

            # Batch computation of A @ Cov @ A.T
            if isinstance(self.covariance, jnp.ndarray):
                # Step 1: A @ Cov -> (..., m, n) @ (..., n, n) -> (..., m, n)
                temp = jnp.einsum("...ij,...jk->...ik", other, self.covariance)
                # Step 2: (A @ Cov) @ A.T -> (..., m, n) @ (..., n, m) -> (..., m, m)
                new_cov = jnp.einsum("...ij,...kj->...ik", temp, other)
            else:
                new_cov = self.covariance.quadratic_form(other)

            return GaussianRV(new_mean, new_cov)

    def __rmul__(self, other: JaxFloatArray | float) -> GaussianRV[Any]:
        """Right multiplication (for scalar/array * GaussianRV)."""
        return self.__mul__(other)

    def __matmul__(self, other: JaxFloatArray) -> GaussianRV[Any]:
        """Matrix multiplication using @ operator (same as __mul__ for matrices)."""
        if not isinstance(other, jnp.ndarray) or other.ndim < 2:
            raise ValueError("@ operator requires a matrix (array with ndim >= 2)")
        return self.__mul__(other)

    def __rmatmul__(self, other: JaxFloatArray) -> GaussianRV[Any]:
        """Right matrix multiplication (for A @ self)."""
        # 'other' is the matrix A on the left
        # We can just call our existing __matmul__ method,
        # which correctly validates 'other' and calls __mul__.
        return self.__matmul__(other)

    def __repr__(self) -> str:
        trace = (
            jnp.trace(self.covariance, axis1=-2, axis2=-1).sum()
            if isinstance(self.covariance, jnp.ndarray)
            else self.covariance.trace()
        )
        return f"GaussianRV(shape={self.shape}, mean_norm={jnp.linalg.norm(self.mean):.3f}, cov_trace={trace})"

    def __getitem__(self, indices: ArrayIndex) -> GaussianRV[Any]:
        """General indexing

        Args:
            indices (ArrayIndex): An index.

        Returns:
            GaussianRV: The requested index of the guassian random variable.
        """

        return GaussianRV(self.mean[indices], self.covariance[indices])

    def marginal(self, indices: ArrayIndex) -> GaussianRV[Any]:
        """Extract marginal distribution for specified indices."""
        idx = jnp.atleast_1d(indices)
        row, col = jnp.ix_(idx, idx)
        if isinstance(self.covariance, CovarianceBase):
            if isinstance(indices, slice):
                mcov = self.covariance[..., indices, indices]
            else:
                mcov = self.covariance.at[..., row, col]

        else:
            mcov = self.covariance[..., row, col]

        return GaussianRV(self.mean[..., indices], mcov)

    def conditional_mean(
        self,
        other: GaussianRV[Any],
        cross_covariance: JaxFloatArray,
        given_value: JaxFloatArray | None = None,
    ) -> GaussianRV[Any]:
        r"""Compute the conditional mean of self given other.

        Given joint distribution of [X1, X2] where:
        - X1 (self) has mean μ1 and covariance Σ11
        - X2 (other) has mean μ2 and covariance Σ22
        - Cross-covariance: Σ12 = Cov(X1, X2) = cross_covariance

        Returns the conditional mean of X1|X2=x2 where:
        - If given_value is provided: condition on X2 = given_value
        - If given_value is None: condition on X2 = μ2 (its mean)


        .. math::
            \mu_1|2 = \mu_1 + \Sigma_{12} @ \Sigma{22}^(-1) @ (x_2 - \mu_2)

        Args:
            other: The GaussianRV to condition on (X2)
            cross_covariance: Cross-covariance matrix Σ12 with shape (..., n1, n2)
                            where n1 = len(self) and n2 = len(other)
            given_value: The value to condition on. If None, uses other.mean
                        Shape should be compatible with other.mean

        Returns:
            The conditional mean of X1|X2=given_value
        """

        if given_value is None:
            x2 = other.mean
        else:
            x2 = jnp.asarray(given_value, dtype=other.mean.dtype)

        residual = x2 - other.mean
        sigma22_inv_residual = solve_symmetric_cholesky(
            other.covariance, residual[..., jnp.newaxis]
        )[..., 0]

        return self.mean + jnp.einsum("...ij,...j->...i", cross_covariance, sigma22_inv_residual)

    def conditional(
        self,
        other: GaussianRV[Any],
        cross_covariance: JaxFloatArray,
        given_value: JaxFloatArray | None = None,
    ) -> GaussianRV[Any]:
        """Compute the conditional distribution of self given other.

        Given joint distribution of [X1, X2] where:
        - X1 (self) has mean μ1 and covariance Σ11
        - X2 (other) has mean μ2 and covariance Σ22
        - Cross-covariance: Σ12 = Cov(X1, X2) = cross_covariance

        Returns the conditional distribution X1|X2=x2 where:
        - If given_value is provided: condition on X2 = given_value
        - If given_value is None: condition on X2 = μ2 (its mean)

        The conditional distribution is:
        X1|X2=x2 ~ N(μ1|2, Σ1|2)
        where:
        - μ1|2 = μ1 + Σ12 @ Σ22^(-1) @ (x2 - μ2)
        - Σ1|2 = Σ11 - Σ12 @ Σ22^(-1) @ Σ21

        Args:
            other: The GaussianRV to condition on (X2)
            cross_covariance: Cross-covariance matrix Σ12 with shape (..., n1, n2)
                            where n1 = len(self) and n2 = len(other)
            given_value: The value to condition on. If None, uses other.mean
                        Shape should be compatible with other.mean

        Returns:
            GaussianRV: The conditional distribution X1|X2=given_value
        """
        # Validate inputs
        cross_covariance = jnp.asarray(cross_covariance, dtype=self.mean.dtype)

        # Check dimensions
        n1 = len(self)
        n2 = len(other)

        if cross_covariance.shape[-2:] != (n1, n2):
            raise ValueError(
                f"Cross-covariance shape {cross_covariance.shape} incompatible "
                f"with self dimension {n1} and other dimension {n2}"
            )

        # Set conditioning value
        if given_value is None:
            x2 = other.mean
        else:
            x2 = jnp.asarray(given_value, dtype=other.mean.dtype)

        residual = x2 - other.mean

        # Compute Σ22^(-1) @ residual
        sigma22_inv_residual = solve_symmetric_cholesky(
            other.covariance, residual[..., jnp.newaxis]
        )[..., 0]

        # Compute Σ22^(-1) @ Σ21
        sigma22_inv_sigma21 = solve_symmetric_cholesky(other.covariance, cross_covariance.mT)

        # Compute conditional mean: μ1 + Σ12 @ Σ22^(-1) @ (x2 - μ2)
        conditional_mean = self.mean + jnp.einsum(
            "...ij,...j->...i", cross_covariance, sigma22_inv_residual
        )

        # Compute conditional covariance: Σ11 - Σ12 @ Σ22^(-1) @ Σ21
        # Shape: (..., n1, n1) - (..., n1, n2) @ (..., n2, n1) -> (..., n1, n1)
        conditional_cov = self.covariance - jnp.einsum(
            "...ik,...kj->...ij", cross_covariance, sigma22_inv_sigma21
        )

        return GaussianRV(conditional_mean, conditional_cov)

    def joint(
        self,
        other: GaussianRV[Any],
        cross_covariance: JaxFloatArray,
        covariance_type: str = "array",
    ) -> GaussianRV[Any]:
        """Create joint distribution of self and other.

        Given:
        - X1 (self) with mean μ1 and covariance Σ11
        - X2 (other) with mean μ2 and covariance Σ22
        - Cross-covariance Σ12 = Cov(X1, X2)

        Returns joint distribution of [X1; X2] with:
        - mean = [μ1; μ2]
        - covariance = [[Σ11, Σ12], [Σ21, Σ22]]

        Args:
            other: Another GaussianRV
            cross_covariance: Cross-covariance matrix with shape (..., n1, n2)

        Returns:
            GaussianRV: Joint distribution
        """
        if covariance_type not in COV_SYMN_:
            raise ValueError(
                f"covariance_type:{covariance_type} not an allowable type. Allowable types are\n:{COV_SYMN_}"
            )

        cross_covariance = jnp.asarray(cross_covariance, dtype=self.mean.dtype)

        # Validate dimensions
        n1 = len(self)
        n2 = len(other)

        if cross_covariance.shape[-2:] != (n1, n2):
            raise ValueError(
                f"Cross-covariance shape {cross_covariance.shape} incompatible "
                f"with dimensions ({n1}, {n2})"
            )

        # Get common batch shape
        batch_shape = jnp.broadcast_shapes(
            self.shape[:-1], other.shape[:-1], cross_covariance.shape[:-2]
        )

        # Broadcast means
        self_mean_bc = jnp.broadcast_to(self.mean, batch_shape + (n1,))
        other_mean_bc = jnp.broadcast_to(other.mean, batch_shape + (n2,))

        # Concatenate means
        joint_mean = jnp.concatenate([self_mean_bc, other_mean_bc], axis=-1)

        # Broadcast covariances
        self_cov = (
            self.covariance if isinstance(self.covariance, jnp.ndarray) else self.covariance.full()
        )
        other_cov = (
            other.covariance
            if isinstance(other.covariance, jnp.ndarray)
            else other.covariance.full()
        )
        self_cov_bc = jnp.broadcast_to(self_cov, batch_shape + (n1, n1))
        other_cov_bc = jnp.broadcast_to(other_cov, batch_shape + (n2, n2))
        cross_cov_bc = jnp.broadcast_to(cross_covariance, batch_shape + (n1, n2))

        # Build joint covariance matrix
        joint_cov = jnp.block([[self_cov_bc, cross_cov_bc], [cross_cov_bc.mT, other_cov_bc]])

        jcov = cholesky_factor(joint_cov) if covariance_type in CHOLESK_SYMN_ else joint_cov
        return GaussianRV(joint_mean, jcov)

    def linear_cross(self, A: JaxFloatArray) -> JaxFloatArray:
        """Compute cross-covariance Cov(X, AX) = Σ_X @ A^T.

        This is useful for computing cross-covariances in filtering applications,
        particularly for Kalman filters where you need Cov(x, Hx) = P @ H^T.

        Args:
            A: Matrix with shape (..., m, n) where n = len(self)
            This transforms the random variable as Y = A @ X

        Returns:
            NDArray: Cross-covariance Cov(X, Y) = Σ_X @ A^T with shape (..., n, m)
        """
        A = jnp.asarray(A, dtype=self.mean.dtype)

        # Check dimensions
        n = len(self)
        if A.shape[-1] != n:
            raise ValueError(
                f"Matrix A column dimension {A.shape[-1]} must match state dimension {n}"
            )

        if isinstance(self.covariance, jnp.ndarray):
            return jnp.einsum("...ij,...kj->...ik", self.covariance, A)

        return linear_cross_covariance(self.covariance, A)  # type: ignore[return-value]

    @classmethod
    def zero_mean(cls, covariance: CovarianceType) -> Self:
        """Zero mean gaussian random variable."""
        return cls(jnp.zeros(covariance.shape[:-1]), covariance)

    @classmethod
    def concatenate[CovarianceT: CovarianceBase](
        cls,
        variables: Iterable[GaussianRV[CovarianceT]],
        axis: int = 0,
    ) -> GaussianRV[CovarianceT]:
        if len(variables) == 0:
            raise ValueError("Variables to concatenate must not be empty.")
        mean = jnp.concatenate([rv.mean for rv in variables], axis=axis)
        if isinstance(variables[0].covariance, jnp.ndarray):
            covariance = jnp.concatenate([rv.covariance for rv in variables], axis=axis)
        else:
            covariance = type(variables[0].covariance).concatenate(
                [rv.covariance for rv in variables], axis=axis
            )

        return GaussianRV(mean, covariance)
