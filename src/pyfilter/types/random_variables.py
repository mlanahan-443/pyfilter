from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self, cast

import numpy as np
from numpy.typing import ArrayLike

from pyfilter.config import FDTYPE_ as FDTYPE
from pyfilter.hints import ArrayIndex, FloatArray
from pyfilter.linear_solve import solve_symmetric_cholesky
from pyfilter.types.covariance import (
    CovarianceBase,
    cholesky_factor,
    linear_cross_covariance,
)

CHOLESK_SYMN_ = {"chofactor", "cho", "cholesky", "square-root"}
ARRAY_SYMN_ = {"array", "np.ndarray", "FloatArray", "Array"}
COV_SYMN_ = CHOLESK_SYMN_.union(ARRAY_SYMN_)

type CovarianceType = CovarianceBase | FloatArray
type Variable = GaussianRV[Any] | FloatArray | CovarianceBase | float


class _ArrayUfuncWrangler:
    """Handles NumPy ufunc dispatching for GaussianRV objects.

    This class provides a clean way to route NumPy ufuncs (add, subtract, multiply, matmul)
    to the appropriate GaussianRV methods.
    """

    def __init__(self, grv_instance: GaussianRV[Any]):
        self.grv = grv_instance

        # Map ufuncs to corresponding GaussianRV methods
        self.ufunc_map: dict[Any, Callable[..., Any]] = {
            np.add: self._handle_add,
            np.subtract: self._handle_subtract,
            np.multiply: self._handle_multiply,
            np.matmul: self._handle_matmul,
        }

    def __call__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """Main entry point for __array_ufunc__."""
        # We only support '__call__' method, not '__reduce__', '__accumulate__', etc.
        if method != "__call__":
            return NotImplemented

        # Check if 'out' parameter is provided (indicating in-place operation)
        out = kwargs.get("out", None)
        if out is not None:
            # For in-place operations, we should delegate to the in-place methods
            # But since NumPy passes 'out', we need to handle this differently
            return NotImplemented

        handler = self.ufunc_map.get(ufunc)
        if handler is None:
            return NotImplemented

        return handler(*inputs, **kwargs)

    def _identify_operands(self, *inputs: Any) -> tuple[GaussianRV[Any], Any]:
        """Identify which operand is the GaussianRV and which is the other."""
        if len(inputs) != 2:
            raise NotImplementedError("Expected exactly 2 inputs")

        # Determine order: (self, other) or (other, self)
        if isinstance(inputs[0], GaussianRV):
            return inputs[0], inputs[1]
        else:
            return inputs[1], inputs[0]

    def _handle_add(self, *inputs: Any, **kwargs: Any) -> GaussianRV[Any]:
        """Handle np.add ufunc."""
        grv_operand, other_operand = self._identify_operands(*inputs)

        # Addition is commutative, so order doesn't matter
        if grv_operand is inputs[0]:
            return grv_operand.__add__(other_operand)
        else:
            return grv_operand.__radd__(other_operand)

    def _handle_subtract(self, *inputs: Any, **kwargs: Any) -> Any:
        """Handle np.subtract ufunc."""
        grv_operand, other_operand = self._identify_operands(*inputs)

        # Subtraction is NOT commutative, so order matters
        if grv_operand is inputs[0]:
            # grv - other
            return grv_operand.__sub__(other_operand)
        else:
            # other - grv
            return grv_operand.__rsub__(other_operand)

    def _handle_multiply(self, *inputs: Any, **kwargs: Any) -> Any:
        """Handle np.multiply ufunc."""
        grv_operand, other_operand = self._identify_operands(*inputs)

        # Multiplication is commutative for our purposes
        if grv_operand is inputs[0]:
            return grv_operand.__mul__(other_operand)
        else:
            return grv_operand.__rmul__(other_operand)

    def _handle_matmul(self, *inputs: Any, **kwargs: Any) -> GaussianRV[Any]:
        """Handle np.matmul ufunc."""
        grv_operand, other_operand = self._identify_operands(*inputs)

        # Matrix multiplication is NOT commutative
        if grv_operand is inputs[0]:
            # grv @ other
            return grv_operand.__matmul__(other_operand)
        else:
            # other @ grv
            return grv_operand.__rmatmul__(other_operand)


@dataclass
class GaussianRV[Covariance: CovarianceType]:
    mean: FloatArray
    covariance: Covariance

    def __post_init__(self) -> None:
        """Validate that mean and covariance have compatible shapes."""

        # Check that mean has at least 1 dimension
        if self.mean.ndim < 1:
            raise ValueError(
                f"Mean must have at least 1 dimension, got shape {self.mean.shape}"
            )

        # Check that covariance has at least 2 dimensions
        if self.covariance.ndim < 2:
            raise ValueError(
                f"Covariance must have at least 2 dimensions, got shape {self.covariance.shape}"
            )

        # Check that the last two dimensions of covariance are square
        if self.covariance.shape[-1] != self.covariance.shape[-2]:
            raise ValueError(
                f"Last two dimensions of covariance must be square, got shape {self.covariance.shape}"
            )

        # Check that dimensions match
        n = self.mean.shape[-1]
        if self.covariance.shape[-1] != n:
            raise ValueError(
                f"Last dimension of mean ({n}) must match last dimensions of covariance ({self.covariance.shape[-1]})"
            )

        # Check that batch dimensions match
        mean_batch = self.mean.shape[:-1]
        cov_batch = self.covariance.shape[:-2]
        if mean_batch != cov_batch:
            raise ValueError(
                f"Batch dimensions of mean {mean_batch} and covariance {cov_batch} must match"
            )

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Allow NumPy ufuncs to operate on GaussianRV objects.

        Delegates to _ArrayUfuncWrangler for clean ufunc handling.
        Supports: add, subtract, multiply, and matmul.
        """
        wrangler = _ArrayUfuncWrangler(self)
        return wrangler(ufunc, method, *inputs, **kwargs)

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
        elif isinstance(other, np.ndarray):
            # For array operations, check if broadcasting is valid
            try:
                np.broadcast_shapes(self.mean.shape, other.shape)
            except ValueError:
                raise ValueError(
                    f"Cannot broadcast shapes {self.mean.shape} and {other.shape}"
                )
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

    def __mul__(self, other: FloatArray | float) -> GaussianRV[Any]:
        """Multiply GaussianRV by a deterministic matrix or scalar.

        For scalar a: Y = aX -> mean_Y = a*mean_X, cov_Y = a²*cov_X
        For matrix A: Y = AX -> mean_Y = A*mean_X, cov_Y = A*cov_X*A^T
        """
        other = np.asarray(other, dtype=FDTYPE)

        if other.ndim == 0:  # scalar
            return GaussianRV(self.mean * other, self.covariance * (other**2))  # type: ignore[operator]
        elif other.ndim == 1:  # element-wise multiplication
            # Treat as diagonal matrix multiplication
            self._check_compatible(other)
            # Broadcasting for batch dimensions
            if isinstance(self.covariance, np.ndarray):
                cov_scale = other[..., :, np.newaxis] * other[..., np.newaxis, :]
                return GaussianRV(self.mean * other, self.covariance * cov_scale)
            else:
                raise TypeError(
                    "If gaussian random variable has a Covariance class covariance, then only elementwise multiplication with scalrs is allowed."
                )

        else:  # matrix multiplication
            # other has shape (..., m, n) where n matches last dim of mean
            if other.shape[-1] != len(self):
                raise ValueError(
                    f"Matrix dimension mismatch: {other.shape} @ {self.mean.shape}"
                )

            # Batch matrix multiply for mean: (..., m, n) @ (..., n) -> (..., m)
            new_mean = np.einsum("...ij,...j->...i", other, self.mean)

            # Batch computation of A @ Cov @ A.T
            if isinstance(self.covariance, np.ndarray):
                # Step 1: A @ Cov -> (..., m, n) @ (..., n, n) -> (..., m, n)
                temp = np.einsum("...ij,...jk->...ik", other, self.covariance)
                # Step 2: (A @ Cov) @ A.T -> (..., m, n) @ (..., n, m) -> (..., m, m)
                new_cov = np.einsum("...ij,...kj->...ik", temp, other)
            else:
                new_cov = self.covariance.quadratic_form(other)

            return GaussianRV(new_mean, new_cov)

    def __rmul__(self, other: FloatArray | float) -> GaussianRV[Any]:
        """Right multiplication (for scalar/array * GaussianRV)."""
        return self.__mul__(other)

    def __matmul__(self, other: FloatArray) -> GaussianRV[Any]:
        """Matrix multiplication using @ operator (same as __mul__ for matrices)."""
        if not isinstance(other, np.ndarray) or other.ndim < 2:
            raise ValueError("@ operator requires a matrix (array with ndim >= 2)")
        return self.__mul__(other)

    def __rmatmul__(self, other: FloatArray) -> GaussianRV[Any]:
        """Right matrix multiplication (for A @ self)."""
        # 'other' is the matrix A on the left
        # We can just call our existing __matmul__ method,
        # which correctly validates 'other' and calls __mul__.
        return self.__matmul__(other)

    def __repr__(self) -> str:
        trace = (
            np.trace(self.covariance, axis1=-2, axis2=-1).sum()
            if isinstance(self.covariance, np.ndarray)
            else self.covariance.trace()
        )
        return f"GaussianRV(shape={self.shape}, mean_norm={np.linalg.norm(self.mean):.3f}, cov_trace={trace})"

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
        idx = np.atleast_1d(cast("ArrayLike", indices))
        row, col = np.ix_(idx, idx)
        if isinstance(self.covariance, CovarianceBase):
            if isinstance(indices, slice):
                mcov = self.covariance[..., indices, indices]  # type: ignore[assignment]
            else:
                mcov = self.covariance.at[..., row, col]  # type: ignore[index,assignment]

        else:
            mcov = self.covariance[..., row, col]  # type: ignore[index]

        return GaussianRV(self.mean[..., indices], mcov)

    def conditional_mean(
        self,
        other: GaussianRV[Any],
        cross_covariance: FloatArray,
        given_value: FloatArray | None = None,
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
            x2 = np.asarray(given_value, dtype=FDTYPE)

        residual = x2 - other.mean
        sigma22_inv_residual = solve_symmetric_cholesky(
            other.covariance, residual[..., np.newaxis]
        )[..., 0]

        return self.mean + np.einsum(
            "...ij,...j->...i", cross_covariance, sigma22_inv_residual
        )

    def conditional(
        self,
        other: GaussianRV[Any],
        cross_covariance: FloatArray,
        given_value: FloatArray | None = None,
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
        cross_covariance = np.asarray(cross_covariance, dtype=FDTYPE)

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
            x2 = np.asarray(given_value, dtype=FDTYPE)

        residual = x2 - other.mean

        # Compute Σ22^(-1) @ residual
        sigma22_inv_residual = solve_symmetric_cholesky(
            other.covariance, residual[..., np.newaxis]
        )[..., 0]

        # Compute Σ22^(-1) @ Σ21
        sigma22_inv_sigma21 = solve_symmetric_cholesky(
            other.covariance, cross_covariance.mT
        )

        # Compute conditional mean: μ1 + Σ12 @ Σ22^(-1) @ (x2 - μ2)
        conditional_mean = self.mean + np.einsum(
            "...ij,...j->...i", cross_covariance, sigma22_inv_residual
        )

        # Compute conditional covariance: Σ11 - Σ12 @ Σ22^(-1) @ Σ21
        # Shape: (..., n1, n1) - (..., n1, n2) @ (..., n2, n1) -> (..., n1, n1)
        conditional_cov = self.covariance - np.einsum(
            "...ik,...kj->...ij", cross_covariance, sigma22_inv_sigma21
        )

        return GaussianRV(conditional_mean, conditional_cov)

    def joint(
        self,
        other: GaussianRV[Any],
        cross_covariance: FloatArray,
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

        cross_covariance = np.asarray(cross_covariance, dtype=FDTYPE)

        # Validate dimensions
        n1 = len(self)
        n2 = len(other)

        if cross_covariance.shape[-2:] != (n1, n2):
            raise ValueError(
                f"Cross-covariance shape {cross_covariance.shape} incompatible "
                f"with dimensions ({n1}, {n2})"
            )

        # Get common batch shape
        batch_shape = np.broadcast_shapes(
            self.shape[:-1], other.shape[:-1], cross_covariance.shape[:-2]
        )

        # Broadcast means
        self_mean_bc = np.broadcast_to(self.mean, batch_shape + (n1,))
        other_mean_bc = np.broadcast_to(other.mean, batch_shape + (n2,))

        # Concatenate means
        joint_mean = np.concatenate([self_mean_bc, other_mean_bc], axis=-1)

        # Broadcast covariances
        self_cov = (
            self.covariance
            if isinstance(self.covariance, np.ndarray)
            else self.covariance.full()
        )
        other_cov = (
            other.covariance
            if isinstance(other.covariance, np.ndarray)
            else other.covariance.full()
        )
        self_cov_bc = np.broadcast_to(self_cov, batch_shape + (n1, n1))
        other_cov_bc = np.broadcast_to(other_cov, batch_shape + (n2, n2))
        cross_cov_bc = np.broadcast_to(cross_covariance, batch_shape + (n1, n2))

        # Build joint covariance matrix
        # [[Σ11, Σ12],
        #  [Σ21, Σ22]]
        joint_cov = np.zeros(batch_shape + (n1 + n2, n1 + n2), dtype=FDTYPE)
        joint_cov[..., :n1, :n1] = self_cov_bc
        joint_cov[..., n1:, n1:] = other_cov_bc
        joint_cov[..., :n1, n1:] = cross_cov_bc
        joint_cov[..., n1:, :n1] = np.swapaxes(cross_cov_bc, -2, -1)

        jcov = (
            cholesky_factor(joint_cov)
            if covariance_type in CHOLESK_SYMN_
            else joint_cov
        )
        return GaussianRV(joint_mean, jcov)

    def linear_cross(self, A: FloatArray) -> FloatArray:
        """Compute cross-covariance Cov(X, AX) = Σ_X @ A^T.

        This is useful for computing cross-covariances in filtering applications,
        particularly for Kalman filters where you need Cov(x, Hx) = P @ H^T.

        Args:
            A: Matrix with shape (..., m, n) where n = len(self)
            This transforms the random variable as Y = A @ X

        Returns:
            NDArray: Cross-covariance Cov(X, Y) = Σ_X @ A^T with shape (..., n, m)
        """
        A = np.asarray(A, dtype=FDTYPE)

        # Check dimensions
        n = len(self)
        if A.shape[-1] != n:
            raise ValueError(
                f"Matrix A column dimension {A.shape[-1]} must match state dimension {n}"
            )

        if isinstance(self.covariance, np.ndarray):
            return np.einsum("...ij,...kj->...ik", self.covariance, A)

        return linear_cross_covariance(self.covariance, A)  # type: ignore[return-value]

    @classmethod
    def zero_mean(cls, covariance: CovarianceType) -> Self:
        """Zero mean gaussian random variable."""
        return cls(np.zeros(covariance.shape[:-1]), covariance)
