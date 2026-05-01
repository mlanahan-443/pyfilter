from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Number
from typing import Any, Self, overload

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ..config import CHOLESKY_CHECK_FINITE_, MONITOR_PERFORMANCE_
from ..hints import ArrayIndex, FloatArray, IndexItem
from ..performance_util import performance_monitor
from ..util import (
    _IndexerMethod,
    implements_ufunc,
    left_broadcast_arrays,
    normalize_index,
)

type CovarianceType = CovarianceBase | FloatArray

ALLOWED_TYPES_ = [
    "CholeskyFactorCovariance",
    "DiagonalCovariance",
    "np.ndarray",
    "Number",
]


def type_error_msg(object: Any) -> str:
    return f"Covariance must be one of: {ALLOWED_TYPES_} not: {type(object)}"


CHOL_UFUNC_CACHE_: dict[Callable[..., Any], Callable[..., Any]] = {}
DIAG_UFUNC_CACHE_: dict[Callable[..., Any], Callable[..., Any]] = {}


class CovarianceBase(ABC):
    """Class for representation of covariances.

    The default is to avoid the use of full matrices and instead the cholesky
    factor for various operations whenever possible for both numerical stability and
    performance.
    """

    _UFUNC_CACHE: dict[Callable[..., Any], Callable[..., Any]] = {}
    # Set higher priority than ndarray to ensure our methods are called first
    __array_priority__ = 1000

    def __init__(self, matrix_shape: tuple[int, int]):
        self._matrix_shape = matrix_shape

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return self._matrix_shape

    @property
    def diagonal_indices(self) -> tuple[Any, ...]:
        return np.diag_indices(self.matrix_shape[0])

    @property
    @abstractmethod
    def ndim(self) -> int: ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: ...

    @abstractmethod
    def copy(self) -> CovarianceBase: ...

    @abstractmethod
    def __id__(self) -> tuple[int, int]:
        """Return (memory_id, content_hash) for the underlying data.

        Returns:
            tuple[int, int]: A tuple of (memory address, content hash)
        """
        pass

    @property
    @abstractmethod
    def variance(self) -> FloatArray:
        """Gets the diagonal elements of the covariance.

        This is called "variance" to avoid confusion with "diagonal" which is ambiguous.

        Returns:
            FloatArray: the variance of the covariance matrix.
        """
        ...

    @property
    @abstractmethod
    def cholesky_factor(self) -> FloatArray:
        """Provides the cholesky factor of the covariance matrix.

        Returns:
            FloatArray: The lower triangular cholesky factorization of the covariance matrix.
        """
        ...

    @abstractmethod
    def full(self) -> FloatArray:
        """Provides the full matrix.

        Returns:
            FloatArray: The full matrix representation of the covariance.
        """
        ...

    @abstractmethod
    def quadratic_form(self, other: FloatArray) -> CovarianceBase:
        """Computes the quadratic product A @ S @ A.T."""
        ...

    @abstractmethod
    def _add_to_covariance(self, other: CovarianceType) -> CovarianceBase:
        """Called when self is added to a Covariance object.

        Args:
            other: The other covariance object.
        """
        ...

    @abstractmethod
    def _add_to_diagonal(self, other: DiagonalCovariance) -> CovarianceBase:
        """Called when self is added to a Diagonal Covariance object.

        Args:
            other: The diagonal covariance object.
        """
        ...

    @abstractmethod
    def __sub__(self, other: CovarianceBase) -> CovarianceBase:
        """Subtraction of one covariance object from another.

        Args:
            other (Other): The other covariance object.

        Returns:
            CovarianceBase: The covariance object
        """
        ...

    @abstractmethod
    def __add__(self, other: CovarianceType) -> CovarianceBase:
        """Addition of one covariance object with another.

        Args:
            other (Other): The other covariance object.

        Returns:
            CovarianceBase: The covariance object
        """
        ...

    @abstractmethod
    def __mul__(self, other: float) -> CovarianceBase:
        """Multiplication of covariance object by scalar value.

        Args:
            other: the scalar value.

        Returns:
            CovarianceBase: The covariance object
        """
        ...

    def trace(self) -> FloatArray:
        """Trace of the matrix.

        Trace of a triangular matrix is the sum of its diagonal components.

        Returns:
            The trace of the batch covariances.
        """
        result: FloatArray = np.sum(self.variance, axis=-1)
        return result

    @property
    def at(self) -> _IndexerMethod[Self]:
        """Accessor for expensive/arbitrary indexing (reconstructs structure)."""
        return _IndexerMethod(self._apply_at_indexing)

    @property
    def biloc(self) -> _IndexerMethod[Self]:
        """Accessor for batch integer location indexing."""
        return _IndexerMethod(self._apply_biloc_indexing)

    @abstractmethod
    def _apply_at_indexing(self, index: ArrayIndex) -> Self: ...

    @abstractmethod
    def _apply_biloc_indexing(self, index: ArrayIndex) -> Self: ...

    @abstractmethod
    def _is_safe_matrix_slice(self, matrix_indexer: tuple[IndexItem, ...]) -> bool:
        """
        Returns True if the matrix_indexer preserves the internal structure
        (Triangular for Cholesky, Diagonal for Diagonal).
        """
        ...

    @abstractmethod
    def _apply_fast_matrix_slice(
        self, batch_idx: tuple[Any, ...], matrix_idx: tuple[Any, ...]
    ) -> Self:
        """
        Performs the fast slicing on the specific underlying data (_L or _D).
        """
        ...

    def _get_norm_index(self, index: ArrayIndex) -> tuple[Any, ...]:
        """Split index into batch and matrix indices."""
        try:
            full_ndim = self.ndim
            norm_index = normalize_index(index, full_ndim)
        except IndexError:
            # Fallback for simple cases if normalizer fails or isn't imported
            if not isinstance(index, tuple):
                norm_index = (index,)
            else:
                norm_index = index

        return norm_index

    def __getitem__(self, index: ArrayIndex) -> Self:
        """Get covariance object at the requested indices.

        Only leading indices, i.e. 0,..., k < n where n is the dimension of the
        Covariance matrix result in a valid cholesky factor.

        Args:
            index: The index for the underlying array.

        Returns:
            The subset of the covariance object.

        Raises
            ValueError: If the passed object is not a valid slice.
        """

        norm_index = self._get_norm_index(index)

        n_indices = len(norm_index)
        batch_dims = self.ndim - 2

        matrix_indexer = norm_index[-2:]
        batch_indexer = norm_index[:-2]

        # Case 1: This is only batch indexing.
        if n_indices < batch_dims:
            return self.biloc[index]

        # Case 2: The indexing is a combination of batch indexing and slicing.
        if self._is_safe_matrix_slice(matrix_indexer):
            return self._apply_fast_matrix_slice(batch_indexer, matrix_indexer)

        return self.at[index]

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> CovarianceBase:
        if func not in self._UFUNC_CACHE:
            return NotImplemented  # type: ignore[return-value]
        result: CovarianceBase = self._UFUNC_CACHE[func](*args, **kwargs)
        return result

    @abstractmethod
    def inverse(self) -> FloatArray:
        """The inverse of the covariance matrix."""
        ...


def cholesky_factor(A: FloatArray) -> CholeskyFactorCovariance:
    """Light wrapper around cho_factor to improve code readability"""

    L = cho_factor(
        A, lower=True, overwrite_a=True, check_finite=CHOLESKY_CHECK_FINITE_
    )[0]
    return CholeskyFactorCovariance(np.tril(L))


class CholeskyFactorCovariance(CovarianceBase):
    _UFUNC_CACHE = CHOL_UFUNC_CACHE_
    """Nominal covariance type."""

    def __init__(self, L: FloatArray, copy: bool = True, immutable: bool = False):
        if L.ndim < 2:
            raise ValueError(
                "Matrix must be at least 2-D to specify a valid covariance."
            )

        if L.shape[-1] != L.shape[-2]:
            raise ValueError("Last two dimensions must specify a square matrix.")

        super().__init__((L.shape[-2], L.shape[-1]))
        self._L = L.copy() if copy else L
        if immutable:
            self._L.setflags(write=False)

    def __id__(self) -> tuple[int, int]:
        """Return unique identifier for current state of _L."""
        return (id(self._L), hash(self._L.tobytes()))

    def copy(self) -> CholeskyFactorCovariance:
        return CholeskyFactorCovariance(self._L.copy())

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying matrix"""
        return self._L.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying matrix"""
        return self._L.shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape"""
        return self._L.shape[:-2]

    @property
    def cholesky_factor(self) -> FloatArray:
        """The cholesky factor is just the matrix L.

        Returns:
            FloatArray: The cholesky factor.
        """
        return self._L

    @property
    def variance(self) -> FloatArray:
        """The variance

        A somewhat surprising result: Var(P) = diag(P) = ||L[i,:]||_2^2

        Returns:
            FloatArray: The variance
        """
        result: FloatArray = np.linalg.norm(self.cholesky_factor, axis=-1) ** 2
        return result

    def _add_to_covariance(self, other: CovarianceType) -> CholeskyFactorCovariance:
        """Addition of one covariance object (other) to self.

        The provided covariance may be either a Covariance or just a FloatArray.
        """
        return cholesky_factor(
            self.full()
            + (other.full() if isinstance(other, CholeskyFactorCovariance) else other)
        )

    def _sub_covariance(
        self, other: CholeskyFactorCovariance | FloatArray
    ) -> CholeskyFactorCovariance:
        """Addition of one covariance object (other) to self.

        The provided covariance may be either a Covariance or just a FloatArray.
        """
        return cholesky_factor(
            self.full()
            - (other.full() if isinstance(other, CholeskyFactorCovariance) else other)
        )

    def _add_to_diagonal(self, other: DiagonalCovariance) -> CholeskyFactorCovariance:
        """Addition of a diagonal covariance object (other) to self.

        The provided covariance object must be a DiagonalCovariance.

        Args:
            other (DiagonalCovariance): The other covariance instance.

        Returns:
            Covariance: The resulting covariance.
        """

        mat = self.full()
        mat[..., *self.diagonal_indices] += other.variance
        return cholesky_factor(mat)

    def _sub_diagonal(self, other: DiagonalCovariance) -> CholeskyFactorCovariance:
        """Addition of a diagonal covariance object (other) to self.

        The provided covariance object must be a DiagonalCovariance.

        Args:
            other (DiagonalCovariance): The other covariance instance.

        Returns:
            Covariance: The resulting covariance.
        """

        mat = self.full()
        mat[..., *self.diagonal_indices] -= other.variance
        return cholesky_factor(mat)

    @overload
    def __add__(self, other: CholeskyFactorCovariance) -> CholeskyFactorCovariance: ...

    @overload
    def __add__(self, other: DiagonalCovariance) -> CholeskyFactorCovariance: ...

    @overload
    def __add__(self, other: FloatArray) -> CholeskyFactorCovariance: ...

    @overload
    def __add__(self, other: CovarianceType) -> CholeskyFactorCovariance: ...

    def __add__(self, other: CovarianceType) -> CholeskyFactorCovariance:
        """Add covariance with another covariance object.

        I believe the fastest way to compute this is just computing
        the full matrices and then refactoring.
        """

        if isinstance(other, (CholeskyFactorCovariance, np.ndarray, Number)):
            return self._add_to_covariance(other)
        elif isinstance(other, DiagonalCovariance):
            return self._add_to_diagonal(other)
        else:
            raise TypeError(type_error_msg(other))

    @overload
    def __radd__(self, other: CholeskyFactorCovariance) -> CholeskyFactorCovariance: ...

    @overload
    def __radd__(self, other: DiagonalCovariance) -> CholeskyFactorCovariance: ...

    @overload
    def __radd__(self, other: FloatArray) -> CholeskyFactorCovariance: ...

    @overload
    def __radd__(self, other: CovarianceType) -> CholeskyFactorCovariance: ...

    def __radd__(self, other: CovarianceType) -> CholeskyFactorCovariance:
        """Addition of is communative.

        Args:
            other (Other): The other covariance object.

        Returns:
            The resulting covariance object.
        """

        return self.__add__(other)

    def __sub__(self, other: CovarianceType) -> CholeskyFactorCovariance:
        """Subtract covariance from the current covariance object.

        I believe the fastest way to compute this is just computing
        the full matrices and then refactoring.
        """

        if isinstance(other, (CholeskyFactorCovariance, np.ndarray, Number)):
            return self._sub_covariance(other)
        elif isinstance(other, DiagonalCovariance):
            return self._sub_diagonal(other)
        else:
            raise TypeError(type_error_msg(other))

    def __mul__(self, other: float) -> CholeskyFactorCovariance:
        """Multiply covariance by a scalar value.

        Args:
            other (float): The scalar value.

        Returns:
            CholeskyFactorCovariance: The resulting covariance object.
        """

        return CholeskyFactorCovariance(other**0.5 * self.cholesky_factor)

    def quadratic_form(self, other: FloatArray) -> CholeskyFactorCovariance:
        """Computes the quadratic product A @ S @ A.T.

        Let S be the covariance (self) and other an array
        that is broadcastable to the last two dimensions of
        the covariance of self. S = L L^T., Then:

        A @ S @ A.T = A @ L @ L.T @ A.T = (A @ L) @ (A @ L).T

        When A is m x n and L is n x n, A @ L is m x n (wide factor).
        We use QR decomposition to get the square m x m Cholesky factor.

        Args:
            other (FloatArray): The matrix A.

        Returns:
            Covariance: The quadratic product of the current covariance.
        """
        # Compute A @ L (may be wide if A is not square)
        AL = np.einsum("...ik,...kj->...ij", other, self._L)

        # If the result is square, return directly
        if AL.shape[-2] == AL.shape[-1]:
            return CholeskyFactorCovariance(AL)

        # Otherwise, use QR to get square Cholesky factor
        # M M^T = (A @ L) @ (A @ L)^T, and M^T = QR, so R^T is the Cholesky factor
        qr_result = np.linalg.qr(AL.swapaxes(-2, -1), mode="r")
        R = qr_result if isinstance(qr_result, np.ndarray) else qr_result[0]
        return CholeskyFactorCovariance(R.swapaxes(-2, -1))

    @performance_monitor(
        warn_threshold=2,
        enable_warnings=MONITOR_PERFORMANCE_,
        monitor=MONITOR_PERFORMANCE_,
    )
    def full(self) -> FloatArray:
        """The full matrix representation

        That is LL^T.

        Returns:
            FloatArray: the full matrix representation of the covariance.
        """
        result: FloatArray = np.einsum("...ik,...jk->...ij", self._L, self._L)
        return result

    @performance_monitor(
        warn_threshold=2,
        enable_warnings=MONITOR_PERFORMANCE_,
        monitor=MONITOR_PERFORMANCE_,
    )
    def _is_safe_matrix_slice(self, items: tuple[slice, ...]) -> bool:
        """
        Determines if a matrix index preserves the triangular structure of L.
        Safe: slice(None), slice(0, k, 1)
        Unsafe: slice(1, k), [0, 1], etc.
        """
        if len(items) != 2:
            return False

        if items[0] != items[1]:
            return False

        item = items[0]
        start_safe = item.start is None or item.start == 0
        step_safe = item.step is None or item.step == 1
        return start_safe and step_safe

    def _apply_fast_matrix_slice(
        self, batch_idx: tuple[Any, ...], matrix_idx: tuple[Any, ...]
    ) -> CholeskyFactorCovariance:
        L_view = self._L[batch_idx]
        return CholeskyFactorCovariance(L_view[..., *matrix_idx], copy=False)

    def _apply_biloc_indexing(self, index: ArrayIndex) -> CholeskyFactorCovariance:
        return CholeskyFactorCovariance(self._L[index], copy=False)

    def _apply_at_indexing(self, index: ArrayIndex) -> CholeskyFactorCovariance:
        new_P = self.full()[index]
        # 2. Check if the last two dimensions (the matrix) are square.
        if new_P.shape[-1] != new_P.shape[-2]:
            raise IndexError(
                f"Indexing key '{index}' resulted in a non-square matrix shape {new_P.shape}. "
                f"Last two dimensions were ({new_P.shape[-2]}, {new_P.shape[-1]})."
            )

        return cholesky_factor(new_P)

    @implements_ufunc(np.broadcast_to, CHOL_UFUNC_CACHE_)
    def broadcast_to(
        self, shape: tuple[int, ...], subok: bool = False
    ) -> CholeskyFactorCovariance:
        norm_index = self._get_norm_index(shape)
        batch_shape, matrix_shape = norm_index[:-2], norm_index[-2:]
        if matrix_shape != self.matrix_shape:
            raise ValueError(
                f"Cannot broadcast the matrix shape of DiagonalCovariance: {self.matrix_shape} to the new matrix shape: {matrix_shape})"
            )
        return CholeskyFactorCovariance(
            np.broadcast_to(self._L, batch_shape + matrix_shape, subok=subok)
        )

    def inverse(self) -> FloatArray:
        identity = np.broadcast_to(np.eye(*self.matrix_shape), self.shape)
        return solve_cholesky_covariance(self, identity)


class DiagonalCovariance(CovarianceBase):
    """Specify a covariance using the standard deviations.

    the standard deviations are used to maintain consistency with CholeskyFactorCovariance.
    """

    _UFUNC_CACHE = DIAG_UFUNC_CACHE_

    def __init__(self, D: FloatArray, copy: bool = True, immutable: bool = False):
        if D.ndim < 1:
            raise ValueError(
                "Matrix must be at least 1-D to specify a valid diagonal covariance."
            )

        super().__init__((D.shape[-1], D.shape[-1]))
        self._D = D.copy() if copy else D
        if immutable:
            self._D.flags.writeable = False

    def copy(self) -> DiagonalCovariance:
        return DiagonalCovariance(self._D.copy())

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying array"""
        return self.batch_shape + self.matrix_shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underyling matrix."""
        return len(self.shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The batch shape"""
        return self._D.shape[:-1]

    @property
    def variance(self) -> FloatArray:
        return self._D**2

    def __id__(self) -> tuple[int, int]:
        """Return unique identifier for current state of self._D"""
        return (id(self._D), hash(self._D.tobytes()))

    @performance_monitor(
        warn_threshold=2,
        enable_warnings=MONITOR_PERFORMANCE_,
        monitor=MONITOR_PERFORMANCE_,
    )
    def full(self) -> FloatArray:
        """The full matrix representation.

        Returns:
            FloatArray: The full matrix.
        """

        out = np.zeros(self.shape)
        out[..., *self.diagonal_indices] = self.variance
        return out

    @property
    def cholesky_factor(self) -> FloatArray:
        """The cholesky factor of a diagonal matrix.

        Returns:
            FloatArray: The cholesky factor.
        """
        out = np.zeros(self.shape)
        out[..., *self.diagonal_indices] = self._D
        return out

    def quadratic_form(self, other: FloatArray) -> CholeskyFactorCovariance:
        """Computes the quadratic product A @ D^2 @ A.T.

        Args:
            other (FloatArray): The matrix A.

        Returns:
            Covariance: The quadratic product of the current covariance.
        """

        return CholeskyFactorCovariance(other * self._D[..., np.newaxis, :])

    def _add_to_covariance(  # type: ignore[override]
        self, other: FloatArray | CholeskyFactorCovariance
    ) -> CholeskyFactorCovariance:
        """Addition of one covariance object (other) to self.

        The provided covariance may be either a Covariance or just a FloatArray.
        """
        mat = (
            other.full()
            if isinstance(other, CholeskyFactorCovariance)
            else other.copy()
        )
        mat[..., *self.diagonal_indices] += self.variance
        return cholesky_factor(mat)

    def _sub_covariance(
        self, other: CholeskyFactorCovariance | FloatArray
    ) -> CholeskyFactorCovariance:
        """Subtraction of one covariance object (other) from self (self - other).

        The provided covariance may be either a Covariance or just a FloatArray.
        """
        # Compute self - other where self is diagonal
        other_mat = (
            other.full()
            if isinstance(other, CholeskyFactorCovariance)
            else other.copy()
        )
        mat = self.full() - other_mat
        return cholesky_factor(mat)

    def _add_to_diagonal(self, other: DiagonalCovariance) -> DiagonalCovariance:
        """Addition of a diagonal covariance object (other) to self.

        The provided covariance object must be a DiagonalCovariance.

        Args:
            other (DiagonalCovariance): The other covariance instance.

        Returns:
            Covariance: The resulting covariance.
        """

        return DiagonalCovariance((other.variance + self.variance) ** 0.5)

    def _sub_diagonal(self, other: DiagonalCovariance) -> DiagonalCovariance:
        """Subtraction of a diagonal covariance object (other) from self.

        The provided covariance object must be a DiagonalCovariance.

        Args:
            other (DiagonalCovariance): The other covariance instance.

        Returns:
            Covariance: The resulting covariance.
        """

        return DiagonalCovariance((self.variance - other.variance) ** 0.5)

    @overload
    def __add__(self, other: DiagonalCovariance) -> DiagonalCovariance: ...

    @overload
    def __add__(self, other: CholeskyFactorCovariance) -> CholeskyFactorCovariance: ...

    @overload
    def __add__(self, other: FloatArray) -> CholeskyFactorCovariance: ...

    @overload
    def __add__(
        self, other: CovarianceType
    ) -> CholeskyFactorCovariance | DiagonalCovariance: ...

    def __add__(
        self, other: CovarianceType
    ) -> CholeskyFactorCovariance | DiagonalCovariance:
        """Add covariance with another covariance object.

        I believe the fastest way to compute this is just computing
        the full matrices and then refactoring.
        """

        if isinstance(other, (CholeskyFactorCovariance, np.ndarray)):
            return self._add_to_covariance(other)
        elif isinstance(other, DiagonalCovariance):
            return self._add_to_diagonal(other)
        else:
            raise TypeError(type_error_msg(other))

    @overload
    def __radd__(self, other: DiagonalCovariance) -> DiagonalCovariance: ...

    @overload
    def __radd__(self, other: CholeskyFactorCovariance) -> CholeskyFactorCovariance: ...

    @overload
    def __radd__(self, other: FloatArray) -> CholeskyFactorCovariance: ...

    @overload
    def __radd__(
        self, other: CovarianceType
    ) -> CholeskyFactorCovariance | DiagonalCovariance: ...

    def __radd__(
        self, other: CovarianceType
    ) -> CholeskyFactorCovariance | DiagonalCovariance:
        """Addition of is communative.

        Args:
            other (Other): The other covariance object.

        Returns:
            The covariance object.
        """

        return self.__add__(other)

    def __sub__(
        self, other: CovarianceType
    ) -> CholeskyFactorCovariance | DiagonalCovariance:
        """Subtract covariance from the current covariance object.

        I believe the fastest way to compute this is just computing
        the full matrices and then refactoring.
        """

        if isinstance(other, (CholeskyFactorCovariance, np.ndarray)):
            return self._sub_covariance(other)
        elif isinstance(other, DiagonalCovariance):
            return self._sub_diagonal(other)
        else:
            raise TypeError(type_error_msg(other))

    def __mul__(self, other: float) -> DiagonalCovariance:
        """Multiply covariance by a scalar value.

        Args:
            other (float): The scalar value.

        Returns:
            CholeskyFactorCovariance: The resulting covariance object.
        """

        return DiagonalCovariance(other**0.5 * self._D)

    def _is_safe_matrix_slice(self, matrix_indexer: tuple[IndexItem, ...]) -> bool:
        """
        Safe as long as the row index equals the column index.
        """
        row_idx, col_idx = matrix_indexer

        # Simple equality check: are we asking for the same rows as cols?
        return row_idx == col_idx

    def _apply_fast_matrix_slice(
        self, batch_idx: tuple[Any, ...], matrix_idx: tuple[Any, ...]
    ) -> DiagonalCovariance:
        # self._D is shape (Batch..., Dim)
        # matrix_idx passed from parent is (row_idx, col_idx)
        diag_idx = matrix_idx[0]

        full_index = batch_idx + (diag_idx,)

        return DiagonalCovariance(self._D[full_index], copy=False)

    def _apply_at_indexing(self, index: ArrayIndex) -> DiagonalCovariance:
        """
        Implementation for DiagonalCovariance.
        """
        norm_index = self._get_norm_index(index)

        # Extract matrix indices (last 2 elements for row and column)
        # For diagonal matrices, row and column indices should be the same
        matrix_row_idx = norm_index[-2] if len(norm_index) >= 2 else slice(None)
        matrix_col_idx = norm_index[-1]

        # For diagonal covariance, we only care about the diagonal elements
        # Use the row index if it's an array, otherwise use column index
        if isinstance(matrix_row_idx, np.ndarray):
            # For np.ix_ style indexing, take the diagonal (first row/col)
            if matrix_row_idx.ndim > 1:
                diagonal_idx = matrix_row_idx.ravel()
            else:
                diagonal_idx = matrix_row_idx
        elif isinstance(matrix_col_idx, np.ndarray):
            if matrix_col_idx.ndim > 1:
                diagonal_idx = matrix_col_idx.ravel()
            else:
                diagonal_idx = matrix_col_idx
        else:
            diag_idx_temp: Any = (
                matrix_row_idx if matrix_row_idx != slice(None) else matrix_col_idx
            )
            diagonal_idx = diag_idx_temp

        batch_indexer = norm_index[:-2]
        new_D = self._D[batch_indexer + (diagonal_idx,)]

        # Validation: Verify we didn't collapse the diagonal dimension accidentally
        if new_D.ndim == 0:
            raise IndexError("Indexing resulted in a scalar, invalid for Covariance.")

        return DiagonalCovariance(new_D, copy=False)

    def _apply_biloc_indexing(self, index: ArrayIndex) -> DiagonalCovariance:
        """
        Implementation for Batch Indexing.
        """
        # Assuming biloc is intended for pure batch dimensions?
        # Implementation depends on your specific definition of 'biloc'.

        new_D = self._D[index]  # Apply logic
        return DiagonalCovariance(new_D, copy=False)

    @implements_ufunc(np.broadcast_to, DIAG_UFUNC_CACHE_)
    def broadcast_to(
        self, shape: tuple[int, ...], subok: bool = False
    ) -> DiagonalCovariance:
        norm_index = self._get_norm_index(shape)

        matrix_shape = (
            (norm_index[-1].squeeze(),)
            if isinstance(norm_index[-1], np.ndarray)
            else norm_index[-1:]
        )

        if matrix_shape != self.matrix_shape[:-1]:
            raise ValueError(
                f"Cannot broadcast the matrix shape of DiagonalCovariance: {self.matrix_shape} to the new matrix shape: ({matrix_shape, matrix_shape})"
            )

        batch_shape = norm_index[:-2]

        return DiagonalCovariance(
            np.broadcast_to(self._D, batch_shape + matrix_shape, subok=subok)
        )

    def inverse(self) -> FloatArray:
        identity = np.broadcast_to(np.eye(*self.matrix_shape), self.shape)
        return solve_diagonal_covariance(self, identity)


def solve_cholesky_covariance(
    P: CholeskyFactorCovariance | CovarianceBase,
    B: FloatArray,
    overwrite_b: bool = False,
) -> FloatArray:
    """Solve the linear system PX = B.

    Where P is a cholesky factor representation of the covariance matrix P.

    Args:
        P: A covariance object.
        B: The result of the linear transformation

    Returns:
        X: The variable.
    """

    result: FloatArray = cho_solve(
        (P.cholesky_factor, True),
        B,
        check_finite=CHOLESKY_CHECK_FINITE_,
        overwrite_b=overwrite_b,
    )
    return result


def solve_diagonal_covariance(P_: DiagonalCovariance, B_: FloatArray) -> FloatArray:
    """Solve the linear system PX = B.

    Where P is a diagonal matrix.

    Args:
        P: A covariance object.
        B: The result of the linear transformation

    Returns:
        X: The variable.
    """
    P, B = left_broadcast_arrays(P_.variance, B_)
    result: FloatArray = B / P
    return result


def linear_cross_covariance(
    P_: CholeskyFactorCovariance | DiagonalCovariance, A_: FloatArray
) -> FloatArray:
    """Compute the cross covariane between a variable and a linear transformation of that variable.

    Let Cov(x) = P and z = Ax. Then Cov(x,z) = Cov(x,Ax) = P_ @ A.T

    Args:
        P_: The covariance.
        A_: The linear transform.

    Returns:
        FloatArray: The cross covariance.
    """

    if isinstance(P_, CholeskyFactorCovariance):
        P, A = left_broadcast_arrays(P_.full(), A_)
        result: FloatArray = np.einsum("...ij,...kj->...ik", P, A)
        return result
    else:
        D, A = left_broadcast_arrays(P_.variance, A_)
        result2: FloatArray = D * A.swapaxes(-1, -2)
        return result2
