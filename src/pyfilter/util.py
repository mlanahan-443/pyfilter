from collections.abc import Callable
from dataclasses import dataclass
from types import EllipsisType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyfilter.hints import ArrayIndex

# Define a full slice object
full_slice = slice(None, None, None)  # Represents the ':'


def normalize_index(array_index: Any, ndim: int) -> tuple[Any, ...]:
    """
    Expands the Ellipsis ('...') in an array index into a series of full slices.

    Args:
        array_index: The user-provided index (ArrayIndex).
        ndim: The number of dimensions of the array being indexed (arr.ndim).

    Returns:
        A tuple representing the normalized index, or raises an IndexError.
    """

    # 1. Ensure it's a tuple
    if not isinstance(array_index, tuple):
        array_index = (array_index,)

    # Check for too many Ellipses (NumPy only allows one)
    if (
        isinstance(array_index, (tuple, list))
        and sum([0 if isinstance(item, EllipsisType) else 1 for item in array_index])
        > 1
    ):
        raise IndexError("an index can only have a single Ellipsis (...)")

    # 2. Count the Non-Ellipsis items that consume a dimension
    # (Excludes Ellipsis and np.newaxis/None)

    # Count of components that *consume* an existing dimension
    n_explicit_consumers = sum(
        1 for item in array_index if item is not Ellipsis and item is not None
    )

    # 3. Determine the required expansion length
    n_ellipsis = ndim - n_explicit_consumers

    if n_ellipsis < 0:
        raise IndexError(
            f"too many indices for array: array is {ndim}-dimensional, "
            f"but {n_explicit_consumers} explicit indices were provided "
            f"(excluding '...') which is > {ndim} dimensions"
        )

    # 4. Create the final normalized index
    normalized = []
    for item in array_index:
        if item is Ellipsis:
            # Replace '...' with the calculated number of full slices
            normalized.extend([full_slice] * n_ellipsis)
        else:
            normalized.append(item)

    return tuple(normalized)


def left_broadcast_arrays(*args: ArrayLike) -> list[np.ndarray[Any, Any]]:
    """
    Broadcasts any number of arrays against each other using left-aligned logic
    (batch dimensions first), rather than NumPy's standard right-aligned logic.

    Args:
        *args: Variable length argument list of array-likes.

    Returns:
        list: A list of broadcasted arrays, all having the same shape.

    Raises:
        ValueError: If the arrays cannot be broadcast even after left-alignment.
    """
    arrays = [np.asanyarray(x) for x in args]

    if not arrays:
        return []

    # Find the maximum dimensionality among all inputs
    max_ndim = max(arr.ndim for arr in arrays)

    # Pad shorter arrays with trailing 1s (on the right)
    aligned_arrays = []
    for arr in arrays:
        if arr.ndim < max_ndim:
            # Calculate how many dimensions are missing
            missing_dims = max_ndim - arr.ndim

            # Create the new shape tuple: current shape + (1, 1, ...)
            new_shape = arr.shape + (1,) * missing_dims

            # Reshape the view (zero copy)
            aligned_arrays.append(arr.reshape(new_shape))
        else:
            aligned_arrays.append(arr)

    # Use standard numpy broadcasting on the now-aligned arrays
    return list(np.broadcast_arrays(*aligned_arrays))


type IndexingFunction[T] = Callable[[ArrayIndex], T]


@dataclass
class _IndexerMethod[T]:
    """
    Generic helper to expose a method via [] syntax.
    """

    # Instead of storing 'obj' and a 'string', we store the bound method directly.
    _func: IndexingFunction[T]

    def __getitem__(self, index: ArrayIndex) -> T:
        # Direct invocation - faster and type-safe
        return self._func(index)


def implements_ufunc(
    np_function: Callable[..., Any],
    handler_cache: dict[Callable[..., Any], Callable[..., Any]],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Utility function for indexing handled NumPy Functions.

    Args:
        np_function: The number function.
        handler_cache: The dictionary to store the mapping from numpy function to user implementation.

    Returns:
        The wrapped function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        handler_cache[np_function] = func
        return func

    return decorator
