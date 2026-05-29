from types import EllipsisType

from jax import numpy as jnp
from numpy.typing import NDArray

type FloatArray = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type VoidArray = NDArray[np.void]
type JaxIntArray = NDArray[np.integer]


type IndexItem = (
    int  # Standard integer
    | jnp.integer  # NumPy integer types (int8, int16, etc.)
    | slice  # A standard slice, e.g., :5
    | EllipsisType  # The ... object
    | None  # For new axes, e.g., jnp.newaxis
    | list[int]  # List for fancy indexing
    | list[bool]  # List for boolean masking
    | JaxIntArray  # Array for fancy indexing
    | BoolArray  # Array for boolean masking
)

# 2. The final ArrayIndex is either one of those items
#    OR a tuple containing any number of those items.
type ArrayIndex = IndexItem | tuple[IndexItem, ...]
