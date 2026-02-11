from kalman_python.config import DTYPE_
from numpy.typing import NDArray
from types import EllipsisType
import numpy as np
from typing import Any

type FloatArray = NDArray[DTYPE_] | NDArray[np.floating[Any]]

type IndexItem = (
    int  # Standard integer
    | np.integer  # NumPy integer types (int8, int16, etc.)
    | slice  # A standard slice, e.g., :5
    | EllipsisType  # The ... object
    | None  # For new axes, e.g., np.newaxis
    | list[int]  # List for fancy indexing
    | list[bool]  # List for boolean masking
    | NDArray[np.integer]  # Array for fancy indexing
    | NDArray[np.bool_]  # Array for boolean masking
)

# 2. The final ArrayIndex is either one of those items
#    OR a tuple containing any number of those items.
type ArrayIndex = IndexItem | tuple[IndexItem, ...]
