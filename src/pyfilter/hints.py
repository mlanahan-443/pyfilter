from types import EllipsisType

import numpy as np
from numpy.typing import NDArray

from pyfilter.config import FDTYPE_

type FloatArray = NDArray[FDTYPE_]
type BoolArray = NDArray[np.bool_]
type VoidArray = NDArray[np.void]
type IntArr = NDArray[np.integer]


type IndexItem = (
    int  # Standard integer
    | np.integer  # NumPy integer types (int8, int16, etc.)
    | slice  # A standard slice, e.g., :5
    | EllipsisType  # The ... object
    | None  # For new axes, e.g., np.newaxis
    | list[int]  # List for fancy indexing
    | list[bool]  # List for boolean masking
    | IntArr  # Array for fancy indexing
    | BoolArray  # Array for boolean masking
)

# 2. The final ArrayIndex is either one of those items
#    OR a tuple containing any number of those items.
type ArrayIndex = IndexItem | tuple[IndexItem, ...]
