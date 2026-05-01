import numpy as np

DEBUG_ = True  # general debug flag.
CHOLESKY_CHECK_FINITE_ = (
    True  # check if the matrix is finite on cholesky decomposition.
)
FDTYPE_ = np.float64  # The datatype to use.
SYM_ATOL_ = 1e-6  # The absolute tolerance to use for symmetry checks.
SYM_RTOL_ = 1e-8  # The relative tolerance to use for symmetr
MONITOR_PERFORMANCE_ = True  # monitor performance
