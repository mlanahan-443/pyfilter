"""Test utility functions."""

import numpy as np
from pyfilter.util import left_broadcast_arrays
import pytest


class TestLeftBroadcast:
    """Test left broadcasting of arrays."""

    def test_basic_extension(self):
        """Extending the right side."""
        # Shape (100, 10) vs (100, 10, 40)
        x = np.zeros((100, 10))
        y = np.zeros((100, 10, 40))

        bx, by = left_broadcast_arrays(x, y)

        np.testing.assert_equal(bx.shape, (100, 10, 40))
        np.testing.assert_equal(by.shape, (100, 10, 40))

    def test_deep_extension(self):
        """Test extending multiple dimensions deep."""
        # Shape (100, 10) vs (100, 10, 40, 20, 30)
        x = np.zeros((100, 10))
        y = np.zeros((100, 10, 40, 20, 30))

        bx, by = left_broadcast_arrays(x, y)

        np.testing.assert_equal(bx.shape, (100, 10, 40, 20, 30))
        np.testing.assert_equal(by.shape, (100, 10, 40, 20, 30))

    def test_interleaved_broadcasting(self):
        """
        Test that standard broadcasting rules still apply to the internal dimensions.
        Left alignment just fixes the ndim; standard broadcasting handles the 1s.
        """
        # X: (100, 1, 40)  <- Middle dim is 1
        # Y: (100, 10)     <- Will become (100, 10, 1)
        # Result should be (100, 10, 40)
        x = np.zeros((100, 1, 40))
        y = np.zeros((100, 10))

        bx, by = left_broadcast_arrays(x, y)

        np.testing.assert_equal(bx.shape, (100, 10, 40))
        np.testing.assert_equal(by.shape, (100, 10, 40))

    def test_scalar_broadcasting(self):
        """Scalars should work against any array."""
        x = np.array(5.0)  # Shape ()
        y = np.zeros((10, 20))

        bx, by = left_broadcast_arrays(x, y)

        np.testing.assert_equal(bx.shape, (10, 20))
        # Ensure values broadcasted correctly
        assert np.all(bx == 5.0)

    def test_incompatible_shapes(self):
        """Ensure it still raises errors for genuine mismatches."""
        # (100, 5) vs (100, 6) -> Should fail on dim 1
        x = np.zeros((100, 5))
        y = np.zeros((100, 6))

        with pytest.raises(ValueError):
            left_broadcast_arrays(x, y)

    def test_batch_mismatch(self):
        """Ensure left-most (batch) dimension mismatches still raise errors."""
        x = np.zeros((50, 10))
        y = np.zeros((100, 10, 5))  # padded x becomes (50, 10, 1)

        # 50 and 100 are incompatible
        with pytest.raises(ValueError):
            left_broadcast_arrays(x, y)
