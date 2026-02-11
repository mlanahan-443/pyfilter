import numpy as np
import pytest
from numpy.testing import assert_allclose

# --- Imports from your file ---
# (Assumes gaussian_rv.py is in the same directory)
from pyfilter.types.random_variables import GaussianRV, DTYPE, FloatArray
from pyfilter.types.covariance import (
    DiagonalCovariance,
    cholesky_factor,
)

# --- Fixtures for reusable test data ---


def _to_array(cov):
    """Convert covariance to array if it's a covariance object."""
    return cov if isinstance(cov, np.ndarray) else cov.full()


@pytest.fixture(
    params=["array_1d", "cholesky_1d", "diagonal_1d"],
    ids=["Array1D", "Cholesky1D", "Diagonal1D"],
)
def covariance_1d(request):
    """Parameterized 1D covariance: array, CholeskyFactorCovariance, DiagonalCovariance."""
    if request.param == "array_1d":
        return np.array([[4.0]], dtype=DTYPE)
    elif request.param == "cholesky_1d":
        return cholesky_factor(np.array([[4.0]], dtype=DTYPE))
    elif request.param == "diagonal_1d":
        return DiagonalCovariance(np.array([2.0], dtype=DTYPE))


@pytest.fixture(
    params=["array_2d", "cholesky_2d", "diagonal_2d"],
    ids=["Array2D", "Cholesky2D", "Diagonal2D"],
)
def covariance_2d(request):
    """Parameterized 2D covariance: array, CholeskyFactorCovariance, DiagonalCovariance."""
    if request.param == "array_2d":
        cov = np.array([[4.0, 1.0], [1.0, 9.0]], dtype=DTYPE)
    elif request.param == "cholesky_2d":
        cov = cholesky_factor(np.array([[4.0, 1.0], [1.0, 9.0]], dtype=DTYPE))
        print(f"Created cholesky_2d with trace: {cov.trace()}, full: {cov.full()}")
    elif request.param == "diagonal_2d":
        cov = DiagonalCovariance(np.array([2.0, 3.0], dtype=DTYPE))
    return cov


@pytest.fixture(
    params=["array_2d_other", "cholesky_2d_other", "diagonal_2d_other"],
    ids=["Array2DOther", "Cholesky2DOther", "Diagonal2DOther"],
)
def covariance_2d_other(request):
    """Parameterized 2D covariance for binary operations."""
    if request.param == "array_2d_other":
        return np.array([[1.0, 0.0], [0.0, 2.0]], dtype=DTYPE)
    elif request.param == "cholesky_2d_other":
        return cholesky_factor(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=DTYPE))
    elif request.param == "diagonal_2d_other":
        return DiagonalCovariance(np.array([1.0, np.sqrt(2.0)], dtype=DTYPE))


@pytest.fixture(
    params=["array_batched", "cholesky_batched", "diagonal_batched"],
    ids=["ArrayBatched", "CholeskyBatched", "DiagonalBatched"],
)
def covariance_batched(request):
    """Parameterized batched (2, 2, 2) covariance."""
    if request.param == "array_batched":
        return np.array(
            [[[4.0, 1.0], [1.0, 9.0]], [[1.0, 0.5], [0.5, 1.0]]], dtype=DTYPE
        )
    elif request.param == "cholesky_batched":
        return cholesky_factor(
            np.array([[[4.0, 1.0], [1.0, 9.0]], [[1.0, 0.5], [0.5, 1.0]]], dtype=DTYPE)
        )
    elif request.param == "diagonal_batched":
        return DiagonalCovariance(np.array([[2.0, 3.0], [1.0, 1.0]], dtype=DTYPE))


@pytest.fixture
def grv_1d(covariance_1d) -> GaussianRV:
    """A 1-dimensional GaussianRV."""
    mean = np.array([1.0], dtype=DTYPE)
    return GaussianRV(mean, covariance_1d)


@pytest.fixture
def grv_2d(covariance_2d) -> GaussianRV:
    """A 2-dimensional GaussianRV."""
    mean = np.array([1.0, 2.0], dtype=DTYPE)
    return GaussianRV(mean, covariance_2d)


@pytest.fixture
def grv_2d_other(covariance_2d_other) -> GaussianRV:
    """Another 2-dimensional GaussianRV for binary operations."""
    mean = np.array([-1.0, 0.5], dtype=DTYPE)
    return GaussianRV(mean, covariance_2d_other)


@pytest.fixture
def grv_batched(covariance_batched) -> GaussianRV:
    """A batched (batch_size=2) 2-dimensional GaussianRV."""
    mean = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)  # Shape (2, 2)
    return GaussianRV(mean, covariance_batched)


@pytest.fixture
def scalar_val() -> float:
    """A simple scalar value."""
    return 3.0


@pytest.fixture
def array_1d() -> FloatArray:
    """A 1D array compatible with grv_2d."""
    return np.array([3.0, 4.0], dtype=DTYPE)


@pytest.fixture
def array_2d_matrix() -> FloatArray:
    """A 2D matrix compatible with grv_2d."""
    return np.array([[1.0, 0.0], [0.5, 2.0]], dtype=DTYPE)


@pytest.fixture
def batched_array_1d() -> FloatArray:
    """A batched 1D array compatible with grv_batched."""
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)  # Shape (2, 2)


@pytest.fixture
def batched_matrix() -> FloatArray:
    """A batched 2D matrix compatible with grv_batched."""
    return np.array(
        [[[1.0, 0.0], [0.5, 2.0]], [[0.0, 1.0], [1.0, 0.0]]], dtype=DTYPE
    )  # Shape (2, 2, 2)


# --- Test Classes ---


class TestGaussianRVInitialization:
    """Tests for __init__ and __post_init__ validation."""

    def test_successful_creation(self, grv_2d):
        assert isinstance(grv_2d, GaussianRV)
        assert grv_2d.mean.shape == (2,)
        assert grv_2d.covariance.shape == (2, 2)

    def test_successful_batched_creation(self, grv_batched):
        assert isinstance(grv_batched, GaussianRV)
        assert grv_batched.mean.shape == (2, 2)
        assert grv_batched.covariance.shape == (2, 2, 2)

    def test_fail_mean_ndim_0(self):
        with pytest.raises(ValueError, match="Mean must have at least 1 dimension"):
            GaussianRV(np.array(1.0), np.array([[1.0]]))

    def test_fail_cov_ndim_1(self):
        with pytest.raises(
            ValueError, match="Covariance must have at least 2 dimensions"
        ):
            GaussianRV(np.array([1.0]), np.array([1.0]))

    def test_fail_cov_not_square(self):
        with pytest.raises(
            ValueError, match="Last two dimensions of covariance must be square"
        ):
            GaussianRV(
                np.array([1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            )

    def test_fail_mean_cov_dim_mismatch(self):
        with pytest.raises(ValueError, match="Last dimension of mean .* must match"):
            GaussianRV(np.array([1.0, 2.0]), np.array([[1.0]]))  # mean dim 2, cov dim 1

    def test_fail_batch_dim_mismatch(self):
        mean = np.random.rand(3, 2)  # Batch (3,)
        cov = np.random.rand(4, 2, 2)  # Batch (4,)
        with pytest.raises(
            ValueError, match="Batch dimensions of mean .* and covariance .* must match"
        ):
            GaussianRV(mean, cov)

    def test_broadcasting_batch_dims(self):
        """Test successful creation when batch dims broadcast."""
        mean = np.random.rand(1, 3, 2)  # Batch (1, 3)
        cov = np.random.rand(5, 1, 2, 2)  # Batch (5, 1)
        # This should fail, as __post_init__ does not broadcast
        with pytest.raises(
            ValueError, match="Batch dimensions of mean .* and covariance .* must match"
        ):
            GaussianRV(mean, cov)


class TestGaussianRVProperties:
    """Tests for shape, len, and repr."""

    def test_shape(self, grv_2d, grv_batched):
        assert grv_2d.shape == (2,)
        assert grv_batched.shape == (2, 2)

    def test_len(self, grv_1d, grv_2d, grv_batched):
        assert len(grv_1d) == 1
        assert len(grv_2d) == 2
        assert len(grv_batched) == 2  # len() is last dim

    def test_repr(self, grv_1d):
        rep = repr(grv_1d)
        assert "GaussianRV" in rep
        assert "shape=(1,)" in rep
        assert "mean_norm=1.000" in rep
        assert "cov_trace=4.0" in rep


class TestGaussianRVCompatibility:
    """Tests for _check_compatible helper."""

    def test_check_compatible_grv_fail(self, grv_1d, grv_2d):
        with pytest.raises(ValueError, match="Incompatible shapes"):
            grv_1d._check_compatible(grv_2d)

    def test_check_compatible_array_fail(self, grv_2d):
        array = np.random.rand(5, 3)  # Not broadcastable to (2,)
        with pytest.raises(ValueError, match="Cannot broadcast shapes"):
            grv_2d._check_compatible(array)

    def test_check_compatible_array_success(self, grv_2d):
        array = np.random.rand(1, 2)  # Broadcastable to (2,)
        grv_2d._check_compatible(array)  # Should not raise

    def test_check_compatible_scalar_success(self, grv_2d):
        grv_2d._check_compatible(5.0)  # Should not raise


class TestGaussianRVAddition:
    """Tests for __add__, __radd__, __iadd__."""

    def test_add_grv(self, grv_2d, grv_2d_other):
        print("\n=== Test starting ===")
        print(f"grv_2d.covariance type: {type(grv_2d.covariance)}")
        print(
            f"grv_2d.covariance trace: {grv_2d.covariance.trace() if hasattr(grv_2d.covariance, 'trace') else np.trace(grv_2d.covariance)}"
        )

        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)
        m2, c2 = grv_2d_other.mean, _to_array(grv_2d_other.covariance)

        print("c1 from _to_array:", c1)
        print("c2 from _to_array:", c2)
        print("grv_2d_other.covariance:", grv_2d_other.covariance)
        if hasattr(grv_2d_other.covariance, "_D"):
            print("grv_2d_other.covariance._D:", grv_2d_other.covariance._D)
            print("grv_2d_other.covariance.variance:", grv_2d_other.covariance.variance)
        res = grv_2d + grv_2d_other

        print("After addition:")
        print("res.covariance:", _to_array(res.covariance))
        print("c1 NOW:", c1)
        print("c2 NOW:", c2)
        print("c1 + c2 NOW:", c1 + c2)

        assert_allclose(res.mean, m1 + m2)
        assert_allclose(_to_array(res.covariance), c1 + c2)  # Covariances add
        assert isinstance(res, GaussianRV)

    def test_add_scalar(self, grv_2d, scalar_val):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = grv_2d + scalar_val

        assert_allclose(res.mean, m1 + scalar_val)
        assert_allclose(_to_array(res.covariance), c1)  # Covariance is unchanged
        assert isinstance(res, GaussianRV)

    def test_add_array(self, grv_2d, array_1d):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = grv_2d + array_1d

        assert_allclose(res.mean, m1 + array_1d)
        assert_allclose(_to_array(res.covariance), c1)
        assert isinstance(res, GaussianRV)

    def test_radd_scalar(self, grv_2d, scalar_val):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = scalar_val + grv_2d

        assert_allclose(res.mean, scalar_val + m1)
        assert_allclose(_to_array(res.covariance), c1)
        assert isinstance(res, GaussianRV)

    def test_radd_array(self, grv_2d, array_1d):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = array_1d + grv_2d

        assert_allclose(res.mean, array_1d + m1)
        assert_allclose(_to_array(res.covariance), c1)
        assert isinstance(res, GaussianRV)

    def test_iadd_grv(self, grv_2d, grv_2d_other, covariance_2d, covariance_2d_other):
        # Skip test if using covariance objects (no in-place operations supported)
        if not isinstance(covariance_2d, np.ndarray) or not isinstance(
            covariance_2d_other, np.ndarray
        ):
            pytest.skip("In-place operations not supported for covariance objects")

        # Must copy fixture as it will be modified in-place
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)
        m2, c2 = grv_2d_other.mean, _to_array(grv_2d_other.covariance)
        grv_copy = GaussianRV(grv_2d.mean.copy(), grv_2d.covariance.copy())

        id_before = id(grv_copy)
        grv_copy += grv_2d_other
        assert id(grv_copy) == id_before
        assert_allclose(grv_copy.mean, m1 + m2)
        assert_allclose(_to_array(grv_copy.covariance), c1 + c2)

    def test_iadd_scalar(self, grv_2d, scalar_val, covariance_2d):
        # Skip test if using covariance objects (no in-place operations supported)
        if not isinstance(covariance_2d, np.ndarray):
            pytest.skip("In-place operations not supported for covariance objects")

        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)
        grv_copy = GaussianRV(grv_2d.mean.copy(), grv_2d.covariance.copy())

        id_before = id(grv_copy)
        grv_copy += scalar_val

        assert id(grv_copy) == id_before
        assert_allclose(grv_copy.mean, m1 + scalar_val)
        assert_allclose(_to_array(grv_copy.covariance), c1)  # Unchanged


class TestGaussianRVSubtraction:
    """Tests for __sub__, __rsub__, __isub__."""

    def test_sub_grv(self, grv_2d, grv_2d_other):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)
        m2, c2 = grv_2d_other.mean, _to_array(grv_2d_other.covariance)

        res = grv_2d - grv_2d_other

        assert_allclose(res.mean, m1 - m2)
        assert_allclose(
            _to_array(res.covariance), c1 + c2
        )  # Covariances add for subtraction!
        assert isinstance(res, GaussianRV)

    def test_sub_scalar(self, grv_2d, scalar_val):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = grv_2d - scalar_val

        assert_allclose(res.mean, m1 - scalar_val)
        assert_allclose(_to_array(res.covariance), c1)
        assert isinstance(res, GaussianRV)

    def test_sub_array(self, grv_2d, array_1d):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = grv_2d - array_1d

        assert_allclose(res.mean, m1 - array_1d)
        assert_allclose(_to_array(res.covariance), c1)
        assert isinstance(res, GaussianRV)

    def test_rsub_scalar(self, grv_2d, scalar_val):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = scalar_val - grv_2d

        assert_allclose(res.mean, scalar_val - m1)
        assert_allclose(_to_array(res.covariance), c1)
        assert isinstance(res, GaussianRV)

    def test_rsub_array(self, grv_2d, array_1d):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = array_1d - grv_2d

        assert_allclose(res.mean, array_1d - m1)
        assert_allclose(_to_array(res.covariance), c1)
        assert isinstance(res, GaussianRV)

    def test_isub_grv(self, grv_2d, grv_2d_other, covariance_2d, covariance_2d_other):
        # Skip test if using covariance objects (no in-place operations supported)
        if not isinstance(covariance_2d, np.ndarray) or not isinstance(
            covariance_2d_other, np.ndarray
        ):
            pytest.skip("In-place operations not supported for covariance objects")

        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)
        m2, c2 = grv_2d_other.mean, _to_array(grv_2d_other.covariance)
        grv_copy = GaussianRV(grv_2d.mean.copy(), grv_2d.covariance.copy())

        id_before = id(grv_copy)
        grv_copy -= grv_2d_other

        assert id(grv_copy) == id_before
        assert_allclose(grv_copy.mean, m1 - m2)
        assert_allclose(_to_array(grv_copy.covariance), c1 + c2)

    def test_isub_scalar(self, grv_2d, scalar_val, covariance_2d):
        # Skip test if using covariance objects (no in-place operations supported)
        if not isinstance(covariance_2d, np.ndarray):
            pytest.skip("In-place operations not supported for covariance objects")

        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)
        grv_copy = GaussianRV(grv_2d.mean.copy(), grv_2d.covariance.copy())

        id_before = id(grv_copy)
        grv_copy -= scalar_val

        assert id(grv_copy) == id_before
        assert_allclose(grv_copy.mean, m1 - scalar_val)
        assert_allclose(_to_array(grv_copy.covariance), c1)


class TestGaussianRVMultiplication:
    """Tests for __mul__, __rmul__, __imul__ (non-matmul)."""

    def test_mul_scalar(self, grv_2d, scalar_val):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = grv_2d * scalar_val

        assert_allclose(res.mean, m1 * scalar_val)
        assert_allclose(_to_array(res.covariance), c1 * (scalar_val**2))

    def test_rmul_scalar(self, grv_2d, scalar_val):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        res = scalar_val * grv_2d

        assert_allclose(res.mean, m1 * scalar_val)
        assert_allclose(_to_array(res.covariance), c1 * (scalar_val**2))

    def test_mul_1d_array(self, grv_2d, array_1d):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)

        if isinstance(grv_2d.covariance, np.ndarray):
            res = grv_2d * array_1d

            expected_mean = m1 * array_1d
            # Create (n, 1) * (1, n) scaling matrix
            cov_scale = array_1d[:, np.newaxis] * array_1d[np.newaxis, :]
            expected_cov = c1 * cov_scale

            assert_allclose(res.mean, expected_mean)
            assert_allclose(_to_array(res.covariance), expected_cov)
        else:
            with pytest.raises(TypeError):
                res = grv_2d * array_1d

    def test_rmul_1d_array(self, grv_2d, array_1d):
        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)
        if isinstance(grv_2d.covariance, np.ndarray):
            res = array_1d * grv_2d

            expected_mean = m1 * array_1d
            cov_scale = array_1d[:, np.newaxis] * array_1d[np.newaxis, :]
            expected_cov = c1 * cov_scale

            assert_allclose(res.mean, expected_mean)
            assert_allclose(_to_array(res.covariance), expected_cov)

        else:
            with pytest.raises(TypeError):
                res = grv_2d * array_1d

    def test_imul_scalar(self, grv_2d, scalar_val, covariance_2d):
        # Skip test if using covariance objects (no in-place operations supported)
        if not isinstance(covariance_2d, np.ndarray):
            pytest.skip("In-place operations not supported for covariance objects")

        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)
        grv_copy = GaussianRV(grv_2d.mean.copy(), grv_2d.covariance.copy())
        id_before = id(grv_copy)

        grv_copy *= scalar_val

        assert id(grv_copy) == id_before
        assert_allclose(grv_copy.mean, m1 * scalar_val)
        assert_allclose(_to_array(grv_copy.covariance), c1 * (scalar_val**2))

    def test_imul_1d_array(self, grv_2d, array_1d, covariance_2d):
        # Skip test if using covariance objects (no in-place operations supported)
        if not isinstance(covariance_2d, np.ndarray):
            pytest.skip("In-place operations not supported for covariance objects")

        m1, c1 = grv_2d.mean, _to_array(grv_2d.covariance)
        grv_copy = GaussianRV(grv_2d.mean.copy(), grv_2d.covariance.copy())
        id_before = id(grv_copy)

        grv_copy *= array_1d

        expected_mean = m1 * array_1d
        cov_scale = array_1d[:, np.newaxis] * array_1d[np.newaxis, :]
        expected_cov = c1 * cov_scale

        assert id(grv_copy) == id_before
        assert_allclose(grv_copy.mean, expected_mean)
        assert_allclose(_to_array(grv_copy.covariance), expected_cov)

    def test_imul_matrix_fail(self, grv_2d, array_2d_matrix):
        with pytest.raises(ValueError, match="In-place multiplication only supported"):
            grv_2d *= array_2d_matrix


class TestGaussianRVMatMul:
    """Tests for __matmul__, __rmatmul__, __mul__ (matrix), and __array_ufunc__."""

    def test_rmatmul(self, grv_2d, array_2d_matrix):
        """Tests A @ grv (Y = AX)"""
        m, c = grv_2d.mean, _to_array(grv_2d.covariance)
        A = array_2d_matrix

        expected_mean = A @ m
        expected_cov = A @ c @ A.T

        res = A @ grv_2d

        assert_allclose(res.mean, expected_mean)
        assert_allclose(_to_array(res.covariance), expected_cov)
        assert res.shape == (2,)
        assert len(res) == 2

    def test_mul_matrix(self, grv_2d, array_2d_matrix):
        """Tests grv * A (which is implemented as Y = AX)"""
        m, c = grv_2d.mean, _to_array(grv_2d.covariance)
        A = array_2d_matrix

        expected_mean = A @ m
        expected_cov = A @ c @ A.T

        res = grv_2d * A

        assert_allclose(res.mean, expected_mean)
        assert_allclose(_to_array(res.covariance), expected_cov)

    def test_rmul_matrix(self, grv_2d, array_2d_matrix):
        """Tests A * grv (which is implemented as Y = AX)"""
        m, c = grv_2d.mean, _to_array(grv_2d.covariance)
        A = array_2d_matrix

        expected_mean = A @ m
        expected_cov = A @ c @ A.T

        res = A * grv_2d

        assert_allclose(res.mean, expected_mean)
        assert_allclose(_to_array(res.covariance), expected_cov)

    def test_matmul(self, grv_2d, array_2d_matrix):
        """Tests grv @ A (which is *also* implemented as Y = AX)"""
        # This tests the behavior as written, even if the
        # intention might have been Y = XA.
        m, c = grv_2d.mean, _to_array(grv_2d.covariance)
        A = array_2d_matrix

        expected_mean = A @ m
        expected_cov = A @ c @ A.T

        res = grv_2d @ A

        assert_allclose(res.mean, expected_mean)
        assert_allclose(_to_array(res.covariance), expected_cov)

    def test_array_ufunc_matmul(self, grv_2d, array_2d_matrix):
        """Tests np.matmul(A, grv) and np.matmul(grv, A)"""
        m, c = grv_2d.mean, _to_array(grv_2d.covariance)
        A = array_2d_matrix

        expected_mean = A @ m
        expected_cov = A @ c @ A.T

        # Test: A @ grv
        res1 = np.matmul(A, grv_2d)
        assert_allclose(res1.mean, expected_mean)
        assert_allclose(_to_array(res1.covariance), expected_cov)

        # Test: grv @ A (which also does A @ grv)
        res2 = np.matmul(grv_2d, A)
        assert_allclose(res2.mean, expected_mean)
        assert_allclose(_to_array(res2.covariance), expected_cov)

    def test_matmul_fail_1d_array(self, grv_2d, array_1d):
        with pytest.raises(ValueError, match="@ operator requires a matrix"):
            grv_2d @ array_1d

    def test_matmul_fail_dim_mismatch(self, grv_1d, array_2d_matrix):
        # grv_1d has dim 1, array_2d_matrix has dim 2 (..., m, n) -> n=2
        with pytest.raises(ValueError, match="Matrix dimension mismatch"):
            array_2d_matrix @ grv_1d

    def test_matmul_batched(self, grv_batched, batched_matrix):
        """Tests batched A @ grv (Y = AX)"""
        m, c = (
            grv_batched.mean,
            _to_array(grv_batched.covariance),
        )  # (2, 2) and (2, 2, 2)
        A = batched_matrix  # (2, 2, 2)

        # Manually compute expected batched matmul
        expected_mean = np.einsum("...ij,...j->...i", A, m)
        temp = np.einsum("...ij,...jk->...ik", A, c)
        expected_cov = np.einsum("...ij,...kj->...ik", temp, A)

        res = A @ grv_batched

        assert res.shape == (2, 2)
        assert_allclose(res.mean, expected_mean)
        assert_allclose(_to_array(res.covariance), expected_cov)


class TestGaussianRVMethods:
    """Tests marginal, joint, conditional, and cross."""

    def test_marginal(self, grv_batched):
        m, c = grv_batched.mean, _to_array(grv_batched.covariance)
        indices = [1]

        marginal_grv = grv_batched.marginal(indices)

        expected_mean = m[..., indices]
        # Using np.ix_ is robust for indexing
        idx_grid = np.ix_(indices, indices)
        expected_cov = c[..., idx_grid[0], idx_grid[1]]

        assert marginal_grv.shape == (2, 1)
        assert_allclose(marginal_grv.mean, expected_mean)
        assert_allclose(_to_array(marginal_grv.covariance), expected_cov)

    def test_marginal_multiple_indices(self, grv_batched):
        # grv_batched is (2, 2) and (2, 2, 2)
        # Let's use a 3D grv
        m3 = np.array([1, 2, 3])
        c3 = np.array([[4, 1, 0.5], [1, 5, 0.2], [0.5, 0.2, 6]])
        grv_3d = GaussianRV(m3, c3)

        indices = [0, 2]
        marginal_grv = grv_3d.marginal(indices)

        expected_mean = m3[indices]  # [1, 3]
        idx_grid = np.ix_(indices, indices)
        expected_cov = c3[idx_grid]  # [[4, 0.5], [0.5, 6]]

        assert marginal_grv.shape == (2,)
        assert_allclose(marginal_grv.mean, expected_mean)
        assert_allclose(marginal_grv.covariance, expected_cov)

    def test_joint(self, grv_2d, grv_1d):
        n1, n2 = len(grv_2d), len(grv_1d)
        cross_cov = np.array([[0.5], [0.2]])  # Shape (n1, n2) = (2, 1)

        m1, c11 = grv_2d.mean, _to_array(grv_2d.covariance)
        m2, c22 = grv_1d.mean, _to_array(grv_1d.covariance)
        c12 = cross_cov
        c21 = cross_cov.T

        joint_grv = grv_2d.joint(grv_1d, cross_cov)

        expected_mean = np.concatenate([m1, m2])
        expected_cov = np.block([[c11, c12], [c21, c22]])

        assert len(joint_grv) == n1 + n2
        assert_allclose(joint_grv.mean, expected_mean)
        assert_allclose(_to_array(joint_grv.covariance), expected_cov)

    def test_joint_fail_cross_cov_shape(self, grv_2d, grv_1d):
        cross_cov = np.random.rand(2, 2)  # Wrong shape (should be 2, 1)
        with pytest.raises(ValueError, match="Cross-covariance shape"):
            grv_2d.joint(grv_1d, cross_cov)

    def test_cross(self, grv_2d, array_2d_matrix):
        """Tests Cov(X, AX) = C @ A.T"""
        c = _to_array(grv_2d.covariance)
        A = array_2d_matrix

        cross_res = grv_2d.linear_cross(A)
        expected_cross = c @ A.T

        assert cross_res.shape == (2, 2)
        assert_allclose(cross_res, expected_cross)

    def test_cross_batched(self, grv_batched, batched_matrix):
        c = _to_array(grv_batched.covariance)  # (2, 2, 2)
        A = batched_matrix  # (2, 2, 2)

        cross_res = grv_batched.linear_cross(A)

        # C @ A.T
        expected_cross = np.einsum("...ij,...kj->...ik", c, A)

        assert cross_res.shape == (2, 2, 2)
        assert_allclose(cross_res, expected_cross)

    def test_cross_fail_dim_mismatch(self, grv_1d, array_2d_matrix):
        # A.shape[-1] (2) != len(grv) (1)
        with pytest.raises(ValueError, match="Matrix A column dimension"):
            grv_1d.linear_cross(array_2d_matrix)

    def test_conditional_on_mean(self, grv_2d, grv_1d):
        """Test X1 | X2 = mu_2"""
        c12 = np.array([[0.5], [0.2]])  # (2, 1)

        m1, _m2 = grv_2d.mean, grv_1d.mean
        c11 = (
            grv_2d.covariance
            if isinstance(grv_2d.covariance, np.ndarray)
            else grv_2d.covariance.full()
        )
        c22 = (
            grv_1d.covariance
            if isinstance(grv_1d.covariance, np.ndarray)
            else grv_1d.covariance.full()
        )
        c21 = c12.T

        # mu_cond = mu1 + C12 @ C22_inv @ (mu2 - mu2) = mu1
        expected_mean = m1

        # C_cond = C11 - C12 @ C22_inv @ C21
        c22_inv = np.linalg.inv(c22)
        expected_cov = c11 - c12 @ c22_inv @ c21

        cond_grv = grv_2d.conditional(grv_1d, c12, given_value=None)

        assert cond_grv.shape == grv_2d.shape
        assert_allclose(cond_grv.mean, expected_mean)
        check_cov = (
            cond_grv.covariance
            if isinstance(cond_grv.covariance, np.ndarray)
            else cond_grv.covariance.full()
        )
        assert_allclose(check_cov, expected_cov)

    def test_conditional_on_value(self, grv_2d, grv_1d):
        """Test X1 | X2 = x2"""
        c12 = np.array([[0.5], [0.2]])  # (2, 1)
        x2_val = np.array([5.0])  # Value to condition on

        m1, m2 = grv_2d.mean, grv_1d.mean
        c11 = _to_array(grv_2d.covariance)
        c22 = _to_array(grv_1d.covariance)
        c21 = c12.T

        residual = x2_val - m2
        c22_inv = np.linalg.inv(c22)

        # mu_cond = mu1 + C12 @ C22_inv @ (x2 - mu2)
        expected_mean = m1 + (c12 @ c22_inv @ residual.T).T.squeeze()

        # C_cond = C11 - C12 @ C22_inv @ C21
        expected_cov = c11 - c12 @ c22_inv @ c21

        cond_grv = grv_2d.conditional(grv_1d, c12, given_value=x2_val)

        assert cond_grv.shape == grv_2d.shape
        assert_allclose(cond_grv.mean, expected_mean)
        assert_allclose(_to_array(cond_grv.covariance), expected_cov)

    def test_conditional_batched(self, grv_batched):
        """Test batched conditional X1 | X2 = x2"""
        # Let's split the batched grv into two 1D marginals
        grv1 = grv_batched.marginal([0])  # (2, 1)
        grv2 = grv_batched.marginal([1])  # (2, 1)

        # Cross-covariance is C[..., [0], [1]]
        c_batched = _to_array(grv_batched.covariance)
        c12 = c_batched[..., 0:1, 1:]  # (2, 1, 1)
        m1, c11 = grv1.mean, _to_array(grv1.covariance)
        m2, c22 = grv2.mean, _to_array(grv2.covariance)
        c21 = np.swapaxes(c12, -1, -2)

        x2_val = np.array([[5.0], [-1.0]])  # (2, 1)

        residual = (x2_val - m2)[..., None]  # (2, 1, 1)
        c22_inv_res = np.linalg.solve(c22, residual)  # (2, 1, 1)

        # mu_cond = mu1 + (C12 @ C22_inv @ residual)
        mean_update = (c12 @ c22_inv_res).squeeze(-1)  # (2, 1)
        expected_mean = m1 + mean_update

        # C_cond = C11 - C12 @ C22_inv @ C21
        c22_inv = np.linalg.inv(c22)  # (2, 1, 1)
        cov_update = c12 @ c22_inv @ c21  # (2, 1, 1)
        expected_cov = c11 - cov_update

        cond_grv = grv1.conditional(grv2, c12, given_value=x2_val)

        assert cond_grv.shape == (2, 1)
        assert_allclose(cond_grv.mean, expected_mean)
        assert_allclose(_to_array(cond_grv.covariance), expected_cov)
