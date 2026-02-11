import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import cho_factor
import re

# Import the classes from the file
from pyfilter.types.covariance import (
    CholeskyFactorCovariance,
    DiagonalCovariance,
    cholesky_factor,
    type_error_msg,
    linear_cross_covariance,
)

# --- Fixtures ---


@pytest.fixture
def dim() -> int:
    """Fixed matrix size (e.g. 4x4) for all tests."""
    return 4


@pytest.fixture(params=[2, 3, 4])
def ndim(request) -> int:
    """
    Parameterizes the rank of the array.
    2 = Single Matrix (N, N)
    3 = Batch of Matrices (B, N, N)
    4 = Batch of Batches (B1, B2, N, N)
    """
    return request.param


@pytest.fixture
def batch_shape(ndim: int) -> tuple[int, ...]:
    """
    Derives the batch shape from the total ndim.
    """
    # Define some arbitrary batch sizes for testing
    # Using different sizes (5, 3) helps catch broadcasting bugs
    full_batch_sizes = (5, 3, 2)

    # Return the slice corresponding to the extra dimensions
    # ndim=2 -> (), ndim=3 -> (5,), ndim=4 -> (5, 3)
    return full_batch_sizes[: ndim - 2]


@pytest.fixture
def P_full(dim: int, batch_shape: tuple[int, ...]) -> np.ndarray:
    """Returns random, positive-definite matrices with correct batch shape."""
    # Shape becomes (*batch_shape, dim, dim)
    full_shape = batch_shape + (dim, dim)

    A = np.random.rand(*full_shape)

    # Make positive definite: A @ A.T + I
    # We use swapaxes to transpose only the last two dimensions for the batch
    A_T = np.swapaxes(A, -1, -2)
    P = A @ A_T + dim * np.eye(dim)
    return P


@pytest.fixture
def L_factor(P_full: np.ndarray) -> np.ndarray:
    """Returns the true lower-triangular Cholesky factor of P."""
    # NOTE: We use np.linalg.cholesky here because it supports
    # batch dimensions natively, whereas scipy.linalg.cho_factor does not.
    return np.linalg.cholesky(P_full)


@pytest.fixture
def chol_cov(L_factor: np.ndarray) -> CholeskyFactorCovariance:
    return CholeskyFactorCovariance(L_factor.copy())


@pytest.fixture
def diag_std(dim: int, batch_shape: tuple[int, ...]) -> np.ndarray:
    """Returns random standard deviations with correct batch shape."""
    # Shape becomes (*batch_shape, dim)
    return np.random.rand(*(batch_shape + (dim,))) + 0.5


@pytest.fixture
def diag_cov(diag_std: np.ndarray) -> DiagonalCovariance:
    return DiagonalCovariance(diag_std.copy())


@pytest.fixture
def A_matrix(dim: int, batch_shape: tuple[int, ...]) -> np.ndarray:
    """Returns a random transformation matrix A."""
    # We make A batched as well to test full batch-on-batch operations
    return np.random.rand(*(batch_shape + (dim, dim))) + 0.1


# --- Test cholesky_factor Helper ---


def test_cholesky_factor_helper(P_full: np.ndarray, L_factor: np.ndarray):
    """Tests the cholesky_factor wrapper function."""
    chol_cov_obj = cholesky_factor(P_full.copy())
    assert isinstance(chol_cov_obj, CholeskyFactorCovariance)
    # Test that it correctly finds the L factor
    assert_allclose(chol_cov_obj.cholesky_factor, L_factor)
    # Test that the resulting .full() is correct
    assert_allclose(chol_cov_obj.full(), P_full)


@pytest.mark.parametrize("cov_type", ["cholesky", "diagonal"])
def test_linear_cross_covariance(
    cov_type: str,
    P_full: np.ndarray,
    chol_cov: CholeskyFactorCovariance,
    diag_cov: DiagonalCovariance,
    A_matrix: np.ndarray,
):
    """Test that the cross covariance is computed as intended."""

    if cov_type == "cholesky":
        cov = chol_cov
        P = P_full
    else:  # diagonal
        cov = diag_cov
        P = diag_cov.full()

    cross = linear_cross_covariance(cov, A_matrix)
    cross_check = P @ A_matrix.swapaxes(-1, -2)

    np.testing.assert_allclose(
        cross,
        cross_check,
        err_msg=f"Cross covariance not computed as intended for {cov_type}.",
    )


# --- Test CholeskyFactorCovariance ---


class TestCovarianceMethods:
    def test_trace_cholesky_factor(
        self, chol_cov: CholeskyFactorCovariance, P_full: np.ndarray
    ):
        """Test that the trace is computed correctly for the cholesky factor covariance."""
        expected_trace = np.trace(P_full, axis1=-2, axis2=-1)
        np.testing.assert_allclose(
            expected_trace,
            chol_cov.trace(),
            err_msg="Trace in cholesky factor covariance not equal to trace in full matrix.",
        )

    def test_trace_diagonal(self, diag_cov: DiagonalCovariance, diag_std: np.ndarray):
        """Test that the trace is computed correctly for the diagonal covariance."""
        expected_trace = np.sum(diag_std**2, axis=-1)
        np.testing.assert_allclose(
            expected_trace,
            diag_cov.trace(),
            err_msg="Trace in diagonal factor covariance not equal to trace in full matrix.",
        )

    def test_slice_cholesky_factor(self, chol_cov: CholeskyFactorCovariance):
        """Test that indexing works as anticipated for the cholesky factor covariance."""

        partial_chol_cov = chol_cov[..., 0:2, 0:2]

        partial_full = partial_chol_cov.full()
        check_full = chol_cov.full()[..., 0:2, 0:2]
        np.testing.assert_allclose(
            check_full,
            partial_full,
            err_msg="Slicing of cholesky factor covariance did not result in expected matrix.",
        )

    def test_slice_diagonal(self, diag_cov: DiagonalCovariance):
        """Test that indexing a diagonal covariance works as intended."""

        partial_diag_cov = diag_cov[..., 0:2, 0:2]
        partial_full = partial_diag_cov.full()
        check_full = diag_cov.full()[..., 0:2, 0:2]
        np.testing.assert_allclose(
            check_full,
            partial_full,
            err_msg="Slicing of diagonal covariance did not result in expected matrix.",
        )

    def test_at_index_cholesky_factor(self, chol_cov: CholeskyFactorCovariance):
        rng = np.arange(0, chol_cov.matrix_shape[0], 2)
        index = np.ix_(rng, rng)
        partial_chol_cov = chol_cov.at[..., *index]

        partial_full = partial_chol_cov.full()
        check_full = chol_cov.full()[..., *index]
        np.testing.assert_allclose(
            check_full,
            partial_full,
            err_msg="At Indexing of cholesky factor covariance did not result in expected matrix.",
        )

    def test_at_index_diagonal(self, diag_cov: DiagonalCovariance):
        rng = np.arange(0, diag_cov.matrix_shape[0], 2)
        index = np.ix_(rng, rng)
        partial_diag = diag_cov.at[..., *index]

        partial_full = partial_diag.full()
        check_full = diag_cov.full()[..., *index]
        np.testing.assert_allclose(
            check_full,
            partial_full,
            err_msg="At Indexing of diagonal covariance did not result in expected matrix.",
        )

    def test_biloc_index_cholesky_factor(self, chol_cov: CholeskyFactorCovariance):
        if chol_cov.ndim > 2:
            bidx = (np.array([0]),) if chol_cov.ndim == 3 else np.ix_([0], [1, 2])
            partial_chol_cov = chol_cov.biloc[*bidx]

            partial_full = partial_chol_cov.full()
            check_full = chol_cov.full()[bidx]
            np.testing.assert_allclose(
                check_full,
                partial_full,
                err_msg="Batch Indexing of cholesky factor covariance did not result in expected matrix.",
            )

    def test_biloc_index_diagonal(self, diag_cov: DiagonalCovariance):
        if diag_cov.ndim > 2:
            bidx = (np.array([0]),) if diag_cov.ndim == 3 else np.ix_([0], [1, 2])
            partial_diag_cov = diag_cov.biloc[*bidx]

            partial_full = partial_diag_cov.full()
            check_full = diag_cov.full()[bidx]
            np.testing.assert_allclose(
                check_full,
                partial_full,
                err_msg="Batch Indexing of cholesky factor covariance did not result in expected matrix.",
            )

    def test_diagonal_broadcast_to(self, diag_cov: DiagonalCovariance):
        bidx = (diag_cov.ndim**2,) + diag_cov.shape
        broadcasted_diag_cov = np.broadcast_to(diag_cov, bidx)

        broadcasted_full = np.broadcast_to(diag_cov.full(), bidx)

        check_full = broadcasted_diag_cov.full()

        np.testing.assert_allclose(
            broadcasted_full,
            check_full,
            err_msg="Broadcasting of diagonal covariance failed.",
        )

    def test_cholesky_broadcast_to(self, chol_cov: CholeskyFactorCovariance):
        bidx = (chol_cov.ndim**2,) + chol_cov.shape
        broadcasted_chol_cov = np.broadcast_to(chol_cov, bidx)

        broadcasted_full = np.broadcast_to(chol_cov.full(), bidx)

        check_full = broadcasted_chol_cov.full()

        np.testing.assert_allclose(
            broadcasted_full,
            check_full,
            err_msg="Broadcasting of cholesky factor covariance failed.",
        )


class TestCholeskyFactorCovariance:
    def test_init(self, L_factor: np.ndarray, P_full: np.ndarray):
        """Tests initialization and basic properties."""
        L = L_factor
        P = P_full
        cov = CholeskyFactorCovariance(L)
        matrix_shape = P_full.shape[-2:]
        shape = P_full.shape
        batch_shape = P_full.shape[:-2]

        assert_allclose(cov.cholesky_factor, L)
        assert_allclose(cov.full(), P)
        assert cov.matrix_shape == matrix_shape
        assert cov.shape == shape
        assert cov.batch_shape == batch_shape

    def test_init_raises(self):
        """Tests that __init__ raises ValueErrors for bad shapes."""
        with pytest.raises(ValueError, match="at least 2-D"):
            CholeskyFactorCovariance(np.array([1.0, 2.0]))

        with pytest.raises(ValueError, match="square matrix"):
            CholeskyFactorCovariance(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_variance(self, chol_cov: CholeskyFactorCovariance, P_full: np.ndarray):
        """
        Tests the .variance property.
        This test checks the bug fix (axis=-1).
        """
        expected_variance = np.diagonal(P_full, axis1=-2, axis2=-1)
        assert_allclose(chol_cov.variance, expected_variance)

    def test_add_chol(self, chol_cov: CholeskyFactorCovariance, P_full: np.ndarray):
        """Tests __add__ with another CholeskyFactorCovariance."""
        # Create a second covariance
        P2 = P_full * 0.5 + 2.0 * np.eye(P_full.shape[-1])
        L2, _ = cho_factor(P2, lower=True)
        chol_cov2 = CholeskyFactorCovariance(np.tril(L2))

        result = chol_cov + chol_cov2
        expected_P = P_full + P2

        assert isinstance(result, CholeskyFactorCovariance)
        assert_allclose(result.full(), expected_P)

    def test_add_ndarray(self, chol_cov: CholeskyFactorCovariance, P_full: np.ndarray):
        """Tests __add__ with a raw np.ndarray."""
        P2 = P_full * 0.5 + 2.0 * np.eye(P_full.shape[-1])

        result = chol_cov + P2
        expected_P = P_full + P2

        assert isinstance(result, CholeskyFactorCovariance)
        assert_allclose(result.full(), expected_P)

    def test_add_diag(
        self,
        chol_cov: CholeskyFactorCovariance,
        P_full: np.ndarray,
        diag_cov: DiagonalCovariance,
    ):
        """Tests __add__ with a DiagonalCovariance."""
        result = chol_cov + diag_cov
        expected_P = P_full + diag_cov.full()

        assert isinstance(result, CholeskyFactorCovariance)
        assert_allclose(result.full(), expected_P)

    def test_sub_chol(self, chol_cov: CholeskyFactorCovariance, P_full: np.ndarray):
        """Tests __sub__ with another CholeskyFactorCovariance."""
        P2 = P_full * 0.5  # P - 0.5*P = 0.5*P (still pos-def)
        L2, _ = cho_factor(P2, lower=True)
        chol_cov2 = CholeskyFactorCovariance(np.tril(L2))

        result = chol_cov - chol_cov2
        expected_P = P_full - P2

        assert isinstance(result, CholeskyFactorCovariance)
        assert_allclose(result.full(), expected_P)

    def test_sub_diag(
        self,
        chol_cov: CholeskyFactorCovariance,
        P_full: np.ndarray,
        diag_cov: DiagonalCovariance,
    ):
        """Tests __sub__ with a DiagonalCovariance."""
        # Make sure P - D is still pos-def
        P_diag_sub = P_full - diag_cov.full() * 0.1
        chol_cov_sub = cholesky_factor(P_diag_sub)
        diag_cov_small = DiagonalCovariance(diag_cov._D * 0.1**0.5)

        result = chol_cov_sub - diag_cov_small
        expected_P = P_diag_sub - diag_cov_small.full()

        assert isinstance(result, CholeskyFactorCovariance)
        assert_allclose(result.full(), expected_P, atol=1e-14)

    def test_mul(
        self,
        chol_cov: CholeskyFactorCovariance,
        P_full: np.ndarray,
        L_factor: np.ndarray,
    ):
        """Tests __mul__ by a scalar."""
        scalar = 4.0
        result = chol_cov * scalar

        expected_P = P_full * scalar
        expected_L = L_factor * (scalar**0.5)

        assert isinstance(result, CholeskyFactorCovariance)
        assert_allclose(result.full(), expected_P)
        assert_allclose(result.cholesky_factor, expected_L)

    def test_imul(self, P_full: np.ndarray, L_factor: np.ndarray):
        """Tests __imul__ by a scalar."""
        chol_cov = CholeskyFactorCovariance(L_factor.copy())
        scalar = 9.0

        result = chol_cov.__imul__(scalar)

        expected_P = P_full * scalar
        expected_L = L_factor * (scalar**0.5)

        # Check that the original object was modified
        assert_allclose(chol_cov.full(), expected_P)
        assert_allclose(chol_cov.cholesky_factor, expected_L)
        # Check that the returned object is the same object
        assert result is chol_cov

    def test_quadratic_form(
        self,
        chol_cov: CholeskyFactorCovariance,
        P_full: np.ndarray,
        A_matrix: np.ndarray,
    ):
        """Tests the quadratic_form method."""
        A = A_matrix
        L = chol_cov.cholesky_factor

        result_cov = chol_cov.quadratic_form(A)

        # Test the Cholesky factor directly
        expected_L = A @ L
        assert_allclose(result_cov.cholesky_factor, expected_L)

        # Test the full matrix

        expected_P = np.einsum("...ik,...kl,...jl->...ij", A, P_full, A, optimize=True)
        assert_allclose(result_cov.full(), expected_P)

    def test_type_errors(self, chol_cov: CholeskyFactorCovariance):
        """Tests that invalid types raise TypeError."""
        with pytest.raises(TypeError, match=re.escape(type_error_msg("string"))):
            chol_cov + "string"

    def test_inverse(self, chol_cov: CholeskyFactorCovariance):
        """Test that the inverse is correctly computed."""
        inv = chol_cov.inverse()
        inv_check = np.linalg.inv(chol_cov.full())

        np.testing.assert_allclose(
            inv_check,
            inv,
            err_msg="Inverse computation failed for CholeskyFactorCovariance",
        )


# --- Test DiagonalCovariance ---


class TestDiagonalCovariance:
    def test_init(self, diag_std: np.ndarray, dim: int):
        """Tests initialization and basic properties."""
        cov = DiagonalCovariance(diag_std)

        assert_allclose(cov._D, diag_std)
        assert cov.matrix_shape == (dim, dim)
        assert cov.shape == diag_std.shape + (dim,)
        assert cov.batch_shape == diag_std.shape[:-1]

    def test_init_raises(self):
        """Tests that __init__ raises ValueError for bad shapes."""
        with pytest.raises(ValueError, match="at least 1-D"):
            # A 0-D array (scalar)
            DiagonalCovariance(np.array(1.0))

    def test_variance(self, diag_cov: DiagonalCovariance, diag_std: np.ndarray):
        """Tests the .variance property."""
        expected_variance = diag_std**2
        assert_allclose(diag_cov.variance, expected_variance)

    def test_full(self, diag_cov: DiagonalCovariance, diag_std: np.ndarray):
        """Tests the .full() method."""
        expected_P = np.zeros(diag_cov.shape)
        expected_P[..., *np.diag_indices(diag_cov.matrix_shape[-1])] = diag_std**2
        assert_allclose(diag_cov.full(), expected_P)

    def test_cholesky_factor(self, diag_cov: DiagonalCovariance, diag_std: np.ndarray):
        """Tests the .cholesky_factor property."""
        expected_L = np.zeros(diag_cov.shape)
        expected_L[..., *np.diag_indices(diag_cov.matrix_shape[-1])] = diag_std
        assert_allclose(diag_cov.cholesky_factor, expected_L)

    def test_add_diag(
        self, diag_cov: DiagonalCovariance, diag_std: np.ndarray, dim: int
    ):
        """Tests __add__ with another DiagonalCovariance."""
        std_2 = np.random.rand(dim) + 0.5
        cov_2 = DiagonalCovariance(std_2)

        result = diag_cov + cov_2

        expected_variance = diag_std**2 + std_2**2
        expected_std = expected_variance**0.5

        assert isinstance(result, DiagonalCovariance)
        assert_allclose(result.variance, expected_variance)
        assert_allclose(result._D, expected_std)

    def test_add_chol(
        self,
        diag_cov: DiagonalCovariance,
        chol_cov: CholeskyFactorCovariance,
        P_full: np.ndarray,
    ):
        """Tests __add__ with a CholeskyFactorCovariance."""
        result = diag_cov + chol_cov
        expected_P = diag_cov.full() + P_full

        assert isinstance(result, CholeskyFactorCovariance)
        assert_allclose(result.full(), expected_P)

    def test_sub_diag(self, diag_cov: DiagonalCovariance, diag_std: np.ndarray):
        """Tests __sub__ with another DiagonalCovariance."""
        std_2 = diag_std * 0.5  # Ensure result is positive
        cov_2 = DiagonalCovariance(std_2)

        result = diag_cov - cov_2  # self - other

        expected_variance = diag_std**2 - std_2**2
        expected_std = expected_variance**0.5

        assert isinstance(result, DiagonalCovariance)
        assert_allclose(result.variance, expected_variance)
        assert_allclose(result._D, expected_std)

    def test_sub_from_chol(
        self,
        diag_cov: DiagonalCovariance,
        chol_cov: CholeskyFactorCovariance,
        P_full: np.ndarray,
    ):
        """Tests subtraction of a DiagonalCovariance from a CholeskyFactorCovariance."""
        # This tests chol - diag
        result = chol_cov - diag_cov
        expected_P = P_full - diag_cov.full()

        assert isinstance(result, CholeskyFactorCovariance)
        assert_allclose(result.full(), expected_P)

    def test_sub_chol_from_diag(
        self,
        diag_cov: DiagonalCovariance,
        chol_cov: CholeskyFactorCovariance,
        P_full: np.ndarray,
    ):
        """Tests subtraction of a CholeskyFactorCovariance from a DiagonalCovariance."""
        # This tests diag - chol
        # We need to make sure diag_cov is "bigger" than chol_cov
        diag_cov_large = DiagonalCovariance(
            np.diagonal(P_full, axis1=-2, axis2=-1) * 2 + 1**0.5
        )

        result = diag_cov_large - chol_cov  # self - other
        expected_P = diag_cov_large.full() - P_full

        assert isinstance(result, CholeskyFactorCovariance)
        assert_allclose(result.full(), expected_P)

    def test_mul(self, diag_cov: DiagonalCovariance, diag_std: np.ndarray):
        """Tests __mul__ by a scalar."""
        scalar = 9.0
        result = diag_cov * scalar

        expected_P = diag_cov.full() * scalar
        expected_std = diag_std * (scalar**0.5)

        assert isinstance(result, DiagonalCovariance)
        assert_allclose(result.full(), expected_P)
        assert_allclose(result._D, expected_std)

    def test_imul(self, diag_std: np.ndarray):
        """Tests __imul__ by a scalar."""
        diag_cov = DiagonalCovariance(diag_std.copy())
        scalar = 4.0

        result = diag_cov.__imul__(scalar)

        expected_std = diag_std * (scalar**0.5)

        assert_allclose(diag_cov._D, expected_std)
        assert result is diag_cov

    def test_quadratic_form(
        self, diag_cov: DiagonalCovariance, diag_std: np.ndarray, A_matrix: np.ndarray
    ):
        """
        Tests the quadratic_form method.
        This test checks the bug fix (A * D broadcasting).
        """
        A = A_matrix

        result_cov = diag_cov.quadratic_form(A)

        # Test the Cholesky factor directly
        # L_new = A @ L_old = A @ diag(D) = A * D
        expected_L = A * diag_std[..., np.newaxis, :]  # (M,N) * (1,N)
        assert_allclose(result_cov.cholesky_factor, expected_L)

        # Test the full matrix
        expected_P = A @ diag_cov.full() @ A.swapaxes(-1, -2)
        assert_allclose(result_cov.full(), expected_P)

    def test_type_errors(self, diag_cov: DiagonalCovariance):
        """Tests that invalid types raise TypeError."""
        with pytest.raises(TypeError, match=re.escape(type_error_msg("string"))):
            diag_cov + "string"

        with pytest.raises(TypeError, match=re.escape(type_error_msg(123))):
            diag_cov - 123

    def test_inverse(self, diag_cov: DiagonalCovariance):
        """Test that the inverse is correctly computed."""
        inv = diag_cov.inverse()
        inv_check = np.linalg.inv(diag_cov.full())

        np.testing.assert_allclose(
            inv_check, inv, err_msg="Inverse computation failed for DiagonalCovariance"
        )
