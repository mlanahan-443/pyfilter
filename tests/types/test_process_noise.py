# test_wiener_process_noise.py
import numpy as np
import pytest

from pyfilter.config import FDTYPE_ as FDTYPE
from pyfilter.hints import FloatArray
from pyfilter.types.process_noise import VanLoanProcessNoise, WeinerProcessNoise


def integrator_chain_A(n: int, p: int) -> np.ndarray:
    """Continuous-time generator for the integrator chain (for cross-checks)."""
    d = n * p
    A = np.zeros((d, d))
    for i in range(p - 1):
        A[i * n : (i + 1) * n, (i + 1) * n : (i + 2) * n] = np.eye(n)
    return A


def s(x: float) -> np.ndarray:
    """Convenience: scalar dt as a 0-d array (matches the typed contract)."""
    return np.asarray(x, dtype=FDTYPE)


class TestConstruction:
    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n and p must be >= 1"):
            WeinerProcessNoise(n=0, p=2, intensity=np.asarray(1.0))

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="n and p must be >= 1"):
            WeinerProcessNoise(n=2, p=0, intensity=np.asarray(1.0))

    def test_state_dim(self) -> None:
        noise = WeinerProcessNoise(n=3, p=4, intensity=np.asarray(1.0))
        assert noise.state_dim == 12

    def test_invalid_1d_intensity_shape(self) -> None:
        noise = WeinerProcessNoise(n=3, p=2, intensity=np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="1-D intensity must have length 3"):
            _ = noise._intensity_matrix

    def test_invalid_matrix_intensity_shape(self) -> None:
        noise = WeinerProcessNoise(n=3, p=2, intensity=np.eye(2))
        with pytest.raises(ValueError, match=r"trailing shape \(3, 3\)"):
            _ = noise._intensity_matrix


class TestConstantVelocity:
    """Standard CV process noise: sigma^2 * [[dt^3/3 I, dt^2/2 I],
    [dt^2/2 I,    dt I]]."""

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("dt_val", [0.01, 0.1, 1.0, 5.0])
    @pytest.mark.parametrize("sigma2", [0.1, 1.0, 10.0])
    def test_matches_analytical(self, n: int, dt_val: float, sigma2: float) -> None:
        noise = WeinerProcessNoise(n=n, p=2, intensity=np.asarray(sigma2))
        Qd = noise.covariance(s(dt_val))

        I = np.eye(n)
        expected = sigma2 * np.block(
            [
                [(dt_val**3 / 3) * I, (dt_val**2 / 2) * I],
                [(dt_val**2 / 2) * I, dt_val * I],
            ]
        )
        np.testing.assert_allclose(Qd, expected, atol=1e-14)


class TestConstantAcceleration:
    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("dt_val", [0.05, 0.5, 2.0])
    def test_matches_analytical(self, n: int, dt_val: float) -> None:
        sigma2 = 1.5
        noise = WeinerProcessNoise(n=n, p=3, intensity=np.asarray(sigma2))
        Qd = noise.covariance(s(dt_val))

        I = np.eye(n)
        np.zeros((n, n))
        expected = sigma2 * np.block(
            [
                [(dt_val**5 / 20) * I, (dt_val**4 / 8) * I, (dt_val**3 / 6) * I],
                [(dt_val**4 / 8) * I, (dt_val**3 / 3) * I, (dt_val**2 / 2) * I],
                [(dt_val**3 / 6) * I, (dt_val**2 / 2) * I, dt_val * I],
            ]
        )
        np.testing.assert_allclose(Qd, expected, atol=1e-13)


class TestIntensityForms:
    def test_scalar_zero_d_array(self) -> None:
        """0-d array intensity should work and broadcast as a scalar."""
        noise = WeinerProcessNoise(n=2, p=2, intensity=np.asarray(3.0))
        Qd = noise.covariance(s(0.1))
        # Top-left block is sigma^2 * dt^3/3 * I_2 = 3 * 0.001/3 * I_2 = 0.001 * I_2
        np.testing.assert_allclose(Qd[:2, :2], 0.001 * np.eye(2), atol=1e-15)

    def test_scalar_vs_diagonal_equivalence(self) -> None:
        """Scalar 2.0 should match diagonal [2, 2, 2]."""
        n_scalar = WeinerProcessNoise(n=3, p=2, intensity=np.asarray(2.0))
        n_diag = WeinerProcessNoise(n=3, p=2, intensity=np.array([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(
            n_scalar.covariance(s(0.5)),
            n_diag.covariance(s(0.5)),
            atol=1e-15,
        )

    def test_anisotropic_diagonal(self) -> None:
        """Different sigma^2 per spatial axis."""
        noise = WeinerProcessNoise(
            n=2,
            p=2,
            intensity=np.array([1.0, 4.0]),
        )
        Qd = noise.covariance(s(0.1))
        # Block (0, 0) should be (dt^3/3) * diag(1, 4)
        expected_top_left = (0.1**3 / 3) * np.diag([1.0, 4.0])
        np.testing.assert_allclose(Qd[:2, :2], expected_top_left, atol=1e-15)

    def test_full_matrix_intensity(self) -> None:
        """Full (n, n) intensity matrix with off-diagonal coupling."""
        Q_tilde = np.array([[2.0, 0.5], [0.5, 3.0]])
        noise = WeinerProcessNoise(n=2, p=2, intensity=Q_tilde)
        Qd = noise.covariance(s(0.1))

        # Top-left block: (dt^3 / 3) * Q_tilde
        np.testing.assert_allclose(
            Qd[:2, :2],
            (0.1**3 / 3) * Q_tilde,
            atol=1e-15,
        )
        # Bottom-right: dt * Q_tilde
        np.testing.assert_allclose(
            Qd[2:, 2:],
            0.1 * Q_tilde,
            atol=1e-15,
        )


class TestAlgebraicProperties:
    @pytest.mark.parametrize("n,p", [(2, 2), (3, 3), (1, 4)])
    def test_zero_dt_gives_zero(self, n: int, p: int) -> None:
        noise = WeinerProcessNoise(n=n, p=p, intensity=np.asarray(1.0))
        Qd = noise.covariance(s(0.0))
        np.testing.assert_allclose(Qd, 0.0, atol=1e-15)

    @pytest.mark.parametrize("n,p", [(2, 2), (3, 3), (2, 4)])
    @pytest.mark.parametrize("dt_val", [0.1, 1.0, 3.0])
    def test_qd_is_exactly_symmetric(self, n: int, p: int, dt_val: float) -> None:
        """The closed form is symmetric by construction (no fp asymmetry)."""
        noise = WeinerProcessNoise(n=n, p=p, intensity=np.asarray(1.5))
        Qd = noise.covariance(s(dt_val))
        np.testing.assert_array_equal(Qd, Qd.T)

    @pytest.mark.parametrize("n,p", [(2, 2), (3, 3), (2, 4)])
    @pytest.mark.parametrize("dt_val", [0.01, 1.0, 5.0])
    def test_qd_is_positive_definite(self, n: int, p: int, dt_val: float) -> None:
        """Cholesky succeeds iff PD."""
        noise = WeinerProcessNoise(n=n, p=p, intensity=np.asarray(1.0))
        Qd = noise.covariance(s(dt_val))
        np.linalg.cholesky(Qd)  # raises LinAlgError if not PD


class TestBatching:
    """The contract: dt is a FloatArray (possibly 0-d). Output shape is
    (*broadcast(dt.shape, intensity.batch_shape), state_dim, state_dim)."""

    def test_zero_d_dt(self) -> None:
        """0-d dt with no intensity batch -> (state_dim, state_dim)."""
        noise = WeinerProcessNoise(n=2, p=2, intensity=np.asarray(1.0))
        Qd = noise.covariance(s(0.5))
        assert Qd.shape == (4, 4)

    def test_one_d_dt(self) -> None:
        """1-d dt -> leading batch dim."""
        noise = WeinerProcessNoise(n=2, p=3, intensity=np.asarray(1.0))
        dts = np.array([0.1, 0.2, 0.5, 1.0])
        Qd = noise.covariance(dts)
        assert Qd.shape == (4, 6, 6)

        # Each slice should match the corresponding 0-d call.
        for i in range(len(dts)):
            np.testing.assert_allclose(
                Qd[i],
                noise.covariance(s(float(dts[i]))),
                atol=1e-14,
            )

    def test_two_d_dt(self) -> None:
        """2-d dt -> two leading batch dims."""
        noise = WeinerProcessNoise(n=1, p=2, intensity=np.asarray(1.0))
        dts = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        Qd = noise.covariance(dts)
        assert Qd.shape == (3, 2, 2, 2)

    def test_batched_intensity_zero_d_dt(self) -> None:
        """Different intensity per batch element, scalar dt."""
        intensities = np.stack([np.eye(2), 2 * np.eye(2), 3 * np.eye(2)])  # (3, 2, 2)
        noise = WeinerProcessNoise(n=2, p=2, intensity=intensities)
        Qd = noise.covariance(s(0.1))
        assert Qd.shape == (3, 4, 4)
        # Linear-in-intensity check
        np.testing.assert_allclose(Qd[1], 2.0 * Qd[0], atol=1e-15)
        np.testing.assert_allclose(Qd[2], 3.0 * Qd[0], atol=1e-15)

    def test_batched_intensity_and_dt_broadcast(self) -> None:
        """Compatible-broadcast batches of dt and intensity."""
        # dt shape (3,), intensity shape (3, 2, 2): broadcasts to (3,)
        dts = np.array([0.1, 0.2, 0.3])
        intensities = np.stack([np.eye(2), 2 * np.eye(2), 3 * np.eye(2)])
        noise = WeinerProcessNoise(n=2, p=2, intensity=intensities)
        Qd = noise.covariance(dts)
        assert Qd.shape == (3, 4, 4)

        # Verify each slice matches a manual scalar computation
        for i, (dt_i, sigma2) in enumerate(zip(dts, [1.0, 2.0, 3.0], strict=False)):
            ref = WeinerProcessNoise(n=2, p=2, intensity=np.asarray(sigma2))
            np.testing.assert_allclose(
                Qd[i],
                ref.covariance(s(float(dt_i))),
                atol=1e-14,
            )

    def test_broadcast_dt_against_intensity(self) -> None:
        """dt shape (3, 1), intensity batch (2, 2, 2) -> broadcast to (3, 2)."""
        dts = np.array([[0.1], [0.2], [0.3]])  # (3, 1)
        intensities = np.stack([np.eye(2), 2 * np.eye(2)])  # (2, 2, 2)
        noise = WeinerProcessNoise(n=2, p=2, intensity=intensities)
        Qd = noise.covariance(dts)
        assert Qd.shape == (3, 2, 4, 4)

        # Spot check: Qd[i, j] should equal the (dt=dts[i, 0], sigma2=j+1) result
        for i in range(3):
            for j in range(2):
                ref = WeinerProcessNoise(
                    n=2,
                    p=2,
                    intensity=np.asarray(float(j + 1)),
                )
                np.testing.assert_allclose(
                    Qd[i, j],
                    ref.covariance(s(float(dts[i, 0]))),
                    atol=1e-14,
                )


class TestVanLoan:
    """Test correctness of van loan discritization."""

    @pytest.mark.parametrize("n,p", [(2, 2), (3, 3), (2, 4)])
    @pytest.mark.parametrize(
        "dt_val",
        [0.1, 1.0, 3.0, np.array([0.1, 1.0, 3.0]), np.array([[0.1, 0.2], [1.4, 2.1]])],
    )
    def test_against_weiner_process(self, n: int, p: int, dt_val: float | FloatArray):
        """Test correctness of van loan against weiner process."""
        intensities = np.array(0.1)
        wpn = WeinerProcessNoise(n, p, intensities)
        A = integrator_chain_A(n, p)
        d = n * p
        Qc = np.zeros((d, d))
        Qc[(p - 1) * n :, (p - 1) * n :] = intensities * np.eye(
            n
        )  # noise only on highest deriv

        vlpn = VanLoanProcessNoise(A, Qc)

        dt = np.asarray(dt_val)

        np.testing.assert_allclose(vlpn.covariance(dt), wpn.covariance(dt), atol=1e-14)
