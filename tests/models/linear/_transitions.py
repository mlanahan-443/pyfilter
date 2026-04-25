# test_transitions.py
import numpy as np
import pytest
import scipy.linalg

from pyfilter.models.linear import IntegratorChainTransition


class TestConstruction:
    def test_state_dim(self) -> None:
        assert IntegratorChainTransition(n=3, p=2).state_dim == 6
        assert IntegratorChainTransition(n=1, p=4).state_dim == 4
        assert IntegratorChainTransition(n=5, p=1).state_dim == 5

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            IntegratorChainTransition(n=0, p=2)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="p must be >= 1"):
            IntegratorChainTransition(n=3, p=0)

    def test_frozen(self) -> None:
        chain = IntegratorChainTransition(n=2, p=2)
        with pytest.raises(Exception):  # FrozenInstanceError
            chain.n = 3  # type: ignore[misc]


class TestAgainstExpm:
    """The closed form should match expm(A*dt) to ~machine precision."""

    @pytest.mark.parametrize(
        "n,p", [(1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4)]
    )
    @pytest.mark.parametrize("dt", [0.1, 1.0, 5.0])
    def test_phi_matches_expm(self, n: int, p: int, dt: float) -> None:
        chain = IntegratorChainTransition(n=n, p=p)
        Phi_closed = chain.matrix(np.asarray(dt))
        Phi_expm = scipy.linalg.expm(chain.A * dt)

        np.testing.assert_allclose(Phi_closed, Phi_expm, atol=1e-12, rtol=1e-12)


class TestBatching:
    def test_scalar_dt(self) -> None:
        chain = IntegratorChainTransition(n=2, p=2)
        Phi = chain.matrix(np.asarray(0.5))
        assert Phi.shape == (4, 4)

    def test_1d_batch(self) -> None:
        chain = IntegratorChainTransition(n=2, p=3)
        dts = np.array([0.1, 0.2, 0.5, 1.0])
        Phi = chain.matrix(dts)
        assert Phi.shape == (4, 6, 6)

        # Each slice should match the corresponding scalar call.
        for i, dt in enumerate(dts):
            Phi_i = chain.matrix(np.asarray(dt))
            np.testing.assert_allclose(Phi[i], Phi_i, atol=1e-14)

    def test_2d_batch(self) -> None:
        chain = IntegratorChainTransition(n=1, p=2)
        dts = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # shape (3, 2)
        Phi = chain.matrix(dts)
        assert Phi.shape == (3, 2, 2, 2)


class TestGeneratorMatrix:
    def test_A_structure_p2(self) -> None:
        """For p=2, n=3: A should have I_3 in the upper-right block."""
        chain = IntegratorChainTransition(n=3, p=2)
        expected = np.block(
            [
                [np.zeros((3, 3)), np.eye(3)],
                [np.zeros((3, 3)), np.zeros((3, 3))],
            ]
        )
        np.testing.assert_array_equal(chain.A, expected)

    def test_A_is_nilpotent(self) -> None:
        """A^p = 0 for an integrator chain of order p."""
        for p in [2, 3, 4, 5]:
            chain = IntegratorChainTransition(n=2, p=p)
            A_pow = np.linalg.matrix_power(chain.A, p)
            np.testing.assert_allclose(A_pow, 0.0, atol=1e-15)

    def test_A_minus_one_power_nonzero(self) -> None:
        """A^(p-1) is nonzero (nilpotency index is exactly p)."""
        for p in [2, 3, 4]:
            chain = IntegratorChainTransition(n=2, p=p)
            A_pow = np.linalg.matrix_power(chain.A, p - 1)
            assert np.linalg.norm(A_pow) > 0
