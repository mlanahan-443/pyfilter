from pyfilter.hints.jax_hints import JaxFloatArray
from pyfilter.models.linear._base import GenericLinearTransform, LTI_Transition
from pyfilter.smooth.linear import LinearGaussianFixedPointSmoother
import pytest
import jax
from jax import numpy as jnp
from tests.model_mock import SimpleProcessNoise
from pyfilter.types import GaussianRV, DiagonalCovariance, CholeskyFactorCovariance, InformationCovariance


@pytest.fixture
def fixed_point_smoother(batch_shape) -> LinearGaussianFixedPointSmoother:
    batch_shape = tuple([1] * len(batch_shape))

    A = jnp.eye(4)

    transition_model = LTI_Transition(A)

    # Process noise
    process_noise = SimpleProcessNoise(4)

    # Measurement model: observe first 2 components
    H = jax.nn.one_hot(jnp.array([0, 1]), 4)
    H, A = [jnp.broadcast_to(arr, batch_shape + arr.shape) for arr in [H, A]]

    measurement_model = GenericLinearTransform(H)
    return LinearGaussianFixedPointSmoother(
        transition_model=transition_model, process_noise=process_noise, measurement_model=measurement_model
    )


class TestLinearGaussianFixedPointSmoother:
    """Test the linear guassian fixed point smoother."""

    @staticmethod
    def _test_lgfps_api(x_s, x_u, sigma, batch_shape, shape):

        # Test the output objects are correct.
        assert isinstance(x_s, GaussianRV)
        assert isinstance(x_u, GaussianRV)
        assert isinstance(sigma, jnp.ndarray)

        # Test yields correctly shaped outputs.
        assert x_s.batch_shape == batch_shape
        assert x_u.batch_shape == batch_shape
        assert x_s.shape == shape
        assert x_u.shape == shape
        assert x_u.batch_shape == batch_shape
        assert sigma.shape == shape + (4,)

    def test_api(
        self,
        P_full: JaxFloatArray,
        fixed_point_smoother: LinearGaussianFixedPointSmoother,
        diag_cov: DiagonalCovariance,
    ) -> None:
        """Test the API I/O of the filter."""

        x_init = jnp.ones(P_full.shape[:-2] + (4,))
        grv = GaussianRV(x_init, P_full)
        t_init = jnp.ones(P_full.shape[:-2])
        x_s, x_u, sigma, t = fixed_point_smoother.initialize_fixed_point(grv, t_init)

        self._test_lgfps_api(x_s, x_u, sigma, P_full.shape[:-2], x_init.shape)

        mcov = [
            diag_cov[..., :2, :2],
            diag_cov.full()[:2, :2],
            CholeskyFactorCovariance(diag_cov.full()[:2, :2]),
            InformationCovariance(1.0 / diag_cov.full()[:2, :2]),
        ]
        for mc in mcov:
            measurement = GaussianRV(jnp.ones(P_full.shape[:-2] + (2,)), covariance=mc)
            new_t = t_init + 1.0
            x_s, x_u, sigma = fixed_point_smoother.update(x_s, x_u, sigma, t, measurement, new_t)

            self._test_lgfps_api(x_s, x_u, sigma, P_full.shape[:-2], x_init.shape)
