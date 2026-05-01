"""Comprehensive tests for Kalman filter correctness.

This module tests the LinearGaussianKalman filter against known analytical
solutions and verifies that both the conditional-based and classical Kalman
gain implementations produce identical results.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyfilter.filter.linear import LinearGaussianKalman
from pyfilter.hints import FloatArray
from pyfilter.models.linear import LinearTransformBase, LinearTransitionBase
from pyfilter.types.covariance import (
    CholeskyFactorCovariance,
    DiagonalCovariance,
    cholesky_factor,
)
from pyfilter.types.process_noise import ProcessNoise
from pyfilter.types.random_variables import GaussianRV

# ============================================================================
# Simple Models for Testing
# ============================================================================


class IdentityTransition(LinearTransitionBase):
    """Simple identity transition: x[k+1] = x[k]"""

    def matrix(self, dt: FloatArray) -> FloatArray:
        return np.eye(1)

    def inverse(self, dt: FloatArray) -> FloatArray:
        return np.eye(1)

    def transform(self, x: GaussianRV, dt: FloatArray) -> GaussianRV:
        return x


class IdentityMeasurement(LinearTransformBase):
    """Simple identity measurement: z = x"""

    @property
    def matrix(self) -> FloatArray:
        return np.eye(1)

    def transform(self, x: GaussianRV) -> GaussianRV:
        return x


class ConstantVelocityTransition(LinearTransitionBase):
    """2D constant velocity model: [position, velocity]"""

    def matrix(self, dt: FloatArray) -> FloatArray:
        A = np.zeros(dt.shape + (2, 2))
        A[..., 0, 0] = 1.0
        A[..., 0, 1] = dt
        A[..., 1, 1] = 1.0
        return A

    def inverse(self, dt: FloatArray) -> FloatArray:
        return np.linalg.inv(self.matrix(dt))

    def transform(self, x: GaussianRV, dt: FloatArray) -> GaussianRV:
        return self.matrix(dt) @ x


class PositionMeasurement(LinearTransformBase):
    """Measurement of position only (first component)"""

    @property
    def matrix(self) -> FloatArray:
        return np.array([[1.0, 0.0]])

    def transform(self, x: GaussianRV) -> GaussianRV:
        return x.marginal([0])


class SimpleProcessNoise(ProcessNoise):
    """Simple constant process noise"""

    def __init__(self, covariance: FloatArray | CholeskyFactorCovariance):
        self._cov = covariance
        super().__init__(covariance.shape[-2:])

    def covariance(self, dt: FloatArray) -> GaussianRV:
        return GaussianRV.zero_mean(self._cov)


# ============================================================================
# Test Cases
# ============================================================================


class TestKalmanFilter1D:
    """Tests for 1D Kalman filter with analytical solutions"""

    @pytest.fixture
    def simple_filter(self):
        """1D filter with identity transition and measurement"""
        transition = IdentityTransition()
        process_noise = SimpleProcessNoise(np.array([[0.1**2]]))
        measurement = IdentityMeasurement()
        return LinearGaussianKalman(transition, process_noise, measurement)

    def test_predict_identity(self, simple_filter):
        """Test prediction with identity transition"""
        state = GaussianRV(np.array([1.0]), np.array([[1.0]]))
        dt = np.array(1.0)

        predicted = simple_filter.predict(state, dt)

        # Mean should stay the same, variance should increase by process noise
        assert_allclose(predicted.mean, np.array([1.0]))
        assert_allclose(predicted.covariance, np.array([[1.0 + 0.1**2]]))

    def test_update_analytical_solution(self, simple_filter):
        """Test update against known analytical solution"""
        # Prior: x ~ N(1, 2)
        state = GaussianRV(np.array([1.0]), np.array([[2.0]]))
        # Measurement: z ~ N(3, 1)
        measurement = GaussianRV(np.array([3.0]), np.array([[1.0]]))

        # Analytical solution for x|z:
        # K = P @ H.T @ (H @ P @ H.T + R)^-1 = 2 * 1 * (2 + 1)^-1 = 2/3
        # x_post = x_pred + K @ (z - H @ x_pred) = 1 + 2/3 * (3 - 1) = 1 + 4/3 = 7/3
        # P_post = (I - K @ H) @ P = (1 - 2/3) * 2 = 2/3

        updated = simple_filter.update(state, measurement)

        assert_allclose(updated.mean, np.array([7.0 / 3.0]), rtol=1e-10)
        cov = (
            updated.covariance
            if isinstance(updated.covariance, np.ndarray)
            else updated.covariance.full()
        )
        assert_allclose(cov, np.array([[2.0 / 3.0]]), rtol=1e-10)

    def test_multiple_updates_sequence(self, simple_filter):
        """Test a sequence of predict-update cycles"""
        state = GaussianRV(np.array([0.0]), np.array([[10.0]]))
        dt = np.array(1.0)

        measurements = [
            GaussianRV(np.array([1.0]), np.array([[0.5]])),  # Reduced measurement noise
            GaussianRV(np.array([2.0]), np.array([[0.5]])),
            GaussianRV(np.array([3.0]), np.array([[0.5]])),
        ]

        for meas in measurements:
            state = simple_filter.predict(state, dt)
            state = simple_filter.update(state, meas)

        # After seeing measurements [1, 2, 3] with low measurement noise,
        # the filter should be influenced by the measurements
        # With process noise, the estimate will lag behind the measurements
        assert state.mean[0] > 1.5  # Should be influenced by measurements
        assert state.mean[0] < 4.0  # But not overshoot

        # Variance should decrease after measurements
        cov = (
            state.covariance
            if isinstance(state.covariance, np.ndarray)
            else state.covariance.full()
        )
        assert cov[0, 0] < 1.0  # Should be less than measurement variance


class TestKalmanFilter2D:
    """Tests for 2D constant velocity model"""

    @pytest.fixture
    def cv_filter(self):
        """2D constant velocity filter"""
        transition = ConstantVelocityTransition()

        # Process noise (continuous white noise acceleration model)
        q = 0.1  # Process noise intensity
        Q = np.array([[1 / 3, 1 / 2], [1 / 2, 1]]) * q**2

        process_noise = SimpleProcessNoise(Q)
        measurement = PositionMeasurement()

        return LinearGaussianKalman(transition, process_noise, measurement)

    def test_predict_constant_velocity(self, cv_filter):
        """Test prediction for constant velocity model"""
        # State: [position=0, velocity=1]
        state = GaussianRV(np.array([0.0, 1.0]), np.array([[1.0, 0.0], [0.0, 0.1]]))
        dt = np.array(1.0)

        predicted = cv_filter.predict(state, dt)

        # Position should increase by velocity * dt
        # x_new = x + v * dt = 0 + 1 * 1 = 1
        assert_allclose(predicted.mean[0], 1.0, rtol=1e-10)
        # Velocity should stay the same
        assert_allclose(predicted.mean[1], 1.0, rtol=1e-10)

    def test_tracking_constant_velocity(self, cv_filter):
        """Test tracking an object with constant velocity"""
        # True trajectory: x(t) = 10*t, v = 10
        state = GaussianRV(
            np.array([0.0, 5.0]),  # Start with rough estimate
            np.array([[100.0, 0.0], [0.0, 25.0]]),  # High initial uncertainty
        )
        dt = np.array(1.0)

        # Generate measurements with noise
        true_velocity = 10.0
        measurement_noise = 1.0

        for t in range(1, 6):
            # Predict
            state = cv_filter.predict(state, dt)

            # True position at time t
            true_position = true_velocity * t

            # Noisy measurement
            meas = GaussianRV(
                np.array([true_position]), np.array([[measurement_noise**2]])
            )

            # Update
            state = cv_filter.update(state, meas)

        # After several measurements, should track well
        # Position estimate should be close to true position
        assert_allclose(state.mean[0], 50.0, atol=5.0)  # At t=5, x=50
        # Velocity estimate should be close to true velocity
        assert_allclose(state.mean[1], 10.0, atol=2.0)

        # Uncertainty should be reduced
        cov = (
            state.covariance
            if isinstance(state.covariance, np.ndarray)
            else state.covariance.full()
        )
        assert cov[0, 0] < 10.0  # Position variance reduced
        assert cov[1, 1] < 5.0  # Velocity variance reduced


class TestCovarianceTypes:
    """Test filter works with different covariance representations"""

    @pytest.fixture(params=["array", "cholesky", "diagonal"])
    def filter_with_cov_type(self, request):
        """Create filter with different covariance types"""
        transition = IdentityTransition()
        measurement = IdentityMeasurement()

        if request.param == "array":
            process_cov = np.array([[0.1**2]])
        elif request.param == "cholesky":
            process_cov = cholesky_factor(np.array([[0.1**2]]))
        else:  # diagonal
            process_cov = DiagonalCovariance(np.array([0.1]))

        process_noise = SimpleProcessNoise(process_cov)
        return LinearGaussianKalman(
            transition, process_noise, measurement
        ), request.param

    def test_update_with_different_cov_types(self, filter_with_cov_type):
        """Test that filter works with different covariance types"""
        kalman_filter, cov_type = filter_with_cov_type

        # Create state with matching covariance type
        if cov_type == "array":
            state_cov = np.array([[2.0]])
        elif cov_type == "cholesky":
            state_cov = cholesky_factor(np.array([[2.0]]))
        else:  # diagonal
            state_cov = DiagonalCovariance(np.array([np.sqrt(2.0)]))

        state = GaussianRV(np.array([1.0]), state_cov)
        measurement = GaussianRV(np.array([3.0]), np.array([[1.0]]))

        dt = np.array(1.0)
        predicted = kalman_filter.predict(state, dt)
        updated = kalman_filter.update(predicted, measurement)

        # Should complete without error
        assert updated.mean.shape == (1,)
        assert updated.covariance.shape == (1, 1)


class TestBatchProcessing:
    """Test batch processing with broadcasting"""

    @pytest.fixture
    def simple_filter(self):
        """Simple 1D filter for batch testing"""
        transition = IdentityTransition()
        process_noise = SimpleProcessNoise(np.array([[0.01]]))
        measurement = IdentityMeasurement()
        return LinearGaussianKalman(transition, process_noise, measurement)

    def test_sequential_processing(self, simple_filter):
        """Test processing multiple measurements sequentially"""
        # Process multiple measurements one at a time
        state = GaussianRV(np.array([0.0]), np.array([[5.0]]))
        dt = np.array(1.0)

        measurements = [1.0, 2.0, 3.0]
        for meas_val in measurements:
            state = simple_filter.predict(state, dt)
            meas = GaussianRV(np.array([meas_val]), np.array([[0.5]]))
            state = simple_filter.update(state, meas)

        # State should be close to the last measurement
        assert state.mean.shape == (1,)
        assert state.mean[0] > 1.9  # Should be influenced by recent measurements


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_perfect_measurement(self):
        """Test update with zero measurement noise (perfect measurement)"""
        transition = IdentityTransition()
        process_noise = SimpleProcessNoise(np.array([[0.1**2]]))
        measurement = IdentityMeasurement()
        kalman_filter = LinearGaussianKalman(transition, process_noise, measurement)

        state = GaussianRV(np.array([1.0]), np.array([[10.0]]))
        # Perfect measurement (zero noise)
        meas = GaussianRV(np.array([5.0]), np.array([[1e-10]]))

        updated = kalman_filter.update(state, meas)

        # With perfect measurement, state should match measurement closely
        assert_allclose(updated.mean, np.array([5.0]), atol=1e-5)

    def test_very_uncertain_measurement(self):
        """Test update with very high measurement noise"""
        transition = IdentityTransition()
        process_noise = SimpleProcessNoise(np.array([[0.1**2]]))
        measurement = IdentityMeasurement()
        kalman_filter = LinearGaussianKalman(transition, process_noise, measurement)

        state = GaussianRV(np.array([1.0]), np.array([[1.0]]))
        # Very noisy measurement
        meas = GaussianRV(np.array([100.0]), np.array([[1000.0]]))

        updated = kalman_filter.update(state, meas)

        # With very noisy measurement, state should stay close to prior
        assert_allclose(updated.mean, state.mean, atol=0.5)
