from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy

from pyfilter.models.linear import LinearTransformBase, LinearTransitionBase
from pyfilter.types import Covariance, CovarianceBase
from pyfilter.types.covariance import CholeskyFactorCovariance

from ..hints import FloatArray
from ..types.process_noise import ProcessNoise
from ..types.random_variables import GaussianRV

type Variable = GaussianRV[Any]


@dataclass
class BaseLinearGaussianKalmanFilter[
    StateCovariance: Covariance,
    MeasurementCovariance: Covariance,
](ABC):
    """Base class for linear guassian kalman filter."""

    transition_model: LinearTransitionBase[GaussianRV[StateCovariance]]
    process_noise: ProcessNoise
    measurement_model: LinearTransformBase[GaussianRV[MeasurementCovariance]]

    def predict(
        self, current_state: GaussianRV[StateCovariance], dt: FloatArray
    ) -> GaussianRV[StateCovariance]:
        """Predict the state forward."""
        return self.transition_model.transform(current_state, dt) + self.process_noise(
            dt
        )

    @abstractmethod
    def update(
        self,
        state_prediction: GaussianRV[StateCovariance],
        measurement: GaussianRV[MeasurementCovariance],
    ) -> GaussianRV[StateCovariance]:
        """Update from residual + prediction method."""
        ...

    def predicted_measurement(
        self,
        state_prediction: GaussianRV[StateCovariance],
        innovation: GaussianRV[StateCovariance],
    ) -> GaussianRV[StateCovariance]:
        # The measurement prediction: z ~ N(H @ x_pred, S) where S = innovation.covariance
        # Since innovation.mean = z_obs - H @ x_pred, we have z_obs = innovation.mean + H @ x_pred
        H = self.measurement_model.matrix
        predicted_measurement_mean = H @ state_prediction.mean

        # Create the predicted measurement distribution
        return GaussianRV(predicted_measurement_mean, innovation.covariance)

    def innovation(
        self,
        state_prediction: GaussianRV[StateCovariance],
        measurement: GaussianRV[MeasurementCovariance],
    ) -> GaussianRV[StateCovariance]:
        """Compute the measurement innovation (residual).

        Args:
            state_prediction: Prior state distribution
            measurement: Observed measurement distribution

        Returns:
            Innovation: y = z - H @ x_pred
        """
        return measurement - self.measurement_model @ state_prediction


@dataclass
class LinearGaussianKalman[
    StateCovariance: Covariance,
    MeasurementCovariance: Covariance,
](BaseLinearGaussianKalmanFilter[StateCovariance, MeasurementCovariance]):
    """Base linear Gaussian Kalman filter."""

    def update(
        self,
        state_prediction: GaussianRV[StateCovariance],
        measurement: GaussianRV[MeasurementCovariance],
    ) -> GaussianRV[Any]:
        """Update state using measurement innovation (Kalman filter update step).

        This method implements the Kalman update using the conditional distribution.
        Given:
        - State prior: x ~ N(x_pred, P)
        - Measurement model: z = H @ x + v, where v ~ N(0, R)
        - Innovation: y = z - H @ x_pred

        We compute x | z using the conditional distribution of the joint (x, z).

        Args:
            state_prediction: Prior state distribution x ~ N(x_pred, P)
            innovation: Innovation distribution y ~ N(y_obs, S) where S = H @ P @ H.T + R
                       The mean is the observed innovation y_obs = z_obs - H @ x_pred

        Returns:
            Posterior state distribution x | z ~ N(x_post, P_post)
        """
        # Compute cross-covariance: Cov(x, z) = P @ H.T
        cross_covariance = state_prediction.linear_cross(self.measurement_model.matrix)

        innovation = self.innovation(state_prediction, measurement)
        predicted_measurement = self.predicted_measurement(state_prediction, innovation)
        measurement_value = innovation.mean + predicted_measurement.mean

        # Compute conditional: x | z = z_obs
        return state_prediction.conditional(
            predicted_measurement, cross_covariance, given_value=measurement_value
        )


class SquareRootLinearGuassianKalman[
    StateCovariance: CholeskyFactorCovariance,
    MeasurementCovariance: CovarianceBase,
](BaseLinearGaussianKalmanFilter[StateCovariance, MeasurementCovariance]):
    """Implements the square root filter.

    This filter maintains the covariance of the state as a cholesky factor, vs. the full covariance matrix.
    This allows for a more efficient update step using a QR decomposition, at the expense of more
    expensive prediction + other steps.

    The measurement covariance must be specified as a covariance type so the cholesky factor can be efficiently
    computed.

    """

    def update(
        self,
        state_prediction: GaussianRV[StateCovariance],
        measurement: GaussianRV[MeasurementCovariance],
    ) -> GaussianRV[Any]:
        innovation = self.innovation(state_prediction, measurement)
        L_pred = state_prediction.covariance.cholesky_factor  # (n, n)
        L_R = measurement.covariance.cholesky_factor

        H = self.measurement_model.matrix  # (m, n)
        n, m = L_pred.shape[-1], L_R.shape[-1]

        # Pre-array (supports batching via leading dims)
        HL = H @ L_pred  # (..., m, n)
        top = np.concatenate([L_R, HL], axis=-1)  # (..., m, m+n)
        bottom = np.concatenate(
            [np.zeros((*L_pred.shape[:-2], n, m)), L_pred], axis=-1
        )  # (..., n, m+n)
        A = np.concatenate([top, bottom], axis=-2)  # (..., m+n, m+n)

        B = np.linalg.qr(A.mT, mode="reduced").R  # upper-tri (..., m+n, m+n)

        L_S_T = B[..., :m, :m]
        KLS_T = B[..., :m, m:]
        L_post_T = B[..., m:, m:]

        # K via triangular solve (cheap)
        # We have KLS = K @ L_S, so K = KLS @ L_S^(-1)
        # Solve L_S.T @ X.T = KLS.T for X, which gives X = KLS @ L_S^(-1)
        K = scipy.linalg.solve_triangular(L_S_T, KLS_T, lower=False).mT

        # mean update
        posterior_mean = state_prediction.mean + K @ innovation.mean

        return GaussianRV(posterior_mean, CholeskyFactorCovariance(L_post_T.mT))
