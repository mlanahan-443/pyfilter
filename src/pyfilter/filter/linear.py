from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy

from pyfilter.models.linear import LinearTransformBase, LinearTransitionBase
from pyfilter.types import Covariance
from pyfilter.types.covariance import CholeskyFactorCovariance

from ..hints import FloatArray
from ..types.process_noise import ProcessNoise
from ..types.random_variables import GaussianRV

type Variable = GaussianRV[Any]


@dataclass
class BaseLinearGaussianKalmanFilter[
    State: GaussianRV[Covariance],
    Measurement: GaussianRV[Covariance],
](ABC):
    """Base class for linear guassian kalman filter."""

    transition_model: LinearTransitionBase[State]
    process_noise: ProcessNoise
    measurement_model: LinearTransformBase[State]

    @abstractmethod
    def predict(self, current_state: State, dt: FloatArray) -> State:
        """Prediction method."""
        ...

    @abstractmethod
    def update(self, state_prediction: State, residual: State) -> FloatArray:
        """Update from residual + prediction method."""
        ...

    def predicted_measurement(
        self, state_prediction: State, innovation: Measurement
    ) -> GaussianRV[Any]:
        # The measurement prediction: z ~ N(H @ x_pred, S) where S = innovation.covariance
        # Since innovation.mean = z_obs - H @ x_pred, we have z_obs = innovation.mean + H @ x_pred
        H = self.measurement_model.matrix
        predicted_measurement_mean = H @ state_prediction.mean

        # Create the predicted measurement distribution
        return GaussianRV(predicted_measurement_mean, innovation.covariance)


@dataclass
class LinearGaussianKalman[
    State: GaussianRV[FloatArray],
    Measurement: GaussianRV[Covariance],
](BaseLinearGaussianKalmanFilter[State, Measurement]):
    """Base linear Gaussian Kalman filter."""

    def predict(self, current_state: State, dt: FloatArray) -> State:
        return self.transition_model.transform(current_state, dt) + self.process_noise(
            dt
        )

    def innovation(
        self, state_prediction: State, measurement: Measurement
    ) -> Measurement:
        """Compute the measurement innovation (residual).

        Args:
            state_prediction: Prior state distribution
            measurement: Observed measurement distribution

        Returns:
            Innovation: y = z - H @ x_pred
        """
        return measurement - self.measurement_model @ state_prediction

    def update(
        self, state_prediction: State, measurement: Measurement
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
    State: GaussianRV[CholeskyFactorCovariance],
    Measurement: GaussianRV[Covariance],
](BaseLinearGaussianKalmanFilter[State, Measurement]):
    def predict(self, current_state: State, dt: FloatArray) -> State:
        return self.transition_model.transform(current_state, dt) + self.process_noise(
            dt
        )

    def innovation(
        self, state_prediction: State, measurement: Measurement
    ) -> Measurement:
        """Compute the measurement innovation (residual).

        Args:
            state_prediction: Prior state distribution
            measurement: Observed measurement distribution

        Returns:
            Innovation: y = z - H @ x_pred
        """
        return measurement - self.measurement_model @ state_prediction

    def update(self, state_prediction: State, measurement: Measurement) -> State:
        innovation = self.innovation(state_prediction, measurement)
        L_pred = state_prediction.covariance.cholesky_factor  # (n, n)
        L_R: FloatArray
        if isinstance(measurement.covariance, np.ndarray):
            L_R = np.linalg.cholesky(measurement.covariance)
        else:
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

        R = np.linalg.qr(A.mT, mode="r")  # upper-tri (..., m+n, m+n)
        B = R.mT  # lower-tri

        L_S = B[..., :m, :m]
        KLS = B[..., m:, :m]
        L_post = B[..., m:, m:]

        # K via triangular solve (cheap)
        # We have KLS = K @ L_S, so K = KLS @ L_S^(-1)
        # Solve L_S.T @ X.T = KLS.T for X, which gives X = KLS @ L_S^(-1)
        K = scipy.linalg.solve_triangular(L_S.mT, KLS.mT, lower=False).mT

        # mean update
        posterior_mean = state_prediction.mean + K @ innovation.mean

        return GaussianRV(posterior_mean, CholeskyFactorCovariance(L_post))


def square_root_quadratic_update(
    P1: CholeskyFactorCovariance,
    A1: FloatArray,
    P2: CholeskyFactorCovariance,
    A2: FloatArray,
) -> CholeskyFactorCovariance:
    """Efficiently computes A1 @ P1 @ A1.T + A2 @ P2 @ A2.T

    By utilizing an efficient QR decomposition on the cholesky factors
    of the matrices of P1 and P2.

    Let L1 and L2 be the cholesky factors of P1 and P2 respectively.
    Then A1 @ P1 @ A1.T + A2 @ P2 @ A2.T
    =(A1 @ L1) @ (A1 @ L1).T + (A2 @ L2) @ (A2 @ L2).T
    = [A_1 L_1, A_2 L_2] [A_1 L_1, A_2 L_2]^T
    = M M^T

    and

    M^T = QR
    MM^T = (QR)^T(QR) = R^T R
    so R^T = M and since R^T is lower triangular and
    R^T R = A1 @ P1 @ A1.T + A2 @ P2 @ A2.T

    then R^T is the cholesky factor of the result.

    Args:
        P1: The first CholeskyFactorCovariance.
        A1: The first matrix in the quadratic update.
        P2: The second CholeskyFactorCovariance.
        A2: The second matrix in the quadratic update.

    Returns:
        The CholeskyFactorCovariance of the result.
    """

    W_1 = P1.quadratic_form(A1)
    W_2 = P2.quadratic_form(A2)
    M = np.concatenate([W_1.cholesky_factor, W_2.cholesky_factor], axis=-1)

    _, R = np.linalg.qr(M.mT, mode="reduced")

    return CholeskyFactorCovariance(R.mT)
