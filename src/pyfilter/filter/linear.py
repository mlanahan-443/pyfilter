from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from pyfilter.models.linear import LinearTransformBase, LinearTransitionBase
from pyfilter.types.covariance import CholeskyFactorCovariance, CovarianceBase

from ..hints import FloatArray
from ..linear_solve import solve_symmetric_cholesky
from ..types.process_noise import ProcessNoise
from ..types.random_variables import GaussianRV

type Covariance = FloatArray | CholeskyFactorCovariance
type Variable = GaussianRV[Any]


class KalmanFilter[State, Time](Protocol):
    def predict(self, current_state: State, dt: Time) -> State: ...

    def update(self, state_prediction: State, residual: State) -> Time: ...

    def innovation(self, state_prediction: State, measurement: State) -> State: ...


@dataclass
class LinearGaussianKalman[
    State: GaussianRV[Any],
    Measurement: GaussianRV[Any],
    Time: FloatArray,
]:
    """Base linear Gaussian Kalman filter."""

    transition_model: LinearTransitionBase[State, Time]
    process_noise: ProcessNoise
    measurement_model: LinearTransformBase[State]

    def predict(self, current_state: State, dt: Time) -> State:
        return self.transition_model.transform(current_state, dt) + self.process_noise(
            dt
        )

    def update(
        self, state_prediction: State, innovation: Measurement
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

        # The measurement prediction: z ~ N(H @ x_pred, S) where S = innovation.covariance
        # Since innovation.mean = z_obs - H @ x_pred, we have z_obs = innovation.mean + H @ x_pred
        H = self.measurement_model.matrix
        predicted_measurement_mean = H @ state_prediction.mean
        measurement_value = innovation.mean + predicted_measurement_mean

        # Create the predicted measurement distribution
        predicted_measurement = GaussianRV(
            predicted_measurement_mean, innovation.covariance
        )

        # Compute conditional: x | z = z_obs
        return state_prediction.conditional(
            predicted_measurement, cross_covariance, given_value=measurement_value
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

    def update_with_gain(
        self, state_prediction: State, innovation: Measurement
    ) -> GaussianRV[Any]:
        """Update state using classical Kalman gain formulation.

        This is an alternative to the conditional-based update() method,
        implementing the classical Kalman filter equations:
        - K = P @ H.T @ S^(-1)
        - x_post = x_pred + K @ y
        - P_post = P - K @ S @ K.T

        Args:
            state_prediction: Prior state distribution x ~ N(x_pred, P)
            innovation: Innovation distribution y ~ N(y_obs, S)

        Returns:
            Posterior state distribution x | z ~ N(x_post, P_post)
        """
        H = self.measurement_model.matrix
        P = state_prediction.covariance
        S = innovation.covariance  # S = H @ P @ H.T + R
        y = innovation.mean

        # Compute cross-covariance: Cov(x, z) = P @ H.T
        if isinstance(P, CovarianceBase):
            P_HT = P.full() @ H.T
        else:
            P_HT = P @ H.T

        # Compute Kalman gain: K = P @ H.T @ S^(-1)
        # We solve: S @ K.T = (P @ H.T).T for K.T, then transpose
        if isinstance(S, CovarianceBase):
            K = solve_symmetric_cholesky(S, P_HT.swapaxes(-1, -2)).swapaxes(-1, -2)
        else:
            K = solve_symmetric_cholesky(S, P_HT.swapaxes(-1, -2)).swapaxes(-1, -2)

        # Updated mean: x_post = x_pred + K @ y
        updated_mean = state_prediction.mean + np.einsum("...ij,...j->...i", K, y)

        # Updated covariance: P_post = P - K @ S @ K.T
        if isinstance(P, CovarianceBase):
            P_full = P.full()
        else:
            P_full = P

        if isinstance(S, CovarianceBase):
            S_full = S.full()
        else:
            S_full = S

        # K @ S @ K.T
        temp = np.einsum("...ij,...jk->...ik", K, S_full)
        K_S_KT = np.einsum("...ij,...kj->...ik", temp, K)
        updated_cov = P_full - K_S_KT

        # Return with same covariance type as input if possible
        if isinstance(P, CholeskyFactorCovariance):
            from pyfilter.types.covariance import cholesky_factor

            return GaussianRV(updated_mean, cholesky_factor(updated_cov))
        else:
            return GaussianRV(updated_mean, updated_cov)


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

    _, R = np.linalg.qr(M.swapaxes(-1, -2), mode="reduced")

    return CholeskyFactorCovariance(R.swapaxes(-1, -2))
