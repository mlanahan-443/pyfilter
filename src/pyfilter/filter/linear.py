from __future__ import annotations
from dataclasses import dataclass
from ..types.random_variables import GaussianRV
from ..types.process_noise import ProcessNoise
from pyfilter.models.linear_transition import LinearTransitionBase
from pyfilter.models.linear_transform import LinearTransformBase
from ..hints import FloatArray
from pyfilter.types.covariance import CholeskyFactorCovariance
import numpy as np
from typing import Protocol

type Covariance = FloatArray | CholeskyFactorCovariance
type Variable = GaussianRV


class KalmanFilter[State, Time](Protocol):
    def predict(self, current_state: State, dt: Time) -> State: ...

    def update(self, state_prediction: State, residual: State) -> Time: ...

    def innovation(self, state_prediction: State, measurement: State) -> State: ...


@dataclass
class LinearGuassianKalman[
    State: GaussianRV,
    Measurement: GaussianRV,
    Time: FloatArray,
]:
    """Base linear gaussian kalman filter."""

    transition_model: LinearTransitionBase
    process_noise: ProcessNoise
    measurement_model: LinearTransformBase

    def predict(self, current_state: State, dt: Time) -> State:
        return self.transition_model.transform(current_state, dt) + self.process_noise(
            dt
        )

    def update(self, state_prediction: State, innovation: Measurement) -> GaussianRV:
        cross_covariance = state_prediction.linear_cross(self.measurement_model.matrix)
        return state_prediction.conditional(innovation, cross_covariance)

    def innovation(
        self, state_prediction: State, measurement: Measurement
    ) -> Measurement:
        return self.measurement_model @ state_prediction - measurement


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
