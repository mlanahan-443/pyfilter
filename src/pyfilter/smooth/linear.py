from __future__ import annotations

from typing import Any

import equinox as eqx
from jax import numpy as jnp
from jax import scipy as jscipy

from pyfilter.hints.jax_hints import JaxFloatArray
from pyfilter.models.linear import LinearTransformBase, LinearTransitionBase
from pyfilter.types.process_noise import ProcessNoise
from pyfilter.types.random_variables import GaussianRV

type Variable = GaussianRV[Any]


class LinearGaussianFixedPointSmoother(eqx.Module):
    """Fixed point smoother under linear gaussian state dynamics and measuremnt model.

    Based on Simon, 2006 Page 269 Equations 9.24-9.25."""

    transition_model: LinearTransitionBase[GaussianRV[Any]]
    process_noise: ProcessNoise
    measurement_model: LinearTransformBase[GaussianRV[Any]]

    @staticmethod
    def initialize_fixed_point(
        x_init: GaussianRV, t_init: JaxFloatArray
    ) -> tuple[GaussianRV, GaussianRV, JaxFloatArray, JaxFloatArray]:
        """Helper function for initializing the fixed point smoother.

        Parameters
        ----------
        x_init : GaussianRV
            The initial state estimate x(0|0) generated using a kalman filter.
        t_init : JaxFloatArray
            The initial time.

        Returns
        -------
        tuple[GaussianRV, GaussianRV, JaxFloatArray, JaxFloatArray]
            A tuple containing the initialized fixed point smoother variables:
                - x_smoothed - Smoothed estimate.
                - x_update - Kalman filter estimate.
                - sigma - cross covariance of smoothed and update estimate.
                - time - the time of the fixed point.
        """
        return (x_init, x_init, x_init.covariance, t_init)

    def update(
        self,
        x_smoothed_previous: GaussianRV,
        x_filtered_previous: GaussianRV,
        sigma_previous: JaxFloatArray,
        prev_time: JaxFloatArray,
        measurement: GaussianRV,
        time: JaxFloatArray,
    ) -> tuple[GaussianRV, GaussianRV, JaxFloatArray]:
        """Update for the fixed point estimate.

        Parameters
        ----------
        x_smoothed_previous : GaussianRV
            The previous smoothed estimate.
        x_filtered_previous : GaussianRV
            The previous kalman filter estimate.
        sigma_previous : JaxFloatArray
            The previous cross covariacnce between smoothed and filtered estimate.
        prev_time : JaxFloatArray
            The previous time for the estimates ``x_previous``,``sigma_previous``.
        measurement : GaussianRV
            The new measurement.
        time : JaxFloatArray
            The time of the new measurement.

        Returns
        -------
        tuple[GaussianRV, GaussianRV, JaxFloatArray]
            - x_smoothed: The updated smoothed estimate of the fixed point.
            - x_filtered: The updated kalman filter estimate.
            - sigma: The updated cross covariance between smoothed and filtered estimate.
        """
        dt = time - prev_time

        F, Q, H, R, P_prev = (
            self.transition_model.matrix(dt),
            self.process_noise.covariance(dt),
            self.measurement_model.matrix,
            measurement.covariance,
            x_filtered_previous.covariance,
        )

        # Common computed terms: Predicted measurement mean and residual mean.
        z_pred = self.measurement_model.transform_array(x_filtered_previous.mean)
        resid_pred = measurement.mean - z_pred

        # Intermediate variables.
        PHt = P_prev @ H.mT  # PH^T
        S = H @ PHt + R  # measurement covariance HPH^T + R
        cS = jscipy.linalg.cho_factor(S)  # get the cholesky factor of the measurement covariance.
        W = sigma_previous @ H.mT  # Interemdiate term for smoother kalman gain

        # both gains from one triangular solve
        X = jnp.concatenate([F @ PHt, W], axis=-2)  # (2n, m)
        # Simon, 9.18,9.20
        L, lam = jnp.split(jscipy.linalg.cho_solve(cS, X.mT).mT, 2, axis=-2)

        # Update variables: Simon 9.25
        FminusLH = F - L @ H
        B = jnp.concatenate([sigma_previous, F @ P_prev], axis=-2) @ FminusLH.mT

        # Get update for smoothed + filtered covariance.
        sigma_new, FP_FminusLH_T = jnp.split(B, 2, axis=-2)
        P_update = FP_FminusLH_T + Q
        Pi_new = x_smoothed_previous.covariance - lam @ W.mT

        x_smoothed_update = x_smoothed_previous.mean + lam @ resid_pred
        x_update = self.transition_model.transform_array(x_filtered_previous.mean, dt) + L @ resid_pred

        return (
            GaussianRV(mean=x_smoothed_update, covariance=Pi_new),
            GaussianRV(mean=x_update, covariance=P_update),
            sigma_new,
        )
