from __future__ import annotations

from typing import Any

import equinox as eqx
from jax import numpy as jnp
from jax import scipy as jscipy

from pyfilter.hints.jax_hints import JaxFloatArray
from pyfilter.models.linear import LinearTransformBase, LinearTransitionBase
from pyfilter.types.process_noise import ProcessNoise
from pyfilter.types.random_variables import GaussianRV
from pyfilter.types import Covariance


class LinearGaussianFixedPointSmoother(eqx.Module):
    """Fixed point smoother under linear gaussian state dynamics and measuremnt model.

    Based on Simon, 2006 Page 269 Equations 9.24-9.25."""

    transition_model: LinearTransitionBase[GaussianRV[JaxFloatArray]]
    process_noise: ProcessNoise
    measurement_model: LinearTransformBase[GaussianRV[JaxFloatArray]]

    @staticmethod
    def initialize_fixed_point(
        x_init: GaussianRV[JaxFloatArray], t_init: JaxFloatArray
    ) -> tuple[GaussianRV[JaxFloatArray], GaussianRV[JaxFloatArray], JaxFloatArray, JaxFloatArray]:
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
        x_smoothed_previous: GaussianRV[JaxFloatArray],
        x_filtered_previous: GaussianRV[JaxFloatArray],
        sigma_previous: JaxFloatArray,
        prev_time: JaxFloatArray,
        measurement: GaussianRV[Covariance],
        time: JaxFloatArray,
    ) -> tuple[GaussianRV[JaxFloatArray], GaussianRV[JaxFloatArray], JaxFloatArray]:
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
        z_pred = self.measurement_model.transform_array(x_filtered_previous.mean[..., jnp.newaxis]).squeeze(
            axis=-1
        )
        resid_pred = measurement.mean - z_pred

        # Intermediate variables.
        PHt = P_prev @ H.mT  # PH^T
        S = H @ PHt + R  # measurement covariance HPH^T + R
        cS = (
            jscipy.linalg.cho_factor(S) if isinstance(S, jnp.ndarray) else S.cholesky_factor
        )  # get the cholesky factor of the measurement covariance.
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

        x_smoothed_update = x_smoothed_previous.mean + jnp.squeeze(
            lam @ resid_pred[..., jnp.newaxis], axis=-1
        )
        x_update = self.transition_model.transform_array(
            x_filtered_previous.mean[..., jnp.newaxis], dt
        ).squeeze(axis=-1) + jnp.squeeze(L @ resid_pred[..., jnp.newaxis], axis=-1)

        return (
            GaussianRV(mean=x_smoothed_update, covariance=Pi_new),
            GaussianRV(mean=x_update, covariance=P_update),
            sigma_new,
        )
