from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import pyfilter.config as config
from pyfilter.hints import FloatArray


class GaussianKalman(ABC):
    def __init__(
        self,
        init_state: FloatArray,
        init_cov: FloatArray,
        selection: FloatArray,
        *args,
        init_with_predict: bool = True,
        **kwargs,
    ):
        if config.DEBUG_:
            assert init_state.shape[0] == init_cov.shape[0], (
                "the number of rows in state must equal the number of rows in the covariance matrix"
            )
            assert init_cov.ndim == 2, "covariance matrix must be at least 2d"
            assert init_cov.shape[0] == init_cov.shape[1], (
                "initial covariance matrix must be square"
            )
            assert _issym(init_cov), "covariance matrix must be symmetric"

        # keep initial values
        self.init_state = init_state
        self.init_cov = init_cov
        self.n = 0

        # estimated state and variance
        self.est_state = np.empty_like(init_state) * np.nan
        self.est_cov = np.empty_like(init_cov) * np.nan

        # predicted state and variance
        self.pred_state = np.empty_like(init_state) * np.nan
        self.pred_cov = np.empty_like(init_cov) * np.nan

        # selection matrix or function
        self.selection = selection

        # initialize with prediction or not
        self.init_with_predict = init_with_predict

    def select(self, state: FloatArray) -> FloatArray:
        if isinstance(self.selection, np.ndarray):
            return self.selection @ state
        else:
            raise NotImplementedError(
                "Selection must be a linear selection matrix for now"
            )

    def update(
        self,
        measure: FloatArray,
        measure_cov: FloatArray,
        pred_state: FloatArray,
        pred_cov: FloatArray,
        kalman_gain: FloatArray,
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Estimate the current state and state covariance of the system using the
        present measurement (measure)  the measurement covariance (measure_cov)
        the previous prediction (pred_state) and previous state covariance prediction
        (pred_cov) calculated kalman gain (kalman_gain)

        Parameters
        ----------
        measure: FloatArray
            the measurement at time step n
        measure_cov: FloatArray
            the covariance of the measurement at time step n
        pred_state: FloatArray
            the predicted state using n-1 measurements
        pred_cov: FloatArray
            the predicted state covaraicne using n-1 measurements
        kalman_gain: FloatArray
            the computed kalman gain

        Returns
        -------
        Tuple[FloatArray,FloatArray]:
            the estimated state and state covaraince at step n using n measurements
        """

        est_state = pred_state + kalman_gain @ (measure - self.selection @ pred_state)
        B_ = np.eye(kalman_gain.shape[0]) - kalman_gain @ self.selection
        est_cov = B_ @ pred_cov @ B_.T + kalman_gain @ measure_cov @ kalman_gain.T
        return est_state, est_cov

    def kalman_gain(self, pred_cov: FloatArray, measure_cov: FloatArray) -> FloatArray:
        """
        compute the kalman gain of step n based on the predicted covariance at
        step n using n-1 measurements (pred_cov), and the new measurement
        covariance (measure_cov) at step n

        Parameters
        ----------
        pred_cov: FloatArray
            the predicted state covaraicne using n-1 measurements
        measure_cov: FloatArray
            the covariance of the measurement at time step n

        Returns
        ---------
        FloatArray:
            the kalman gain

        NOTE:
        --------
        the computation takes advantage of the symmetric matrices and uses
        scipy's cholesky decomposition, see [`solve_symmetric_cholesky_dense`](#solve_symmetric_cholesky_dense)

        """
        if config.DEBUG_:
            assert _issym(pred_cov), "predicted covariance is not symmetric"
            assert _issym(measure_cov), (
                "provided measurement covariance is not symmetric"
            )

        return solve_symmetric_cholesky_dense(
            self.selection @ pred_cov @ self.selection.T + measure_cov,
            self.select(pred_cov),
            overwrite=True,
        ).T

    @abstractmethod
    def predict(self, *args, **kwargs) -> Tuple[FloatArray, FloatArray]:
        """
        The system dynamical model n+1 prediction of the system state and uncertainty
        using n previous estimates of the system state.

        Parameters
        ----------
        *args: arguments
        **kwargs: key-word arguments

        Returns
        ----------
        Tuple[FloatArray,FloatArray]:
            A tuple containing the predicted state at the first entry and the
            predicted variance at the second entry
        """
        pass

    def initialize(self, process_variance: FloatArray):
        """
        initialize the filter
        """

        self.est_state = self.init_state
        self.est_cov = self.init_cov

        if self.init_with_predict:
            # initialize predictions treating estimates as "0 state"
            self.pred_state, self.pred_cov = self.predict(
                self.est_state, self.est_cov, process_variance
            )
        else:
            # just set predictions to the initial values
            self.pred_state = self.est_state
            self.pred_cov = self.est_cov

    def step(
        self,
        measurement: FloatArray,
        measurement_cov: FloatArray,
        process_cov: FloatArray,
    ) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """
        Using the present measurement of the system state (measurement), the
        measurement covariance (measurement_cov), and the process covaraince
        (process_cov) return the estimated state, the estimated state covariance,
        the predicted state, the predicted covariance, and the kalman gain

        Parameters
        ----------
        measurement: FloatArray
            the state measurement
        measurement_cov: FloatArray
            the covariance in the state measurement
        process_cov: FloatArray
            the process covariance

        Returns:
        --------
        Tuple[FloatArray,FloatArray,FloatArray,FloatArray,FloatArray] :
            a tuple containing:
            (estimated_state,estimated_covariance,predicted_state, predicted_covariance,kalman_gain)
        """

        if np.all(np.isnan(self.pred_state)):
            self.initialize(process_cov)

        kalman_gain = self.kalman_gain(self.pred_cov, measurement_cov)
        self.est_state, self.est_cov = self.update(
            measurement, measurement_cov, self.pred_state, self.pred_cov, kalman_gain
        )

        self.pred_state, self.pred_cov = self.predict(
            self.est_state, self.est_cov, process_cov
        )

        self.n += 1
        return self.est_state, self.est_cov, self.pred_state, self.pred_cov, kalman_gain


class LTI_GaussianKalman(GaussianKalman):
    def __init__(
        self,
        A: FloatArray,
        b: FloatArray,
        init_state: FloatArray,
        init_cov: FloatArray,
        selection: FloatArray,
        init_with_predict=True,
    ):
        if config.DEBUG_:
            assert A.ndim == 2, "A must be a 2d matrix"
            assert b.squeeze().ndim == 1, "b must be a 1d row vector"
            assert b.shape[0] == A.shape[0], "A and b must have the same number of rows"

        super().__init__(
            init_state, init_cov, selection, init_with_predict=init_with_predict
        )

        self.A = A
        self.b = b

    def predict(
        self, est_state: FloatArray, est_cov: FloatArray, process_cov: FloatArray
    ) -> Tuple[FloatArray, FloatArray]:
        # predict the next state based on the current estimated state
        # according to the linear transformation

        pred_state = self.A @ est_state + self.b
        # covariance extrapolation - exact for gaussian random variables
        pred_cov = self.A @ est_cov @ self.A.T + process_cov

        return pred_state, pred_cov
