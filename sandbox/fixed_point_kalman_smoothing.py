"""Development of a fixed point kalman smoother using jax."""
from pyfilter.filter.linear import SquareRootLinearGuassianKalman
from pyfilter.hints.jax_hints import JaxFloatArray
from jax import numpy as jnp
import jax

from jax import numpy as jnp
import matplotlib.pyplot as plt

from pyfilter.models.linear import (
    GaussianSelectionTransform, IntegratorChainTransition, GenericLinearTransform, LinearTransitionBase
)
from pyfilter.types.process_noise import ProcessNoise
import jax

from dataclasses import dataclass

from pyfilter.types.covariance import CholeskyFactorCovariance, DiagonalCovariance
from pyfilter.types.process_noise import WeinerProcessNoise
from pyfilter.types.random_variables import GaussianRV 
from jax import scipy as jscipy

@dataclass
class Simulation:
    """Simulation container."""
    n: int
    dt: float
    sigma: float

    @property
    def generating_transition(self) ->IntegratorChainTransition:
        return IntegratorChainTransition(n=1, p=2)

    @property
    def time_step(self) -> jnp.ndarray:
        return jnp.array(self.dt)

    def scan(self,x: jnp.ndarray,xs) -> tuple[jnp.ndarray,jnp.ndarray]:
        xnew = self.generating_transition.transform(x,self.time_step)
        return xnew,xnew
    
    def __call__(
            self,
            x0: jnp.ndarray,
            key: jnp.ndarray
    ) -> tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        """Do the simulation."""
        dt = jnp.repeat(self.time_step,self.n + 1)
        time = jnp.cumsum(dt)
        _,x = jax.lax.scan(self.scan,x0,length = len(time))
        noise = jax.random.normal(key, shape=(self.n + 1, 2))*self.sigma

        return time, x, x + noise

@dataclass
class FilterScan:

    filter: SquareRootLinearGuassianKalman

    def step(self,state, xs):
        measurement, dt = xs
        update = self.filter.step_update(state, measurement, dt)
        return update, update

    def __call__(
        self,
        init_state : GaussianRV,
        measurements : GaussianRV,
        time_steps: jnp.ndarray
    ) -> GaussianRV:
        _, estimate_sq = jax.lax.scan(self.step, init_state, (measurements, time_steps))
        return estimate_sq
    
def linear_pred_filter(
        x_init: JaxFloatArray,
        P_init: JaxFloatArray,
        intensity: float,
        z: JaxFloatArray,
        R: JaxFloatArray,
        dt: float
) -> tuple[JaxFloatArray,JaxFloatArray]:
    square_root_filter = SquareRootLinearGuassianKalman(
        IntegratorChainTransition(n=1, p=2),
        WeinerProcessNoise(1, 2,intensity),
        GaussianSelectionTransform(slice(0, 1), 2),
    )
    L = jnp.linalg.cholesky(P_init)
    state = GaussianRV(
        x_init.squeeze(),CholeskyFactorCovariance(L)
    )
    time_steps = jnp.repeat(jnp.array(dt),len(z))
    measurements = GaussianRV(z,DiagonalCovariance(R))
    fscan = FilterScan(square_root_filter)
    estimate_sq = fscan(state,measurements,time_steps)
    state_pred = square_root_filter.transition_model.transform(estimate_sq[-1:],jnp.array(dt))
    return state_pred.mean,state_pred.covariance.full()
    

def smooth_fp(
    z: JaxFloatArray,
    R: JaxFloatArray,
    x_init: JaxFloatArray,
    P_init: JaxFloatArray,
    H: JaxFloatArray,
    F: JaxFloatArray,
    Q: JaxFloatArray
) -> tuple[JaxFloatArray,JaxFloatArray]:
    """Fixed point smoothing of x.

    Args:
        z: Measurements[j] at time[j], j = 0,...,N
        R: The measurement covariance.
        x_init: The initial filter mean estimate.
        P_init: The initial filter covariance estimate.
        H: The measurement equation.
        F: The state dynamic equation.
        Q: The process noise of the state prediction.

    Returns:
        A tuple containing series for measurements j = 0,..,N-1:
            1. The smoothed mean estimate
            2. The state apriori mean estimate 
            3. The state apriori covariance estimate
            4. The state covariance smoothed estimate
    """
    def _step(
        state: tuple[JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray],
        carry: tuple[JaxFloatArray,JaxFloatArray,JaxFloatArray]
    ) -> tuple[
        tuple[JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray],
        tuple[JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray]
    ]:
        
        x_s,x_prev,P_prev,Pi_prev,Sigma_prev = state
        z,R,Q  = carry

        #Common computed terms.
        z_pred = H @ x_prev
        S_pred = H @ P_prev @ H.mT + R
        S_pred_inv = jnp.linalg.inv(S_pred)
        resid_pred = z - z_pred
        
        #Intermediate scan variables.
        L = F @ P_prev @ H.mT @ S_pred_inv
        lam = Sigma_prev @ H.mT @ S_pred_inv
        
        #Update scan variables.
        x_smoothed = x_s + lam @ resid_pred
        x_update = F @ x_prev + L @ resid_pred
        P_update = F @ P_prev @ (F - L @ H).mT + Q
        Pi_new = Pi_prev - Sigma_prev @ H.mT @ lam.mT
        Sigma_new = Sigma_prev  @ (F - L @ H).mT

        out = (x_smoothed,x_update,P_update,Pi_new,Sigma_new)
        return out, out
    
    
    init_state = (x_init,x_init,P_init,P_init,P_init)
    carry = (z,R,Q)

    _,result= jax.lax.scan(_step, init_state,carry)

    return tuple([
        jnp.concatenate([v_init[jnp.newaxis,:],v_scan],axis = 0) for v_init,v_scan in zip(init_state[[0,3]],result[[0,3]],strict = True)
    ])


def smooth_fp_efficient(
    z: JaxFloatArray,
    R: JaxFloatArray,
    x_init: JaxFloatArray,
    P_init: JaxFloatArray,
    H: JaxFloatArray,
    F: JaxFloatArray,
    Q: JaxFloatArray
) -> tuple[JaxFloatArray,JaxFloatArray]:
    """Fixed point smoothing of x.

    Args:
        z: Measurements[j] at time[j], j = 0,...,N
        R: The measurement covariance.
        x_init: The initial filter mean estimate.
        P_init: The initial filter covariance estimate.
        H: The measurement equation.
        F: The state dynamic equation.
        Q: The process noise of the state prediction.

    Returns:
        A tuple containing series for measurements j = 0,..,N-1:
            1. The smoothed mean estimate
            2. The state apriori mean estimate 
            3. The state apriori covariance estimate
            4. The state covariance smoothed estimate
    """
    def _step(
        state: tuple[JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray],
        carry: tuple[JaxFloatArray,JaxFloatArray,JaxFloatArray]
    ) -> tuple[
        tuple[JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray],
        tuple[JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray,JaxFloatArray]
    ]:
        
        x_s,x_prev,P_prev,Pi_prev,Sigma_prev = state
        z,R,Q  = carry

        #Common computed terms.
        z_pred = H @ x_prev
        S_pred_T = (H @ P_prev @ H.mT + R).mT
        resid_pred = z - z_pred
        
        #Intermediate scan variables.
        L = jscipy.linalg.solve(S_pred_T,H @ P_prev @ F.mT,assume_a = "sym").mT
        lam = jscipy.linalg.solve(S_pred_T, H @ Sigma_prev.mT,assume_a = "sym").mT
        
        #Update scan variables.
        FminusLH_T = (F - L @ H).mT
        x_smoothed = x_s + lam @ resid_pred
        x_update = F @ x_prev + L @ resid_pred
        P_update = F @ P_prev @ FminusLH_T + Q
        Pi_new = Pi_prev - Sigma_prev @ (lam @ H).mT
        Sigma_new = Sigma_prev  @ FminusLH_T

        out = (x_smoothed,x_update,P_update,Pi_new,Sigma_new)
        return out, out
    
    
    init_state = (x_init,x_init,P_init,P_init,P_init)
    carry = (z,R,Q)

    _,result= jax.lax.scan(_step, init_state,carry)

    return tuple([
        jnp.concatenate([v_init[jnp.newaxis,:],v_scan],axis = 0) for v_init,v_scan in zip(init_state[[0,3]],result[[0,3]],strict = True)
    ])


def smooth_fp_grv(
    measurements: GaussianRV,
    timesteps: JaxFloatArray,
    x_init: GaussianRV,
    measurement_model: GenericLinearTransform,
    transition_model: LinearTransitionBase,
    process_noise: ProcessNoise
) -> tuple[JaxFloatArray,JaxFloatArray]:
    
    
    def _step(
        state: tuple[GaussianRV,GaussianRV,JaxFloatArray],
        carry: tuple[GaussianRV,JaxFloatArray]
    ) -> tuple[
        tuple[GaussianRV,GaussianRV,JaxFloatArray],
        tuple[GaussianRV,GaussianRV,JaxFloatArray],
    ]:
        
        x_smoothed,x_prev, Sigma_prev = state
        z,dt = carry

        F, Q, H, R, P_prev = (transition_model.matrix(dt), process_noise.covariance(dt),
                        measurement_model.matrix, z.covariance, x_prev.covariance)
    
        #Common computed terms.
        z_pred = measurement_model.transform_array(x_prev.mean)
        resid_pred = z.mean - z_pred
        
        #Intermediate scan variables.
        PHt = P_prev @ H.mT                              # reused by S and L
        S   = H @ PHt + R
        cS  = jscipy.linalg.cho_factor(S)                          # factor ONCE, not twice
        W   = Sigma_prev @ H.mT                           # reused by lam and Pi

        # both gains from one triangular solve
        X        = jnp.concatenate([F @ PHt, W], axis=-2)        # (2n, m)
        L, lam   = jnp.split(jscipy.linalg.cho_solve(cS, X.mT).mT, 2, axis=-2)
        
        #Update scan variables.
        FminusLH = F - L @ H
        B = jnp.concatenate([Sigma_prev,F @ P_prev],axis = -2) @ FminusLH.mT
        Sigma_new,FP_FminusLH_T = jnp.split(B,2,axis = -2)
        P_update =  FP_FminusLH_T + Q
        Pi_new = x_smoothed.covariance - lam @ W.mT

        x_smoothed_update = x_smoothed.mean + lam @ resid_pred
        x_update = transition_model.transform(x_prev.mean,dt) + L @ resid_pred

        out = (
            GaussianRV(mean = x_smoothed_update, covariance = Pi_new),
            GaussianRV(mean = x_update,covariance = P_update),
            Sigma_new
        )
        return out, out
    
    
    init_state = (x_init,x_init,x_init.covariance)
    carry = (measurements,timesteps)

    _,result= jax.lax.scan(_step, init_state,carry)

    mean = jnp.concatenate([x_init.mean[jnp.newaxis,...],result[0].mean],axis = 0)
    covariance = jnp.concatenate([x_init.covariance[jnp.newaxis,...],result[0].covariance],axis = 0)
    return mean, covariance




def main():

    meas_model = GaussianSelectionTransform(slice(0, 1), 2)
    H = meas_model.matrix
    x0 = jnp.array([0.0,-1.])
    simulation = Simulation(30,1.0,1.0)
    F = simulation.generating_transition.matrix(jnp.array(simulation.dt))
    key = jax.random.key(45)
    n_sim = 50

    estimatation_error = []

    for iter_key in jax.random.split(key,n_sim):
        time, truth, x = simulation(x0,iter_key)
    
        #Prior for the initial x.

        # Sample from the initial prior
        init_key = jax.random.split(iter_key)[0]
        intensity = 1.0
        P_init = jnp.eye(x0.shape[0])*intensity
        x_init = jax.random.multivariate_normal(init_key,x0,P_init)
        
        #Setup the measurements
        measurements = jnp.einsum("ij,...j->...i",H,x)
        R = jnp.repeat(jnp.array([[simulation.sigma**2]]),len(measurements),axis = 0)
        process_noise_model = WeinerProcessNoise(1,2,intensity) 

        fs_break = 15
        z_to_filter,z_to_smooth = measurements[:fs_break],measurements[fs_break:]
        R_to_filter, R_to_smooth = R[:fs_break],R[fs_break:]

        #Run KF until breakpoint
        x_filter,P_filter = linear_pred_filter(x_init,P_init,intensity,z_to_filter,R_to_filter,simulation.dt)

        #Get smoothed estimate
        meas_rv = GaussianRV(z_to_smooth,R_to_smooth)
        x_init_rv = GaussianRV(x_filter.squeeze(axis = 0),P_filter.squeeze(axis = 0))
        time_steps = jnp.repeat(simulation.dt,len(meas_rv.mean))
        x_smoothed,P_smoothed = smooth_fp_grv(
            meas_rv,time_steps,x_init_rv,meas_model,simulation.generating_transition,process_noise_model
        )
    

        error = jnp.linalg.norm(truth[fs_break:fs_break + 1] - x_smoothed,axis = -1)**2 
        estimatation_error.append(error[jnp.newaxis,...])
        cov_improvement = jnp.linalg.trace(P_filter - P_smoothed)/jnp.linalg.trace(P_filter)
        tr_P = jnp.linalg.trace(P_smoothed)


    error_arr = jnp.concatenate(estimatation_error,axis = 0 )
    error_mean = error_arr.mean(axis = 0)
    error_std = error_arr.std(axis = 0)

    fig,ax = plt.subplots(figsize = (8,5))

    steps = jnp.arange(len(error))

    upper = error_mean + 2*error_std
    lower = error_mean - 2*error_std
    ax.plot(steps,error_mean/error_mean[0],lw = 1.5,color = "red",label = r"$||\hat{\varepsilon_s}||_2^2$")
    ax.plot(steps,upper/error_mean[0],lw = 0.75,color = "red")
    ax.plot(steps,lower/error_mean[0],lw = 0.75,color = "red")
    ax.fill_between(steps,lower/error_mean[0],upper/error_mean[0],alpha = 0.2,color = 'red')
    ax.plot(steps,tr_P/tr_P[0],lw = 1.5,color = "k",label = r"$tr(P)$")
    ax.plot(steps, cov_improvement,lw = 1.5,color = "k",ls = '--',label = r"$tr(P - P_{smoothed})/tr(P)$")

    ax.set_xlabel("Time Steps",fontsize = 12)
    ax.set_ylabel("Normalized Estimate Errors (Actual, Estimated)",fontsize = 12)
    ax.legend(fontsize = 12)

    #fig.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    main()



