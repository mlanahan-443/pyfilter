import numba
import numpy as np
import rich
from numpy.random import default_rng
from numpy.typing import NDArray

from pyfilter.gutil import LineProfiler
from pyfilter.models.linear import GaussianSelectionTransform, IntegratorChainTransition
from pyfilter.types.process_noise import WeinerProcessNoise


def linear_kalman_update[T:NDArray[np.floating]](x: T,P: T,z: T,H: T,R: T, I: T) -> tuple[T,T]:

    HP = H @ P
    S = HP @ H.mT + R
    gain = np.linalg.solve(S,HP).mT
    residual = z - H @ x
    x_update = x + gain @ residual
    I_minus_WH = I - gain @ H
    P_update = I_minus_WH @ P @ I_minus_WH.mT + gain @ R @ gain.mT
    return x_update,P_update


def linear_kalman_prediction[T: NDArray[np.floating]](x: T,P:T,F: T,Q: T) -> tuple[T,T]:

    x_pred = F @ x
    P_pred =  F@ P @ F.mT + Q
    return x_pred,P_pred

def run_numpy[Arr: NDArray[np.floating]](
        x0: Arr,
        P0: Arr,
        z: Arr,
        F: Arr,
        H: Arr,
        R: Arr,
        Q: Arr,
) -> tuple[Arr,Arr]:

    x = x0.copy()
    P = P0.copy()
    eye = np.eye(x0.shape[-1])
    x_out = []
    P_out = []
    for i in range(z.shape[0]):
        x_pred,P_pred = linear_kalman_prediction(x,P,F,Q)
        x,P= linear_kalman_update(x_pred, P_pred,z[i],H,R,eye)
        x_out.append(x.copy())
        P_out.append(P.copy())

    return np.concatenate(x_out,axis = 0),np.concatenate(P_out,axis = 0)


@numba.njit(cache=True, fastmath=False, boundscheck=False,nopython = True)
def numbda_kalman_step(
    x: np.ndarray, P: np.ndarray, z: np.ndarray,
    F: np.ndarray, H: np.ndarray, R: np.ndarray, Q: np.ndarray,
    eye: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Predict
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    # Update (Joseph form)
    HP = H @ P_pred
    S = HP @ H.T + R
    gain = np.linalg.solve(S, HP).T
    residual = z - H @ x_pred
    x_new = x_pred + gain @ residual
    A = eye - gain @ H
    P_new = A @ P_pred @ A.T + gain @ R @ gain.T
    return x_new, P_new


@numba.njit(cache=True, fastmath=False, boundscheck=False,nopython = True)
def numbda_filter(
    x0: np.ndarray, P0: np.ndarray, z: np.ndarray,
    F: np.ndarray, H: np.ndarray, R: np.ndarray, Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    T = z.shape[0]
    n = x0.shape[0]
    eye = np.eye(n)
    x = x0.copy()
    P = P0.copy()
    x_out = np.empty((T, n), dtype=x0.dtype)
    P_out = np.empty((T, n, n), dtype=P0.dtype)
    for i in range(T):
        x, P = numbda_kalman_step(x, P, z[i], F, H, R, Q, eye)
        x_out[i] = x
        P_out[i] = P
    return x_out, P_out

def generate_data(
        x0: NDArray[np.floating],
        n: int,
        F: NDArray[np.floating]
    ) -> NDArray[np.floating]:
    x = np.zeros((n + 1,9),dtype = x0.dtype)
    x[0] = x0.copy()
    for i in range(1,n+1):
        x[i] = F @ x[i-1]

    return x


def main():

    x0 = np.array([1,-10,1,-0.15,0.03,1,0.001,0.01,-0.2],dtype = np.float32)
    F = IntegratorChainTransition(n = 3,p = 3).matrix(np.array(0.5))
    H = GaussianSelectionTransform(slice(0,3),9).matrix.copy()
    Q = WeinerProcessNoise(n = 3,p = 3,intensity= np.array(0.1)).covariance(np.array(0.5))

    print(F.shape,H.shape,Q.shape)
    n = 100
    rng = default_rng(seed = 45)
    noise = rng.normal(scale = 0.1,size = (n+1,3)).astype(x0.dtype)
    z = generate_data(x0,n,F)[:,:3] + noise
    R = np.eye(3)*0.1
    P0 = np.eye(9)*10

    numpy_profiler = LineProfiler("NumPy")
    numpy_profiler.timeit(
        lambda: run_numpy(
        x0,P0,z,F,H,R,Q
        ),
        number = 5,
        repeat = 3
    )

    rich.print(numpy_profiler)

    numbda_filter(x0.astype(np.float64),P0.astype(np.float64), z.astype(np.float64), F.astype(np.float64), H.astype(np.float64), R.astype(np.float64), Q.astype(np.float64))

    numba_profiler = LineProfiler("Numba")
    numba_profiler.timeit(
        lambda: numbda_filter(x0.astype(np.float64),P0.astype(np.float64), z.astype(np.float64), F.astype(np.float64), H.astype(np.float64), R.astype(np.float64), Q.astype(np.float64)),
        number =5,
        repeat= 3
    )

    rich.print(numba_profiler)




if __name__ == "__main__":
    main()





