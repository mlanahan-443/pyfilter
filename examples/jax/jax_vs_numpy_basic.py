import numpy as np
import jax
import time
import jax.numpy as jnp

import time
import numpy as np
from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import Callable



@dataclass
class Profiler:
    name: str
    _times: list[float] = field(default_factory=list, init=False, repr=False)
    _t0: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._t0 is None:
            raise RuntimeError("Profiler.stop() called before start()")
        self._times.append(time.perf_counter() - self._t0)
        self._t0 = None

    def __enter__(self) -> "Profiler":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def timeit(self, fn: Callable[[], Any], repeat: int = 10) -> "Profiler":
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            self._times.append(time.perf_counter() - t0)
        return self

    def reset(self) -> None:
        self._times.clear()
        self._t0 = None

    @property
    def times(self) -> np.ndarray:
        return np.array(self._times)

    def report(self) -> None:
        t = self.times
        if len(t) == 0:
            print(f"[{self.name}] No trials recorded.")
            return
        print(
            f"[{self.name}] n={len(t)} | "
            f"mean={t.mean()*1e3:.1f}ms | "
            f"std={t.std()*1e3:.1f}ms | "
            f"min={t.min()*1e3:.1f}ms | "
            f"max={t.max()*1e3:.1f}ms"
        )

def f(x):  # function we're benchmarking (works in both NumPy & JAX)
    return x.T @ (x - x.mean(axis=0))

def basic_comparison():
    """Copied from Jax website."""
    x_np = np.ones((1000, 1000), dtype=np.float64)  # same as JAX default dtype



    np_profile = Profiler("NumPy")
    np_profile.timeit(lambda: f(x_np),repeat = 20)
    np_profile.report()

    x_jax = jnp.array(x_np)
    f_jit = jax.jit(f)
    start = time.time()
    f_jit(x_jax).block_until_ready()  # measure JAX compilation time
    end = time.time() 
    print(f"Jax compilation time: {round((end - start)*1e3,2)} ms")

    jax_profile = Profiler("Jax")
    jax_profile.timeit(lambda: f_jit(x_jax).block_until_ready(),repeat = 20)
    jax_profile.report()


def linear_kalman_update[T: jax.Array | NDArray](x: T,P: T,z: T,H: T,R: T, I: T,solve: Callable[[T,T],T]) -> tuple[T,T]:

    HP = H @ P
    S = HP @ H.mT + R
    gain = solve(S,HP).mT
    residual = z - H @ x 
    x_update = x + gain @ residual
    I_minus_WH = I - gain @ H
    P_update = I_minus_WH @ P @ I_minus_WH.mT + gain @ R @ gain.mT
    return x_update,P_update
    

def linear_kalman_prediction(x,P,F,Q):

    x_pred = F @ x
    P_pred =  F@ P @ F.mT + Q

def recursive_comparison():

    jnp.linalg.solve()


if __name__ == "__main__":
    basic_comparison()