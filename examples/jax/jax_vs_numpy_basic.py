import time

import jax
import jax.numpy as jnp
import rich

from pyfilter.gutil import LineProfiler


def f(x):  # function we're benchmarking (works in both NumPy & JAX)
    return x.T @ (x - x.mean(axis=0))

def basic_comparison():
    """Copied from Jax website."""
    x_np = np.ones((1000, 1000), dtype=np.float64)  # same as JAX default dtype



    np_profile = LineProfiler("NumPy")
    np_profile.timeit(lambda: f(x_np),repeat = 20)
    rich.print(np_profile)

    x_jax = jnp.array(x_np)
    f_jit = jax.jit(f)
    start = time.time()
    f_jit(x_jax).block_until_ready()  # measure JAX compilation time
    end = time.time()
    print(f"Jax compilation time: {round((end - start)*1e3,2)} ms")

    jax_profile = LineProfiler("Jax")
    jax_profile.timeit(lambda: f_jit(x_jax).block_until_ready(),repeat = 20)
    rich.print(jax_profile)

def recursive_comparison():

    jnp.linalg.solve()


if __name__ == "__main__":
    basic_comparison()
