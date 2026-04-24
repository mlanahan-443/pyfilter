import time

import numpy as np


def batched_matmul(n: int, m: int):
    A = np.random.random(size=(n, m, m))
    x = np.random.random(size=(m, m))

    start = time.time()
    A @ x
    end = time.time()

    return end - start


def main():
    batch_dims = 10 ** np.arange(1, 8, dtype=int)
    for n in batch_dims:
        comp_time = batched_matmul(n, 9)
        print(
            f"n = {n}, time = {comp_time} [s], processing rate = {n * 1e3 / comp_time} [kHz]"
        )


if __name__ == "__main__":
    main()
