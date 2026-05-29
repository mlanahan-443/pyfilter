import jax.scipy
import jax.scipy.linalg
from jax import numpy as jnp
from numpy.random import default_rng
from scipy.sparse import dia_matrix


def main():
    N = 10_000
    n = 9

    generator = default_rng(seed=100)
    L_1 = generator.normal(scale=100, size=(N, n, n))
    P = jnp.einsum("...ik,...jk->...ij", L_1, L_1)

    L_q = generator.normal(scale=100, size=(N, n, n))
    Q = jnp.einsum("...ik,...jk->...ij", L_q, L_q)

    d = jnp.broadcast_to(jnp.asarray([[1, 0.01]]).T, (2, n))
    F = jnp.asarray(dia_matrix((jnp.array(d), (0, 1)), shape=(n, n)).toarray())

    A = jnp.concatenate([jnp.einsum("...ij,...jk->...ik", F, L_1), L_q], axis=2)

    Q_qr, R = jax.scipy.linalg.qr(A.mT, mode="economic")
    L_qr = R.transpose([0, 2, 1])

    P_update = F @ P @ F.mT + Q
    P_qr = jnp.einsum("...ik,...jk->...ij", L_qr, L_qr)

    print(jnp.allclose(P_update, P_qr))


if __name__ == "__main__":
    main()
