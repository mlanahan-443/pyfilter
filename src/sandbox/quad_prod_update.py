import numpy as np
from numpy.random import default_rng
import scipy.linalg
from scipy.sparse import dia_matrix
import scipy


def main():
    N = 10_000
    n = 9

    generator = default_rng(seed=100)
    L_1 = generator.normal(scale=100, size=(N, n, n))
    P = np.einsum("...ik,...jk->...ij", L_1, L_1)

    L_q = generator.normal(scale=100, size=(N, n, n))
    Q = np.einsum("...ik,...jk->...ij", L_q, L_q)

    d = np.ones([2, n])
    d[1, :] *= 0.01
    F = dia_matrix((d, (0, 1)), shape=(n, n))
    F = np.broadcast_to(F.toarray(), Q.shape)

    A = np.concatenate([np.einsum("...ij,...jk->...ik", F, L_1), L_q], axis=2)

    Q_qr, R = scipy.linalg.qr(A.transpose([0, 2, 1]), mode="economic")
    L_qr = R.transpose([0, 2, 1])

    P_update = F @ P @ F.transpose([0, 2, 1]) + Q
    P_qr = np.einsum("...ik,...jk->...ij", L_qr, L_qr)

    print(np.allclose(P_update, P_qr))


if __name__ == "__main__":
    main()
