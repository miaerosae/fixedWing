import numpy as np
import scipy.linalg as sla
import numpy.linalg as nla


def clqr(A: np.array, B: np.array, Q: np.array, R: np.array) -> np.array:
    P = sla.solve_continuous_are(A, B, Q, R)
    K = nla.inv(R).dot(np.transpose(B)).dot(P)

    return K


def dlqr(A: np.array, B: np.array, Q: np.array, R: np.array) -> np.array:
    P = sla.solve_discrete_are(A, B, Q, R)
    K = nla.inv(R).dot(np.transpose(B)).dot(P)

    return K


if __name__ == "__init__":
    pass
