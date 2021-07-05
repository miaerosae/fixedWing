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


# lqi design, augmented vector included
def lqi_design(Alon, Blon):
    Alon[2, :] = Alon[2, :] - Alon[1, :]  # state transformation from theta to gamma
    Blon[2, :] = Blon[2, :] - Blon[1, :]

    # longitudinal state = [VT, alp, gamma, q]
    # performance output = [VT, gamma]
    Elon = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])
    AlonAug1 = np.vstack((Alon, Elon))
    AlonAug2 = np.vstack((np.zeros((4, 2)), np.zeros((2, 2))))
    AlonAug = np.hstack((AlonAug1, AlonAug2))
    BlonAug = np.vstack((Blon, np.zeros((2, 2))))

    return AlonAug, BlonAug


if __name__ == "__init__":
    pass
