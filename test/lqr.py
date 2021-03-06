import numpy as np
import matplotlib.pyplot as plt

from fw.model.fixedWing import F16
from fw.agents.lqr import clqr, lqi_design

from fym.core import BaseEnv, BaseSystem
import fym.logging


class Env(BaseEnv):
    def __init__(self, long, euler, omega, pos, POW, u):
        super().__init__(dt=0.01, max_t=10)
        self.x0 = np.vstack((long, euler, omega, pos, POW))
        self.plant = F16(long, euler, omega, pos, POW)
        self.u = u

        # A, B, *_ = self.plant.lin_mode(self.x0, u)
        # Q = np.diag((1, 100, 10, 100))
        # R = np.diag((1, 100))
        # self.K = clqr(A, B, Q, R)
        # lqi
        # Aaug, Baug = lqi_design(A, B)
        # Qi = np.diag((1, 100, 10, 100, 1, 100))
        # Ri = np.diag((1, 100))
        # self.K = clqr(Aaug, Baug, Qi, Ri)
        # self.e = BaseSystem(np.vstack((0, 0)))

    def step(self):
        *_, done = self.update()
        return done

    # def get_ref(self, t):
    #     long_des = np.vstack((20, 0, 0))
    #     euler_des = np.vstack(())
    #     omega_des = np.vstack((0, 0, 0))
    #     vel_des = np.vstack((0, 0, 300))
    #     POW_des = np.array(())
    #     ref = np.vstack((long_des, euler_des, omega_des, vel_des, POW_des))

    #     return ref

    def set_dot(self, t):
        x = self.plant.state
        x0 = self.x0
        xlon = np.vstack((x[0], x[1], x[4], x[7]))
        x0lon = np.vstack((x0[0], x0[1], x0[4], x0[7]))
        diff = np.vstack((xlon - x0lon))

        # lqi
        # self.e.dot = np.vstack((x[0] - x0[0],
        #                         (x[4] - x0[4]) - (x[1] - x0[1])))
        # diff = np.vstack((diff, self.e.state))

        # u = -self.K.dot(diff)
        # u[0] = np.clip(u[0], 0, 1)
        # u[1] = np.clip(u[1], self.plant.control_limits["dele"].min(),
        #                self.plant.control_limits["dele"].max())
        # u_plant = np.vstack((u[0], u[1], 0, 0))
        # self.plant.set_dot(t, u_plant)
        self.plant.set_dot(t, self.u)

        return dict(t=t, x=self.plant.observe_dict())


def run(long, euler, omega, pos, POW, u):
    env = Env(long, euler, omega, pos, POW, u)
    env.logger = fym.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


def exp1_plot():
    data = fym.logging.load("data.h5")

    # state variables
    plt.figure()

    ax = plt.subplot(511)
    plt.plot(data["t"], data["x"]["long"][:, 0, 0], label="VT")
    plt.plot(data["t"], data["x"]["long"][:, 1, 0], label="alp")
    plt.plot(data["t"], data["x"]["long"][:, 2, 0], label="bet")
    plt.legend()

    plt.subplot(512, sharex=ax)
    plt.plot(data["t"], data["x"]["euler"][:, 0, 0], label="phi")
    plt.plot(data["t"], data["x"]["euler"][:, 1, 0], label="theta")
    plt.plot(data["t"], data["x"]["euler"][:, 2, 0], label="psi")
    plt.legend()

    plt.subplot(513, sharex=ax)
    plt.plot(data["t"], data["x"]["omega"][:, 0, 0], label="p")
    plt.plot(data["t"], data["x"]["omega"][:, 1, 0], label="q")
    plt.plot(data["t"], data["x"]["omega"][:, 2, 0], label="r")
    plt.legend()

    plt.subplot(514, sharex=ax)
    plt.plot(data["t"], data["x"]["pos"][:, 0, 0], label="pn")
    plt.plot(data["t"], data["x"]["pos"][:, 1, 0], label="pe")
    plt.plot(data["t"], data["x"]["pos"][:, 2, 0], label="pd")
    plt.legend()

    plt.subplot(515, sharex=ax)
    plt.plot(data["t"], data["x"]["POW"][:, 0], label="POW")
    plt.legend()

    plt.tight_layout()

    # input
    # plt.figure()

    # plt.plot(data["t"], data["u"][:, 0, 0], label="delt")
    # plt.plot(data["t"], data["u"][:, 1, 0], label="dele")
    # plt.legend()

    # plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    long = np.vstack((502, 5.05807418e-02, 7.85459681e-07))
    euler = np.vstack((0., 5.05807418e-02, 0.))
    omega = np.vstack((0., 0., 0.))
    pos = np.vstack((0., 0., 0.))
    POW = 1.00031243e+1
    u = np.vstack((0.154036408, -1.21242062e-2, -2.91493958e-7, 1.86731213e-6))
    run(long, euler, omega, pos, POW, u)
    exp1_plot()
