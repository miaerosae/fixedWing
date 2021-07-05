import numpy as np
import matplotlib.pyplot as plt

from fw.model.aircraft import F16
from fw.agents.lqr import clqr, lqi_design

from fym.core import BaseEnv, BaseSystem
import fym.logging


class Env(BaseEnv):
    def __init__(self, long, euler, omega, pos, POW, u):
        super().__init__(dt=0.01, max_t=10)
        self.x0 = np.vstack((long, euler, omega, pos, POW))
        self.plant = F16(long, euler, omega, pos, POW)

        self.A, self.B, *_ = self.plant.lin_mode(self.x0, u)
        Q = np.diag((1, 100, 10, 100))
        R = np.diag((1, 100))
        self.K = clqr(self.A, self.B, Q, R)

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
        xlon = np.vstack((x[0], x[1], x[4], x[7]))
        x0 = self.x0
        x0lon = np.vstack((x0[0], x0[1], x0[4], x0[7]))
        u = -self.K.dot(xlon - x0lon)
        u[0] = np.clip(u[0], 0, 1)
        u[1] = np.clip(u[1], self.plant.control_limits["dele"].min(),
                       self.plant.control_limits["dele"].max())
        u_plant = np.vstack((u[0], u[1], 0, 0))
        self.plant.set_dot(t, u_plant)

        return dict(t=t, x=self.plant.observe_dict(), u=u)


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
    plt.figure()

    plt.plot(data["t"], data["u"][:, 0, 0], label="delt")
    plt.plot(data["t"], data["u"][:, 1, 0], label="dele")
    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    long = np.vstack((502., 2.39110108e-1, 0.))
    euler = np.vstack((0., 2.39110108e-1, 0.))
    omega = np.vstack((0., 0., 0.))
    pos = np.vstack((0., 0., 0.))
    POW = 6.41323e+1
    u = np.vstack((0.835, 0.43633231, -0.37524579, 0.52359878))
    # run(long, euler, omega, pos, POW, u)
    exp1_plot()
