import numpy as np
import matplotlib.pyplot as plt

from fw.model.aircraft import F16
from fw.agents.lqr import clqr, lqi_design

from fym.core import BaseEnv, BaseSystem
import fym.logging


class Env(BaseEnv):
    def __init__(self, *x, u):
        super().__init__(dt=0.001, max_t=10)
        self.x0 = x
        self.plant = F16(*x, u)

        self.A, self.B = self.plant.lin_mode(*x, u)
        Q = np.diag((1, 100, 10, 100))
        R = np.array((1, 100))
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
        ref = self.get_ref(t)
        u = -self.K.dot(x - self.x0)
        self.plant.set_dot(t, u)

        return dict(t=t, x=self.plant.observe_dict(), ref=ref)


def run(*x, u):
    env = Env(*x, u)
    env.logger = fym.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


if __name__ == "__main__":
    long = np.vstack((502., 2.39110108e-1, 0.))
    euler = np.vstack((0., 2.39110108e-1, 0.))
    omega = np.vstack((0., 0., 0.))
    pos = np.vstack((0., 0., 0.))
    POW = 6.41323e+1
    u = np.vstack((0.835, 0.43633231, -0.37524579, 0.52359878))
    run(long, euler, omega, pos, POW, u)
