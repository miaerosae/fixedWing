import numpy as np

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import angle2dcm, quat2dcm


class Mixer:
    '''
    takes force and moment commands and translate them to actuator commands.
    actuator command means the force generated by each rotor
    '''
    def __init__(self, d, c, rtype="quad"):
        self.d = d
        self.c = c

        # quadcopter
        if rtype == "quad":
            self.B = np.array([[1, 1, 1, 1],
                               [0, -d, 0, d],
                               [d, 0, -d, 0],
                               [-c, c, -c, c]])
        # hexacopter
        elif rtype == "hexa":
            self.B = np.array([[1, 1, 1, 1, 1, 1],
                               [-d, d, d/2, -d/2, -d/2, d/2],
                               [0, 0, d*np.sqrt(3)/2, -d*np.sqrt(3)/2,
                                d*np.sqrt(3)/2, -d*np.sqrt(3)/2],
                              [c, -c, c, -c, -c, c]])

        self.Binv = np.linalg.pinv(self.B)

    def inverse(self, rotors):
        return self.B.dot(rotors)

    def __call__(self, forces):
        return self.Binv.dot(forces)


# linearized model
class Copter_linear(BaseEnv):
    '''
    Tayoung Lee, Melvin Leok and N. Harris McClmroch,
    "Control of Complex MAneuvers for a Quadrotor UAV Using Geometric
    Method on SE(3)", 2010
    '''
    J = np.diag((0.0820, 0.0845, 0.1377))  # kg-m^2
    Jinv = np.linalg.inv(J)
    m = 4.34  # kg
    d = 0.315  # m
    c = 8.004e-4  # m
    g = 9.81  # m/s^2
    rotor_max = m * g
    rotor_min = 0

    def __init__(self,
                 pos=np.zeros((3, 1)),
                 vel=np.zeros((3, 1)),
                 angle=np.zeros((3, 1)),
                 omega=np.zeros((3, 1))):
        # angle = [phi, theta, psi].T
        # omega = [dot_phi, dot_theta, dot_psi].T
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.angle = BaseSystem(angle)
        self.omega = BaseSystem(omega)

        self.mixer = Mixer(d=self.d, c=self.c)

    def deriv(self, pos, vel, angle, omega, rotors):
        F, M1, M2, M3 = self.mixer.inverse(rotors)

        M = np.vstack((M1, M2, M3))

        m, g, J = self.m, self.g, self.J
        e3 = np.vstack((0, 0, 1))

        dpos = vel
        dcm = angle2dcm(*angle[::-1])
        dvel = g*e3 - F*dcm.T.dot(e3)/m
        dangle = omega
        domega = self.Jinv.dot(M - np.cross(omega, J.dot(omega), axis=0))

        return dpos, dvel, dangle, domega

    def set_dot(self, t, rotors):
        states = self.observe_list()
        dots = self.deriv(*states, rotors)
        self.pos.dot, self.vel.dot, self.angle.dot, self.omega.dot = dots


# nonlinear model
class Copter_nonlinear(BaseEnv):
    '''
    Tayoung Lee, Melvin Leok and N. Harris McClmroch,
    "Control of Complex MAneuvers for a Quadrotor UAV Using Geometric
    Method on SE(3)", 2010
    '''
    J = np.diag((0.0820, 0.0845, 0.1377))  # kg-m^2
    Jinv = np.linalg.inv(J)
    m = 4.34  # kg
    d = 0.315  # m
    c = 8.004e-4  # m
    g = 9.81  # m/s^2
    rotor_max = m * g
    rotor_min = 0

    def __init__(self,
                 pos=np.zeros((3, 1)),
                 vel=np.zeros((3, 1)),
                 quat=np.vstack((1, 0, 0, 0)),
                 omega=np.zeros((3, 1))):
        super().__init__()
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.quat = BaseSystem(quat)
        self.omega = BaseSystem(omega)

        self.mixer = Mixer(d=self.d, c=self.c)

    def deriv(self, pos, vel, quat, omega, rotors):
        F, M1, M2, M3 = self.mixer.inverse(rotors)

        M = np.vstack((M1, M2, M3))

        m, g, J = self.m, self.g, self.J
        e3 = np.vstack((0, 0, 1))

        dpos = vel
        dcm = quat2dcm(quat)
        dvel = g*e3 - F*dcm.T.dot(e3)/m
        # DCM integration (Note: dcm; I to B) [1]
        _w = np.ravel(omega)
        # unit quaternion integration [4]
        dquat = 0.5 * np.array([[0., -_w[0], -_w[1], -_w[2]],
                                [_w[0], 0., _w[2], -_w[1]],
                                [_w[1], -_w[2], 0., _w[0]],
                                [_w[2], _w[1], -_w[0], 0.]]).dot(quat)
        eps = 1 - (quat[0]**2+quat[1]**2+quat[2]**2+quat[3]**2)
        k = 1
        dquat = dquat + k*eps*quat
        domega = self.Jinv.dot(M - np.cross(omega, J.dot(omega), axis=0))

        return dpos, dvel, dquat, domega

    def set_dot(self, t, rotors):
        states = self.observe_list()
        dots = self.deriv(*states, rotors)
        self.pos.dot, self.vel.dot, self.quat.dot, self.omega.dot = dots


if __name__ == "__main__":
    system = Copter_nonlinear()
    system.set_dot(t=0, rotors=np.zeros((4, 1)))
    print(repr(system))