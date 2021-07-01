import numpy as np

from fym.core import BaseEnv, BaseSystem


class MorphingLon(BaseEnv):
    g = 9.80665  # [m/s^2]

    # mass and geometric property
    m = 10  # [kg]
    cbar = 0.288  [m]
    b = 3  # [m]
    bmin = 3  # [m]
    bmax = 4.446  # [m]

    # control surface
    control_limits = {
        "dela": (-0.5, 0.5),
        "dele": np.deg2rad(-10, 10),
        "delr": (-0.5, 0.5),
        "delt": (0, 1)
    }

    # thruster
    Tmax = 50  # [N]
    zeta = 1
    omega_n = 20  # [s^-1]


