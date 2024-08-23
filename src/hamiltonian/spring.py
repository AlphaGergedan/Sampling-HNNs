import numpy as np

from .type import HamiltonianType
from .base import BaseHamiltonian

class Spring(BaseHamiltonian):
    """
    1 dof
    """
    def __init__(self, m=1., k=1.):
        """
        m: Mass of pendulum
        k: Spring constant
        """
        self.m = m
        self.k = k

    def H(self, x):
        x = x.reshape(-1, 2)
        (q,p) = x[:,0], x[:,1]
        f = (self.k * q**2) / 2 + p**2 / (2 * self.m)
        return f.reshape(-1, 1)

    def H_grad(self, x):
        x = x.reshape(-1, 2)
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dq, dH/dp)
        (q,p) = x[:,0], x[:,1]
        dq = self.k * q
        dp = p / self.m
        df = np.array([dq, dp]).T
        return df.reshape(-1, 2)

    def type(self):
        return HamiltonianType.SPRING
