import numpy as np

from .type import HamiltonianType
from .base import BaseHamiltonian

class SinglePendulum(BaseHamiltonian):
    """
    1 dof
    """
    def __init__(self, m=1., l=1., g=1., f=1.):
        """
        m: Mass of pendulum
        l: Length of pendulum
        g: Gravitational acceleration
        f: frequency of the trigonometric expression, if you increase this value
           then the function derivatives become bigger, and function oscillates more
        """
        self.m = m
        self.l = l
        self.g = g
        self.f = f

    def H(self, x):
        x = x.reshape(-1, 2)
        (q,p) = x[:,0], x[:,1]
        f = p**2 / (2 * self.m * self.l**2) + self.m * self.g * self.l * (1 - np.cos(self.f * q))
        return f.reshape(-1, 1)

    def H_grad(self, x):
        x = x.reshape(-1, 2)
        # output is array of values [y_1, y_2, ...] where y_i = (dH/dq, dH/dp)
        (q,p) = x[:,0], x[:,1]
        dq = self.m * self.g * self.l * self.f * np.sin(self.f * q)
        dp = p / (self.m * self.l**2)
        df = np.array([dq, dp]).T
        return df.reshape(-1, 2)

    def type(self):
        return HamiltonianType.SINGLE_PENDULUM
