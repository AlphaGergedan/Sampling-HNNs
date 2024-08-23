import numpy as np

from .type import HamiltonianType
from .base import BaseHamiltonian

# see https://diego.assencio.com/?index=e5ac36fcb129ce95a61f8e8ce0572dbf for a good source
# see cranmer-2020
# see jakovac-2022
# see chen-2021
# see jin-2021

class DoublePendulum(BaseHamiltonian):
    """
    2 dof
    """
    def __init__(self, m1=1., m2=1., l1=1., l2=1., g=1.):
        """
        m1: Mass of first pendulum
        m2: Mass of second pendulum
        l1: length of the first pendulum
        l2: length of the second pendulum
        g: Gravitational acceleration
        """
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    def H(self, x):
        x = x.reshape(-1,4)
        (q1,q2,p1,p2) = x[:,0], x[:,1], x[:,2], x[:,3]
        f = ( (self.m2 * self.l2**2 * p1**2 + (self.m1 + self.m2) * self.l1**2 * p2**2 - 2 * self.m2 * self.l1 * self.l2 * p1 * p2 * np.cos(q1 - q2) ) / (2 * self.m2 * self.l1**2 * self.l2**2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2)) ) - (self.m1 + self.m2) * self.g * self.l1 * np.cos(q1) - self.m2 * self.g * self.l2 * np.cos(q2)
        return f.reshape(-1, 1)

    def H_grad(self, x):
        x = x.reshape(-1, 4)
        (q1,q2,p1,p2) = x[:,0], x[:,1], x[:,2], x[:,3]

        # define helper equations used commonly in dH/dq1 and dH/dq2
        h1 = (p1 * p2 * np.sin(q1 - q2)) / (self.l1 * self.l2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2))
        h2 = (self.m2 * self.l2**2 * p1**2 + (self.m1 + self.m2) * self.l1**2 * p2**2 - 2 * self.m2 * self.l1 * self.l2 * p1 * p2 * np.cos(q1 - q2)) / (2 * self.l1**2 * self.l2**2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2)**2)

        # from paper wu-2020 it is defined without the square in the last term
        # h2 = (self.m2 * self.l2**2 * p1**2 + (self.m1 + self.m2) * self.l1**2 * p2**2 - 2 * self.m2 * self.l1 * self.l2 * p1 * p2 * np.cos(q1 - q2)) / (2 * self.l1**2 * self.l2**2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2))

        dq1 = (self.m1 + self.m2) * self.g * self.l1 * np.sin(q1) + h1 - h2 * np.sin(2 * (q1 - q2))
        dq2 = self.m2 * self.g * self.l2 * np.sin(q2) - h1 + h2 * np.sin(2 * (q1 - q2))
        dp1 = (self.l2 * p1 - self.l1 * p2 * np.cos(q1 - q2)) / (self.l1**2 * self.l2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2))
        dp2 = ((self.m1 + self.m2) * self.l1 * p2 - self.m2 * self.l2 * p1 * np.cos(q1 - q2)) / (self.m2 * self.l1 * self.l2**2 * (self.m1 + self.m2 * np.sin(q1 - q2)**2))

        df = np.array([ dq1, dq2, dp1, dp2 ]).T
        return df.reshape(-1, 4)

    def type(self):
        return HamiltonianType.DOUBLE_PENDULUM
