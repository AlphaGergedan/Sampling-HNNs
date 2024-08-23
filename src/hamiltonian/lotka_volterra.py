import numpy as np

from .type import HamiltonianType
from .base import BaseHamiltonian

class LotkaVolterra(BaseHamiltonian):
    """
    Hamiltonian Formulations for the well-known Lotka-Volterra model which is a
    mathematical model used to describe the dynamics of predation-prey populations.

    REF: Y. Tong, S. Xiong, X. He, G. Pan, and B. Zhu, “Symplectic neural networks in Taylor series form for Hamiltonian systems,” J. Comput. Phys., vol. 437, p. 110325, 2021.

    """
    def __init__(self, alpha=-2., beta=-1., gamma=-1., delta=-1.):
        """
        Choose these parameters wisely, they change the phase plot.
        For the classical rabbit and fox model you can set
        alpha=1.1, beta=0.4, gamma=0.1, delta=0.4

        @param alpha: prey's: the maximum prey per capita growth rate
        @param beta : prey's: the effect of the presence of predators on the prey death rate
        @param gamma: predator's: predator's per capita death rate
        @param delta: predator's: the effect of the presence of prey on the predator growth rate
        """
        self.delta = delta
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    def H(self, x):
        x = x.reshape(-1, 2)
        (q,p) = x[:,0], x[:,1]
        # f = delta*e^p - gamma*p + beta*e^q - alpha*q
        f = self.delta * np.e**p - self.gamma * p + self.beta * np.e**q - self.alpha * q
        return f.reshape(-1, 1)

    def H_grad(self, x):
        x = x.reshape(-1, 2)
        (q,p) = x[:,0], x[:,1]
        dq = self.beta * np.e**q - self.alpha
        # dq = -alpha + beta*e^q
        dp = self.delta * np.e**p - self.gamma
        # dp = -gamma + delta*e^p
        df = np.array([dq, dp]).T
        return df.reshape(-1, 2)

    def type(self):
        return HamiltonianType.LOTKA_VOLTERRA
