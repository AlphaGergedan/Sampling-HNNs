import numpy as np

from .type import HamiltonianType
from .base import BaseHamiltonian

# see han-2021 adaptable HNN paper and also Y. Tong, S. Xiong, X. He, G. Pan, and B. Zhu, “Symplectic neural networks in Taylor series form for Hamiltonian systems,” J. Comput. Phys., vol. 437, p. 110325, 2021.
# a good description of the system and where it is used is in han-2021

class HenonHeiles(BaseHamiltonian):
    """
    2 dof
    """
    def __init__(self, alpha=1.):
        """
        alpha: bifurcation parameter 0 <= alpha <= 1, sets the magnitude of the nonlinear potential function
        TODO: test on alpha = 0, 0.7, 0.9, 1 as they did in their paper han-2021, and they plot p2 and q2 with q1=0
        """
        self.alpha = alpha

    def H(self, x):
        """
        Hamiltonian with 2 dof, non-integrable, chaotic, training should be done in [-0.5,0.5]x[-0.5,0.5] according to tong-2021, they used alpha==1
        """
        assert x.shape == (x.shape[0],4)
        (q1,q2,p1,p2) = x[:,0], x[:,1], x[:,2], x[:,3]
        f = ( ( (p1**2) + (p2**2) ) / 2 ) + ( ( (q1**2) + (q2**2) ) / 2 ) + ( self.alpha * ( ((q1**2) * q2) - ((q2**3) / 3)) )
        return f.reshape(-1, 1)

    def H_grad(self, x):
        assert x.shape == (x.shape[0],4)
        (q1,q2,p1,p2) = x[:,0], x[:,1], x[:,2], x[:,3]

        # define helper equations used commonly in dH/dq1 and dH/dq2
        dq1 = q1 + self.alpha * (2*q1*q2)
        dq2 = q2 + self.alpha * (q1**2 - q2**2)
        dp1 = p1
        dp2 = p2

        # (x.shape[0], 4)
        df = np.array([ dq1, dq2, dp1, dp2 ]).T
        return df.reshape(-1, 4)

    def type(self):
        return HamiltonianType.HENON_HEILES
