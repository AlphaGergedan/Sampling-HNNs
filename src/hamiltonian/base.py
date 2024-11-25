import numpy as np

from abc import ABC, abstractmethod
from typing import Any

from .type import HamiltonianType

class BaseHamiltonian(ABC):

    def dt(self, x) -> Any:
        """
        Returns system dynamics (time derivatives of the input (q,p)) using Hamilton's equations

        @param x    : input


        @return: (q_dot, p_dot)
        """

        # gradients of the Hamiltonian of shape (_, 2), one for w.r.t. q, one for p
        H_grad = self.H_grad(x)
        H_grad_q, H_grad_p = np.split(H_grad, 2, axis=1)

        # Hamilton's Equations
        q_dot =  H_grad_p
        p_dot = -H_grad_q

        return np.hstack((q_dot, p_dot))

    @abstractmethod
    def H(self, x) -> Any:
        """
        Hamiltonian value given input.

        @param x: input array
        @return: scalar Hamiltonian value
        """
        pass

    @abstractmethod
    def H_grad(self, x) -> Any:
        """
        Gradient of the Hamiltonian value given input.

        @param x: input array
        @return: gradient of the Hamiltonian value
        """
        pass

    @abstractmethod
    def type(self) -> HamiltonianType:
        """
        Hamitonian type.

        @return: enum value of the Hamiltonian type
        """
        pass
