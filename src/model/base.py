from typing import Any; from abc import ABC, abstractmethod

from torch import Tensor; from numpy import ndarray

from util import to_tensor
from error import l2_error_rel


class BaseModel(ABC):

    is_torch_model: bool
    T = Tensor | ndarray

    def forward(self, x: T) -> T:
        """
        Forward pass for the models, outputs system dynamics through x_dot,
        for all the models, since this is the common output for all the models.

        @param x    : input, torch tensor or numpy ndarray depending on the model

        @return     : x_dot
        """
        return self.dt(x, create_graph=True)

    def evaluate_dt(self, x, x_dot_true) -> float:
        """
        Given inputs and true values of system dynamics (x_dot)
        evaluates the model's prediction error. All models should
        have this same.

        @param x            : inputs as torch tensors or numpy arrays
        @param x_dot_true   : true time derivatives

        @return             : error as numpy array
        """
        # for torch models convert to torch tensors
        if self.is_torch_model:
            x = to_tensor(x, requires_grad=True)
            x_dot_true = to_tensor(x_dot_true)

            x_dot_pred = self.dt(x, create_graph=False)
            assert isinstance(x_dot_pred, Tensor)

            error = l2_error_rel(x_dot_true, x_dot_pred)
            assert isinstance(error, float)
        else:
            x_dot_pred = self.dt(x, create_graph=False)
            assert isinstance(x_dot_pred, ndarray)

            error = l2_error_rel(x_dot_true, x_dot_pred)
            assert isinstance(error, float)

        return error

    @abstractmethod
    def dt(self, x: T, create_graph=False) -> T:
        """
        Outputs the time derivaitve of the given input

        @param x                : input, torch tensor or numpy ndarray depending on the model
        @param create_graph     : for higher order derivatives

        @return                 : x_dot
        """
        pass

    @abstractmethod
    def evaluate_H(self, x, H_true) -> float | None:
        pass

    @abstractmethod
    def evaluate_H_grad(self, x, H_grad_true) -> float | None:
        pass

    @abstractmethod
    def init_params(self):
        """
        Initialize the model parameters.
        """
        pass

    @abstractmethod
    def H(self, x: T, create_graph=False) -> T:
        """
        Hamiltonian part of the network.

        @param x                : input, torch tensor or numpy ndarray depending on the model
        @param create_graph     : for higher order derivatives

        @return                 : predicted Hamiltonian function value given input x
        """
        pass

    @abstractmethod
    def H_grad(self, x: T, create_graph=False) -> T:
        """
        Gradient of the Hamiltonian w.r.t. to the input.

        @param x                : input, torch tensor or numpy ndarray depending on the model
        @param create_graph     : for higher order derivatives

        @return                 : Gradient of the Hamiltonian w.r.t. to the input
        """
        pass

    @abstractmethod
    def H_last_layer(self) -> Any:
        """
        Returns the last layer of the part that learns the Hamiltonian. Useful
        for setting the bias directly using an assumed known value (H(x)=x for some x)
        """
        pass
