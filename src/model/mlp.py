import torch; import torch.nn as nn; from torch import Tensor;

from activation import ActivationType

from .base import BaseModel


class MLP(nn.Module, BaseModel):

    T = BaseModel.T

    def __init__(self, input_dim, hidden_dim, output_dim, activation: ActivationType, random_seed):
        super(MLP, self).__init__()

        torch.manual_seed(random_seed)

        self.is_torch_model = True

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.flatten = nn.Flatten()
        self.linear_first = nn.Linear(input_dim, hidden_dim)

        match activation:
            case ActivationType.TANH:
                self.activation = torch.tanh

        # plain ODE-Net MLP outputs the vector field on x (x_dot), which has the same shape as x
        # HNN, using the MLP, should set output_dim=1 as it outputs the Hamiltonian and
        # Hamiltonian is scalar.
        self.linear_last = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return BaseModel.forward(self, x)

    def dt(self, x: T, create_graph=False) -> T:
        assert isinstance(x, Tensor)

        return self.__forward_dt(x)

    def evaluate_H(self, x, H_true) -> float | None:
        # Plain ODE-Net does not support Hamiltonian extraction
        return None

    def evaluate_H_grad(self, x, H_grad_true) -> float | None:
        return None

    def init_params(self):
        nn.init.xavier_normal_(self.linear_first.weight)
        nn.init.zeros_(self.linear_first.bias)

        nn.init.xavier_normal_(self.linear_last.weight)
        nn.init.zeros_(self.linear_last.bias)

    def H(self, x: T, create_graph=False) -> T:
        raise ValueError("plain ODE-Net does not support Hamiltonian output")

    def H_grad(self, x: T, create_graph=False) -> T:
        """
        Actually it is possible to extract Hamiltonian from plain ODE-Net
        using Hamilton's equations, so we just implement that here.

        @param x            : input torch tensor

        @return             : gradient of the Hamiltonian w.r.t. to the input
        """
        assert isinstance(x, Tensor)

        # H grad can be recovered from x_dot outputs
        x_dot = self.__forward_dt(x)
        q_dot, p_dot = torch.split(x_dot, x_dot.shape[1] // 2, dim=1)

        # Hamilton's Equations
        dHdq = -p_dot
        dHdp =  q_dot

        dHdx = torch.cat((dHdq, dHdp), dim=1)

        return dHdx

    def H_last_layer(self):
        raise ValueError("plain ODE-Net does not support Hamiltonian")

    def __forward_dt(self, x: Tensor) -> Tensor:
        assert x.dim() == 2 and x.shape[1] == self.input_dim # should be (batch_size, input_dim)

        # in plain MLP, output of the model is the time derivatives

        x = self.flatten(x)
        x = self.linear_first(x)

        x = self.activation(x)
        x = self.linear_last(x)

        return x
