import torch; import torch.nn as nn; from torch import Tensor

from activation import ActivationType
from error import l2_error_rel
from util import to_tensor

from .base import BaseModel
from .mlp import MLP


class HNN(nn.Module, BaseModel):

    T = BaseModel.T

    mlp: MLP

    def __init__(self, input_dim, hidden_dim, activation: ActivationType, random_seed):
        super(HNN, self).__init__()

        self.is_torch_model = True

        # MLP for Hamiltonian (conservative component)
        self.mlp = MLP(input_dim, hidden_dim, 1, activation, random_seed)

    def forward(self, x):
        return BaseModel.forward(self, x)

    def dt(self, x: T, create_graph=False) -> T:
        assert isinstance(x, Tensor)

        # decompose the gradient into q and p parts
        grad = self.H_grad(x, create_graph=create_graph)
        assert isinstance(grad, Tensor)

        grad_q, grad_p = torch.split(grad, grad.shape[1] // 2, dim=1)

        # Hamilton's Equations
        q_dot = grad_p
        p_dot = -grad_q

        return torch.cat((q_dot, p_dot), dim=1)

    def evaluate_H(self, x, H_true) -> float | None:
        x = to_tensor(x)
        H_true = to_tensor(H_true)

        H_pred = self.H(x)
        assert isinstance(H_pred, Tensor)

        error = l2_error_rel(H_true, H_pred)
        assert isinstance(error, float)
        return error

    def evaluate_H_grad(self, x, H_grad_true) -> float | None:
        x = to_tensor(x, requires_grad=True)
        H_grad_true = to_tensor(H_grad_true)

        H_grad_pred = self.H_grad(x)
        assert isinstance(H_grad_pred, Tensor)

        error = l2_error_rel(H_grad_true, H_grad_pred)
        assert isinstance(error, float)
        return error

    def init_params(self):
        self.mlp.init_params()

    def H(self, x: T, create_graph=False) -> T:
        assert isinstance(x, Tensor)
        return self.mlp.forward(x)

    def H_grad(self, x: T, create_graph=False) -> T:
        assert isinstance(x, Tensor)

        y = self.H(x)
        y_deriv = torch.autograd.grad(y.sum(), x, create_graph=create_graph, retain_graph=create_graph)[0]

        return y_deriv

    def H_last_layer(self):
        return self.mlp.linear_last
