import torch.nn as nn; from numpy import ndarray; import numpy as np; from typing import Any

from error import l2_error_rel

from .base import BaseModel
from .s_mlp import S_MLP


class S_HNN(nn.Module, BaseModel):

    T = BaseModel.T

    mlp: S_MLP

    def __init__(self, input_dim, hidden_dim, activation, resample_duplicates, rcond, random_seed, elm_bias_start, elm_bias_end, **kwargs):
        super(S_HNN, self).__init__()

        self.is_torch_model = False

        # MLP for Hamiltonian (conserved component)
        self.mlp = S_MLP(input_dim, hidden_dim, 1, activation, resample_duplicates, rcond, random_seed, elm_bias_start, elm_bias_end)

    def forward(self, x):
        return BaseModel.forward(self, x)

    def dt(self, x: T, create_graph=False) -> T:
        assert isinstance(x, ndarray)

        # decompose the gradient into q and p parts
        grad = self.H_grad(x, create_graph=create_graph)
        assert isinstance(grad, ndarray)

        grad_q, grad_p = np.split(grad, 2, axis=1)

        # Hamilton's Equations
        q_dot = grad_p
        p_dot = -grad_q

        return np.hstack((q_dot, p_dot))

    def evaluate_H(self, x, H_true) -> float | None:
        assert isinstance(x, ndarray)
        assert isinstance(H_true, ndarray)

        H_pred = self.H(x)
        assert isinstance(H_pred, ndarray)

        error = l2_error_rel(H_true, H_pred)
        assert isinstance(error, float)
        return error

    def evaluate_H_grad(self, x, H_grad_true) -> float | None:
        assert isinstance(x, ndarray)
        assert isinstance(H_grad_true, ndarray)

        H_grad_pred = self.H_grad(x)
        assert isinstance(H_grad_pred, ndarray)

        error = l2_error_rel(H_grad_true, H_grad_pred)
        assert isinstance(error, float)
        return error

    def init_params(self):
        self.mlp.init_params()

    def H(self, x: T, create_graph=False) -> T:
        # in HNN, the MLP used here outputs Hamiltonian scalar directly
        assert isinstance(x, ndarray)
        return self.mlp.forward(x)

    def H_grad(self, x: T, create_graph=False) -> T:
        assert isinstance(x, ndarray)

        # get dense layers and linear layer
        linear_layer: Any = self.H_last_layer()

        # derivative of the last hidden layer w.r.t. input
        grad_last_hidden = self.mlp.compute_grad_last_hidden_wrt_input(x)

        # network derivative w.r.t. input
        grad = grad_last_hidden @ linear_layer.weights

        return grad.reshape(x.shape) # (K, D)

    def H_last_layer(self) -> Any:
        return self.mlp.pipeline[-1]
