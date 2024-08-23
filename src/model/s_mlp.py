import torch.nn as nn; from numpy import ndarray; import numpy as np; from sklearn.pipeline import Pipeline; from swimnetworks.swimnetworks import Dense, Linear; from typing import Any

from activation import BaseActivation, Activation
from trainer import ParameterSampler

from .base import BaseModel


class S_MLP(nn.Module, BaseModel):

    T = BaseModel.T

    pipeline: Pipeline
    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: BaseActivation

    # initial sampling parameters
    resample_duplicates: bool
    rcond: float
    random_seed: int

    # only for ELM relevant settings
    elm_bias_start: float
    elm_bias_end: float

    def __init__(self, input_dim, hidden_dim, output_dim, activation, resample_duplicates, rcond, random_seed, elm_bias_start, elm_bias_end, **kwargs):
        super(S_MLP, self).__init__()

        self.is_torch_model = False

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = Activation.new(activation)
        self.resample_duplicates = resample_duplicates
        self.rcond = rcond
        self.random_seed = random_seed
        self.elm_bias_start = elm_bias_start
        self.elm_bias_end = elm_bias_end

        steps = [
            (
                "dense",
                Dense(layer_width=hidden_dim, activation=self.activation.forward, elm_bias_start=elm_bias_start, elm_bias_end=elm_bias_end,
                      parameter_sampler=ParameterSampler.A_PRIORI.value, sample_uniformly=True, resample_duplicates=resample_duplicates, random_seed=random_seed),
            ),
            (
                "linear",
                Linear(layer_width=output_dim, regularization_scale=rcond),
            )
        ]
        self.pipeline = Pipeline(steps)

    def forward(self, x):
        return BaseModel.forward(self, x)

    def dt(self, x: T, create_graph=False) -> T:
        assert isinstance(x, ndarray)

        return self.__forward_dt(x)

    def evaluate_H(self, x, H_true) -> float | None:
        # Plain ODE-Net does not support Hamiltonian extraction
        return None

    def evaluate_H_grad(self, x, H_grad_true) -> float | None:
        return None

    def init_params(self):
        """
        Zero initialization of weights and biases
        """

        dense_layer: Any = self.pipeline[0]
        linear_layer: Any = self.pipeline[1]

        dense_layer.weights = np.zeros((self.hidden_dim, self.input_dim)).T
        dense_layer.biases = np.zeros((1, self.hidden_dim))

        # output_dim is set to 1 for Hamiltonian approximations (S-HNN)
        # and to input_dim for ODE-Net (S-MLP)
        linear_layer.weights = np.zeros((self.output_dim, self.hidden_dim)).T
        linear_layer.biases = np.zeros((1, self.output_dim))

    def H(self, x: T, create_graph=False) -> T:
        raise ValueError("plain ODE-Net does not support Hamiltonian output")

    def H_grad(self, x: T, create_graph=False) -> T:
        """
        Actually it is possible to extract Hamiltonian from plain ODE-Net
        using Hamilton's equations, so we just implement that here.

        @param x            : input numpy ndarray

        @return             : gradient of the Hamiltonian w.r.t. to the input
        """
        assert isinstance(x, ndarray)

        # H grad can be recovered from x_dot outputs
        x_dot = self.__forward_dt(x)
        q_dot, p_dot = np.split(x_dot, 2, axis=1)

        # Hamilton's Equations
        dHdq = -p_dot
        dHdp =  q_dot

        dHdx = np.hstack((dHdq, dHdp))

        return dHdx

    def H_last_layer(self):
        raise ValueError("plain ODE-Net does not support Hamiltonian")

    def compute_grad_last_hidden_wrt_input(self, x):
        """
        Gives gradients of the last hidden layer output w.r.t x, of shape (KD,N_last)

        @param x            : input data points in the phase space (K,D)

        @return dx         : derivatives of hidden layer output w.r.t x of shape (KD,N_last)
        """
        # get dense and linear layer
        assert len(x.shape) == 2
        dense_layer: Any = self.pipeline[0]

        # calculate first dense layer derivative w.r.t. x => of shape (KD,M) where M is the last hidden layer size
        dense_layer.activation = self.activation.grad

        d_activation_wrt_x = dense_layer.transform(x) # (K,N_1)
        dense_layer.activation = self.activation.forward

        return np.einsum('ij,kj->ikj', d_activation_wrt_x, dense_layer.weights) # (K,D,N_1)

    def __forward_dt(self, x: ndarray) -> ndarray:
        # in plain MLP, output of the model is the time derivatives
        return self.pipeline.transform(x)
