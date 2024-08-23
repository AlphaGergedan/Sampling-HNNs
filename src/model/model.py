from .type import ModelType

# ODE-Nets
from .mlp import MLP
from .s_mlp import S_MLP

# Hamiltonian Neural Networks
from .hnn import HNN
from .s_hnn import S_HNN


class Model():
    @staticmethod
    def new(model: ModelType, input_dim, network_width, activation, random_seed, **kwargs):
        match model:
            case ModelType.MLP:
                # output_dim is set to input_dim, because plain ODE-Net outputs x_dot,
                # which has the same shape as x
                return MLP(input_dim, network_width, output_dim=input_dim, activation=activation, random_seed=random_seed)

            case ModelType.S_MLP:
                # same as MLP, but sampled instead of traditional training
                # output_dim is set to input_dim, because plain ODE-Net outputs x_dot,
                # which has the same shape as x
                return S_MLP(input_dim, network_width, output_dim=input_dim, activation=activation, random_seed=random_seed, elm_bias_start=kwargs["min_input"], elm_bias_end=kwargs["max_input"], **kwargs)

            case ModelType.HNN:
                return HNN(input_dim, network_width, activation, random_seed=random_seed)

            case ModelType.S_HNN:
                return S_HNN(input_dim, network_width, activation, random_seed=random_seed, elm_bias_start=kwargs["min_input"], elm_bias_end=kwargs["max_input"], **kwargs)

            case _:
                raise NotImplementedError(f"model type {type} is not implemented yet")
