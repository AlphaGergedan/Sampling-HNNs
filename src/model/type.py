from enum import StrEnum

class ModelType(StrEnum):
    # Baseline: ODE-Net
    MLP = "MLP"

    # Baseline ODE-Net (supervised sampled model)
    S_MLP = "S-MLP"

    # Hamiltonian Neural Network: https://github.com/greydanus/hamiltonian-nn/tree/master
    HNN = "HNN"

    # Sampled Hamiltonian Neural Network (ours)
    S_HNN = "S-HNN"
