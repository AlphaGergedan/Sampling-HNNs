# Used for traditional training
from dataclasses import dataclass

from activation import ActivationType
from hamiltonian import BaseHamiltonian
from util.device_type import DeviceType

@dataclass
class TraditionalDomainParams:
    target: BaseHamiltonian
    q_lims: list[list[float]]
    p_lims: list[list[float]]
    train_size: int
    test_size: int
    batch_size: int
    data_random_seed: int

@dataclass
class TraditionalModelParams:
    activation: ActivationType
    network_width: int
    learning_rate: float
    weight_decay: float
    device: DeviceType
    total_steps: int
    model_random_seed: int

@dataclass
class TraditionalModelResults:
    # rel. L^2
    train_losses: list[float]

    # final error on the test set
    test_H_error: float
    train_time: float

@dataclass
class TraditionalExperiment:
    domain_params: TraditionalDomainParams
    model_params: TraditionalModelParams
    results: TraditionalModelResults

