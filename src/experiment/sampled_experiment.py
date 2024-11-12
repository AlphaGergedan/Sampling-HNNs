from dataclasses import dataclass

from activation import ActivationType
from hamiltonian import BaseHamiltonian
from model.s_hnn import S_HNN


@dataclass
class SampledDomainParams:
    target: BaseHamiltonian
    q_lims: list[list[float]]
    p_lims: list[list[float]]
    train_size: int
    test_size: int
    repeat: int
    start_data_random_seed: int

@dataclass
class SampledModelParams:
    activation: ActivationType
    network_width: int
    resample_duplicates: bool
    rcond: float
    elm_bias_start: float
    elm_bias_end: float
    start_model_random_seed: int

@dataclass
class SampledModels:
    elm_models: list[S_HNN]
    uswim_models: list[S_HNN]
    aswim_models: list[S_HNN]
    swim_models: list[S_HNN]

@dataclass
class SampledModelResults:
    # rel. L^2
    elm_test_H_errors: list[float]
    uswim_test_H_errors: list[float]
    aswim_test_H_errors: list[float]
    swim_test_H_errors: list[float]

    elm_train_times: list[float]
    uswim_train_times: list[float]
    aswim_train_times: list[float]
    swim_train_times: list[float]

@dataclass
class SampledExperiment:
    domain_params: SampledDomainParams
    model_params: SampledModelParams
    results: SampledModelResults
