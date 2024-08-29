"""
For general purpose main, for running local toy examples
"""

import argparse
from typing import Any

from hamiltonian import HamiltonianType
from model.type import ModelType
from model.sampled_network_type import SampledNetworkType
from activation import ActivationType
from trainer.param_sampler import ParameterSampler
from util import DeviceType


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="main", description="Train shallow networks to approximate Hamiltonians.")

    # model to train
    parser.add_argument("--model", type=ModelType, required=True, choices=list(ModelType), help="model to train")

    # target task
    parser.add_argument("--target", type=HamiltonianType, required=True, choices=list(HamiltonianType), help="target Hamiltonian system to approximate")
    parser.add_argument("--rho", type=float, required=False, default=0., help="Dissipation coefficient of the system, 0 for no dissipation, default=0")

    # common training/fitting args
    parser.add_argument("--train-size", type=int, required=False, default=10000, help="number of train points, default=10000")
    parser.add_argument("--test-size", type=int, required=False, default=2000, help="number of test points, default=2000")
    parser.add_argument("--network-width", type=int, required=False, default=512, help="number of basis functions, default=512")
    parser.add_argument("--activation", type=ActivationType, required=False, default=ActivationType.TANH, choices=list(ActivationType), help="nonlinearity used in the model, default=tanh")
    parser.add_argument("--data-random-seed", type=int, required=False, default=3943, help="for sampling train and test points, default=3943")
    parser.add_argument("--model-random-seed", type=int, required=False, default=992472, help="for model training, default=992472")

    # traditional training args
    parser.add_argument("--device", type=DeviceType, required=False, default=DeviceType.GPU, choices=list(DeviceType), help="which device to utilize, only relevant for traditionally trained models {HNN,D-HNN}, default=cuda")
    parser.add_argument("--batch-size", type=int, required=False, default=256, help="batch size for training, only relevant for traditionally trained models {HNN,D-HNN}, default=256")
    parser.add_argument("--total-steps", type=int, required=False, default=5000, help="number of total training steps, only relevant for traditionally trained models {HNN,D-HNN}, default=5000")
    parser.add_argument("--learning-rate", type=float, required=False, default=1e-3, help="learning rate, only relevant for traditionally trained models {HNN,D-HNN}, default=1e-03")
    parser.add_argument("--weight-decay", type=float, required=False, default=0., help="weight decay regularization, only relevant for traditionally trained models {HNN,D-HNN}, default=0.")

    # sampling args
    parser.add_argument("--sampling-type", type=SampledNetworkType, required=False, default=SampledNetworkType.A_SWIM, choices=list(SampledNetworkType), help="sampled network type, default=A-SWIM")
    parser.add_argument("--param-sampler", type=ParameterSampler, required=False, default=ParameterSampler.TANH, choices=list(ParameterSampler), help="parameter sampler used in the SWIM algorithm, for sampled models {S-MLP,S-HNN}, default=tanh")
    parser.add_argument("--rcond", type=float, required=False, default=1e-13, help="how precise the least-squares-solution is, for sampled models {S-MLP, S-HNN}, default=1e-13")
    parser.add_argument("--resample-duplicates", action="store_true", required=False, default=False, help="whether to resample from data if duplicate weights are detected until we get unique weights, for sampled models {S-MLP,S-HNN}, default=True")

    # other args
    parser.add_argument("--dry-run", action="store_true", required=False, default=False, help="prints the task and model summary and quits")

    kwargs = vars(parser.parse_args())

    # the following variable decides whether to use torch.Tensor as type
    kwargs["is_torch"] = False
    if kwargs["model"] is ModelType.MLP or kwargs["model"] is ModelType.HNN:
        kwargs["is_torch"] = True

    return kwargs
