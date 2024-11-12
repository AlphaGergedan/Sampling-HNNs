"""
For experimenting with HNN (gradient based training) specifically.
"""

import argparse
from typing import Any

from activation import ActivationType
from util.device_type import DeviceType


def hnn_on_cpu_experiment_argparser() -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="sampled hnn experiment", description="for experimenting with sampled networks")

    parser.add_argument("--data-random-seed", type=int, required=False, default=3943, help="seed for data generation (numpy)")
    parser.add_argument("--model-random-seed", type=int, required=False, default=992472, help="seed for model training (torch)")
    parser.add_argument("--batch-size", type=int, required=False, default=2048, help="batch size in the gradient-descent based training")

    parser.add_argument("--activation", type=ActivationType, required=False, default=ActivationType.TANH, choices=list(ActivationType), help="nonlinearity used in the model, default=tanh'")
    parser.add_argument("--learning-rate", type=float, required=False, default=1e-4, help="learning rate in the gradient-descent based training")
    parser.add_argument("--weight-decay", type=float, required=False, default=1e-13, help="L^2 regularization in the gradient-descent based training")
    parser.add_argument("--network-width", type=int, required=False, default=5000, help="Num. of neurons used in the hidden layer")
    parser.add_argument("--total-steps", type=int, required=False, default=180000, help="Num. of gradient steps in the gradient-descent based training")
    parser.add_argument("--device", type=DeviceType, required=False, default=DeviceType.CPU, help="use 'cuda' (GPU) or 'cpu' (CPU) for training")

    # other args
    parser.add_argument("--save-dir", type=str, help="dir to save the resulting models and error", required=True)

    kwargs = vars(parser.parse_args())
    return kwargs
