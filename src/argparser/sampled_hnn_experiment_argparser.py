"""
For experimenting with S-HNN specifically.
"""

import argparse
from typing import Any

from activation import ActivationType


def sampled_hnn_experiment_argparser() -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog="sampled hnn experiment", description="for experimenting with sampled networks")

    # common training/fitting args
    parser.add_argument("--activation", type=ActivationType, required=False, default=ActivationType.TANH, choices=list(ActivationType), help="nonlinearity used in the model, default=tanh")
    parser.add_argument("--start-data-random-seed", type=int, required=False, default=3943, help="start point (is incremented after each experiment run) for the random seed used for sampling train and test points, default=3943")
    parser.add_argument("--start-model-random-seed", type=int, required=False, default=992472, help="start point (is incremented after each experiment run) for the model training, default=992472")

    # sampling args
    parser.add_argument("--rcond", type=float, required=False, default=1e-13, help="how precise the least-squares-solution is, for sampled models {S-MLP, S-HNN}, default=1e-13")
    parser.add_argument("--resample-duplicates", action="store_true", required=False, default=False, help="whether to resample from data if duplicate weights are detected until we get unique weights, for sampled models {S-MLP, S-HNN}, default=True")

    # other args
    parser.add_argument("--repeat", type=int, help="number of runs to experiment", required=True)
    parser.add_argument("--save-dir", type=str, help="dir to save the resulting models and error", required=True)

    kwargs = vars(parser.parse_args())
    return kwargs
