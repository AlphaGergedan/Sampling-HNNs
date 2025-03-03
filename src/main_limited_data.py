# system libs
from time import time; from typing import Any

# data
import numpy as np
from data import get_train_test_set

# logger, model saving
from logger import print_model_summary, print_errors

# argument parser
from argparser import parse_args

from hamiltonian import Hamiltonian
from model import Model
from model.sampled_network_type import SampledNetworkType
from trainer import Trainer, TrainerType
from util import get_model_error
from error import l2_error_rel

import torch
torch.set_default_dtype(torch.float64)


# read/parse command line arguments
kwargs: dict[str, Any] = parse_args()

# prepare target function
input_dim, (q_lims, p_lims), target = Hamiltonian.new(**kwargs)
kwargs["q_lims"] = q_lims
kwargs["p_lims"] = p_lims
kwargs["min_input"] = np.min([q_lims, p_lims])
kwargs["max_input"] = np.max([q_lims, p_lims])

MODEL_RANDOM_SEED = kwargs["model_random_seed"]
DATA_RANDOM_SEED = kwargs["data_random_seed"]

# prepare model
model = Model.new(input_dim=input_dim, random_seed=MODEL_RANDOM_SEED, **kwargs)

if kwargs["dry_run"]:
    print_model_summary(model, target, kwargs)
    exit(0)

# init model parameters
model.init_params()

# prepare inputs and truths (of type ndarray for sampling, and Tensor for traditional training on gpu)

"""
# this should sample target hamiltonian specific data, e.g. for spring specific, pendulum specific
# you can also incorporate dissipative components here
# train_inputs, test_inputs, train_input_x_0, train_x_0_truth  = Data.new(**kwargs)
"""

# Datasets for training and evaluating the model

train_set, test_set = get_train_test_set(input_dim // 2, target, kwargs["train_size"], kwargs["test_size"], q_lims, p_lims, rng=np.random.default_rng(DATA_RANDOM_SEED), use_fd=True)

( ((train_inputs, train_inputs_next), train_dt_truths, train_H_truths, train_H_grad_truths), (train_x_0, train_x_0_H_truth) ) = train_set
( test_inputs, test_dt_truths, test_H_truths, test_H_grad_truths ) = test_set

train_errors = get_model_error(model, train_inputs, (train_dt_truths, train_H_truths, train_H_grad_truths))
test_errors = get_model_error(model, test_inputs, (test_dt_truths, test_H_truths, test_H_grad_truths))
print_errors(train_errors, test_errors)

# TRAINING

if model.is_torch_model:
    trainer = Trainer.new(TrainerType.TRADITIONAL, **kwargs)
else:
    trainer = Trainer.new(TrainerType.SAMPLER, **kwargs)

train_truths = None
if kwargs["sampling_type"] is SampledNetworkType.SWIM:
    train_truths = train_H_truths

assert isinstance(train_inputs_next, np.ndarray)

time_begin = time()
# use symplectic euler scheme
train_inputs_next_q, _ = np.split(train_inputs_next, 2, axis=1)
_, train_inputs_p = np.split(train_inputs, 2, axis=1)
trainer.train(model, np.hstack((train_inputs_next_q, train_inputs_p)), train_dt_truths, train_x_0, train_x_0_H_truth, kwargs["device"], train_H_truths=train_truths)
time_end = time()

time_str = "{:.2f}".format(time_end - time_begin)
print(f"\n-> training took {time_str} seconds")

# fixme: make it an argument
dt_obs = 1e-1

train_H_preds = model.H(train_inputs)
error = l2_error_rel(train_H_truths, train_H_preds)
print(f"-> train error:             {error : .2E}")

train_H_grad_preds = model.H_grad(train_inputs)
train_H_grad_preds_q_part, train_H_grad_preds_p_part = np.split(train_H_grad_preds, 2, axis=1)
train_H_preds = train_H_preds + (dt_obs/2) * np.einsum("ij,ij->i", train_H_grad_preds_q_part, train_H_grad_preds_p_part).reshape(-1, 1)
error = l2_error_rel(train_H_truths, train_H_preds)
print(f"-> train corrected error:   {error: .2E}")

#--------

test_H_preds = model.H(test_inputs)
error = l2_error_rel(test_H_truths, test_H_preds)
print(f"-> test error:              {error : .2E}")

test_H_grad_preds = model.H_grad(test_inputs)
test_H_grad_preds_q_part, test_H_grad_preds_p_part = np.split(test_H_grad_preds, 2, axis=1)
test_H_preds = test_H_preds + (dt_obs/2) * np.einsum("ij,ij->i", test_H_grad_preds_q_part, test_H_grad_preds_p_part).reshape(-1, 1)
error = l2_error_rel(test_H_truths, test_H_preds)
print(f"-> test corrected error:    {error: .2E}")
