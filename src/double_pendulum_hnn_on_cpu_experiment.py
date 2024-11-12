"""
This file conducts double pendulum experiment and train an HNN on cpu
to make a fair comparison (uses the same hardware as the sampled ones).
"""

# system libs
import os; from time import time; from typing import Any

# data
import numpy as np
from data import get_train_test_set

# logger, model saving
from joblib import dump
from experiment import TraditionalDomainParams, TraditionalModelParams, TraditionalExperiment, TraditionalModelResults

# gradient based network training using pytorch
import torch
num_cores = os.cpu_count()
print(f'-> found {num_cores} CPUs')
assert isinstance(num_cores, int)
torch.set_num_threads(num_cores)

from activation.type import ActivationType
from model.hnn import HNN
from trainer.traditional_trainer import TraditionalTrainer
from util.device_type import DeviceType

# argument parser
from argparser import hnn_on_cpu_experiment_argparser

# system
from hamiltonian.double_pendulum import DoublePendulum
double_pendulum = DoublePendulum()

kwargs: dict[str, Any] = hnn_on_cpu_experiment_argparser()

#########################################################################

# we only train for the following specific domain to make a comparison against sampling
train_size = 20000
test_size = 20000
data_random_seed = 3943 # same seed is used in the sampling experiments
model_random_seed = 992472 # same seed used in the sampling experiments
q_lims = [ [-np.pi, np.pi], [-np.pi, np.pi] ]
p_lims = [ [-1., 1.], [-1., 1.] ]
batch_size = 2048

assert kwargs["data_random_seed"] == data_random_seed
assert kwargs["model_random_seed"] == model_random_seed
assert kwargs["batch_size"] == batch_size

domain_params = TraditionalDomainParams(target=double_pendulum, q_lims=q_lims, p_lims=p_lims, train_size=train_size, test_size=test_size, batch_size=batch_size, data_random_seed=data_random_seed)

# train parameters
network_width = 5000
total_steps = 180000
learning_rate = 1e-04
weight_decay = 1e-13
device = DeviceType.CPU
activation = ActivationType.TANH

# assert these parameters for the experiment.
assert kwargs["learning_rate"] == learning_rate
assert kwargs["weight_decay"] == weight_decay
assert kwargs["network_width"] == network_width
assert kwargs["activation"] == activation
assert kwargs["total_steps"] == total_steps
assert kwargs["device"] == device

model_params = TraditionalModelParams(activation=ActivationType.TANH, network_width=network_width, learning_rate=learning_rate, weight_decay=weight_decay, device=device, total_steps=total_steps, model_random_seed=model_random_seed)

# gather data
train_set, test_set = get_train_test_set(2, domain_params.target, domain_params.train_size, domain_params.test_size, domain_params.q_lims, domain_params.p_lims, rng = np.random.default_rng(domain_params.data_random_seed))
( ((train_inputs, _), train_dt_truths, train_H_truths, train_H_grad_truths), (train_x_0, train_x_0_H_truth) ) = train_set
( test_inputs, test_dt_truths, test_H_truths, test_H_grad_truths ) = test_set

# setup model, model_random_seed seeds the MLP using torch.manual
hnn = HNN(input_dim=4, hidden_dim=model_params.network_width, activation=model_params.activation, random_seed=model_params.model_random_seed)

# setup trainer (uses cpu to make a fair comparison)
traditional_trainer = TraditionalTrainer(model_params.total_steps, domain_params.batch_size, model_params.learning_rate, model_params.weight_decay, model_params.device)

# train the model using gradient_descent
time_begin = time()
train_losses = traditional_trainer.train(hnn, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, device)
time_end = time()

time_hnn = time_end - time_begin

# l2 relative error
test_H_error = hnn.evaluate_H(test_inputs, test_H_truths); assert isinstance(test_H_error, float)

print(f"HNN test (H) error (rel.l2) : {test_H_error: .2E}")
print(f"HNN train time              : {time_hnn : .1f}")

results = TraditionalModelResults(train_losses, test_H_error, time_hnn)
experiment = TraditionalExperiment(domain_params, model_params, results)

# save model and experiment
dump(hnn, os.path.join(kwargs["save_dir"], f"hnn_trained_on_cpu.pkl"))
dump(experiment, os.path.join(kwargs["save_dir"], f"experiment_hnn_on_cpu.pkl"))

print(f"-> finished training gradient-descent based training of HNN on cpu, saved under {kwargs['save_dir']}")
