"""
This file conducts all the single pendulum frequency experiments
"""

# system libs
import os; from time import time; from typing import Any

# data
import numpy as np
from data import get_train_test_set

# logger, model saving
from joblib import dump
from experiment import SampledDomainParams, SampledModelParams, SampledExperiment, SampledModelResults, SampledModels
from hamiltonian.single_pendulum import SinglePendulum

# argument parser
from argparser import sampled_hnn_experiment_argparser

# sampling params
from model.s_hnn import S_HNN
from activation.type import ActivationType
from model.sampled_network_type import SampledNetworkType
from trainer.param_sampler import ParameterSampler
from trainer.sampler import Sampler

# does not have any affect on sampling, only for traditional hnn
from util.device_type import DeviceType

kwargs: dict[str, Any] = sampled_hnn_experiment_argparser()

####################################################################################################
#           EXPERIMENT SINGLE PENDULUM ON DOMAIN [-pi,pi]x[-0.5,0.5] FREQUENCY (10) TRAIN SIZE SCALING

single_pendulum = SinglePendulum(f=10.)
q_lims = [ [-np.pi, np.pi] ]
p_lims = [ [-0.5, 0.5] ]
kwargs["min_input"] = np.min([q_lims, p_lims])
kwargs["max_input"] = np.max([q_lims, p_lims])

for train_size in range(1000, 21000, 1000): #[1000, 2000, .. , 20000]
    START_MODEL_RANDOM_SEED = kwargs["start_model_random_seed"]
    START_DATA_RANDOM_SEED = kwargs["start_data_random_seed"]

    domain_params = SampledDomainParams(single_pendulum, q_lims, p_lims, train_size, 10000, kwargs["repeat"], START_DATA_RANDOM_SEED)
    model_params = SampledModelParams(ActivationType.TANH, network_width=1500, resample_duplicates=kwargs["resample_duplicates"], rcond=kwargs["rcond"], elm_bias_start=kwargs["min_input"], elm_bias_end=kwargs["max_input"], start_model_random_seed=START_MODEL_RANDOM_SEED)

    elm_models = []; uswim_models = []; aswim_models = []; swim_models = []
    elm_test_errors = []; uswim_test_errors = []; aswim_test_errors = []; swim_test_errors = []
    elm_train_times = []; uswim_train_times = []; aswim_train_times = []; swim_train_times = []

    print(f"-> starting experiment with\ndomain params: {domain_params}\nmodel params: {model_params}")

    MODEL_RANDOM_SEED = START_MODEL_RANDOM_SEED
    DATA_RANDOM_SEED = START_DATA_RANDOM_SEED
    for run_index in range(domain_params.repeat):
        train_set, test_set = get_train_test_set(1, single_pendulum, domain_params.train_size, domain_params.test_size, domain_params.q_lims, domain_params.p_lims, rng = np.random.default_rng(DATA_RANDOM_SEED))

        ( (train_inputs, train_dt_truths, train_H_truths, train_H_grad_truths), (train_x_0, train_x_0_H_truth) ) = train_set
        ( test_inputs, test_dt_truths, test_H_truths, test_H_grad_truths ) = test_set

        # ELM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.ELM, ParameterSampler.A_PRIORI)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU)
        time_end = time()

        elm_models.append(model); elm_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        elm_test_errors.append(error_H);

        # U-SWIM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.U_SWIM, ParameterSampler.TANH)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU)
        time_end = time()

        uswim_models.append(model); uswim_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        uswim_test_errors.append(error_H)

        # A-SWIM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.A_SWIM, ParameterSampler.TANH)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU)
        time_end = time()

        aswim_models.append(model); aswim_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        aswim_test_errors.append(error_H)

        # SWIM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.SWIM, ParameterSampler.TANH)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU, train_H_truths)
        time_end = time()

        swim_models.append(model); swim_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        swim_test_errors.append(error_H)

        MODEL_RANDOM_SEED += 1
        DATA_RANDOM_SEED += 1
        print("model random seed is now", MODEL_RANDOM_SEED)
        print("data random seed is now", DATA_RANDOM_SEED)


    models = SampledModels(elm_models, uswim_models, aswim_models, swim_models)
    results = SampledModelResults(
            elm_test_errors, uswim_test_errors, aswim_test_errors, swim_test_errors,
            elm_train_times, uswim_train_times, aswim_train_times, swim_train_times,
    )
    experiment = SampledExperiment(domain_params, model_params, results)

    dump(models, os.path.join(kwargs["save_dir"], f"freq_10_models_train_size_{train_size}.pkl"))
    dump(experiment, os.path.join(kwargs["save_dir"], f"freq_10_experiment_train_size_{train_size}.pkl"))
    print(f"-> finished freq 10 train size {train_size}")


####################################################################################################
#           EXPERIMENT SINGLE PENDULUM ON DOMAIN [-pi,pi]x[-0.5,0.5] FREQUENCY (15) TRAIN SIZE SCALING

single_pendulum = SinglePendulum(f=15.)
q_lims = [ [-np.pi, np.pi] ]
p_lims = [ [-0.5, 0.5] ]
kwargs["min_input"] = np.min([q_lims, p_lims])
kwargs["max_input"] = np.max([q_lims, p_lims])

for train_size in range(1000, 21000, 1000): #[1000, 2000, .. , 20000]
    START_MODEL_RANDOM_SEED = kwargs["start_model_random_seed"]
    START_DATA_RANDOM_SEED = kwargs["start_data_random_seed"]

    domain_params = SampledDomainParams(single_pendulum, q_lims, p_lims, train_size, 10000, kwargs["repeat"], START_DATA_RANDOM_SEED)
    model_params = SampledModelParams(ActivationType.TANH, network_width=1500, resample_duplicates=kwargs["resample_duplicates"], rcond=kwargs["rcond"], elm_bias_start=kwargs["min_input"], elm_bias_end=kwargs["max_input"], start_model_random_seed=START_MODEL_RANDOM_SEED)

    elm_models = []; uswim_models = []; aswim_models = []; swim_models = []
    elm_test_errors = []; uswim_test_errors = []; aswim_test_errors = []; swim_test_errors = []
    elm_train_times = []; uswim_train_times = []; aswim_train_times = []; swim_train_times = []

    print(f"-> starting experiment with\ndomain params: {domain_params}\nmodel params: {model_params}")

    MODEL_RANDOM_SEED = START_MODEL_RANDOM_SEED
    DATA_RANDOM_SEED = START_DATA_RANDOM_SEED
    for run_index in range(domain_params.repeat):
        train_set, test_set = get_train_test_set(1, single_pendulum, domain_params.train_size, domain_params.test_size, domain_params.q_lims, domain_params.p_lims, rng = np.random.default_rng(DATA_RANDOM_SEED))

        ( (train_inputs, train_dt_truths, train_H_truths, train_H_grad_truths), (train_x_0, train_x_0_H_truth) ) = train_set
        ( test_inputs, test_dt_truths, test_H_truths, test_H_grad_truths ) = test_set

        # ELM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.ELM, ParameterSampler.A_PRIORI)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU)
        time_end = time()

        elm_models.append(model); elm_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        elm_test_errors.append(error_H);

        # U-SWIM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.U_SWIM, ParameterSampler.TANH)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU)
        time_end = time()

        uswim_models.append(model); uswim_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        uswim_test_errors.append(error_H)

        # A-SWIM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.A_SWIM, ParameterSampler.TANH)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU)
        time_end = time()

        aswim_models.append(model); aswim_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        aswim_test_errors.append(error_H)

        # SWIM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.SWIM, ParameterSampler.TANH)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU, train_H_truths)
        time_end = time()

        swim_models.append(model); swim_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        swim_test_errors.append(error_H)

        MODEL_RANDOM_SEED += 1
        DATA_RANDOM_SEED += 1
        print("model random seed is now", MODEL_RANDOM_SEED)
        print("data random seed is now", DATA_RANDOM_SEED)


    models = SampledModels(elm_models, uswim_models, aswim_models, swim_models)
    results = SampledModelResults(
            elm_test_errors, uswim_test_errors, aswim_test_errors, swim_test_errors,
            elm_train_times, uswim_train_times, aswim_train_times, swim_train_times,
    )
    experiment = SampledExperiment(domain_params, model_params, results)

    dump(models, os.path.join(kwargs["save_dir"], f"freq_15_models_train_size_{train_size}.pkl"))
    dump(experiment, os.path.join(kwargs["save_dir"], f"freq_15_experiment_train_size_{train_size}.pkl"))
    print(f"-> finished freq 10 train size {train_size}")


####################################################################################################
#           EXPERIMENT SINGLE PENDULUM ON DOMAIN [-pi,pi]x[-0.5,0.5] FREQUENCY (20) TRAIN SIZE SCALING

single_pendulum = SinglePendulum(f=20.)
q_lims = [ [-np.pi, np.pi] ]
p_lims = [ [-0.5, 0.5] ]
kwargs["min_input"] = np.min([q_lims, p_lims])
kwargs["max_input"] = np.max([q_lims, p_lims])

for train_size in range(1000, 21000, 1000): #[1000, 2000, .. , 20000]
    START_MODEL_RANDOM_SEED = kwargs["start_model_random_seed"]
    START_DATA_RANDOM_SEED = kwargs["start_data_random_seed"]

    domain_params = SampledDomainParams(single_pendulum, q_lims, p_lims, train_size, 10000, kwargs["repeat"], START_DATA_RANDOM_SEED)
    model_params = SampledModelParams(ActivationType.TANH, network_width=1500, resample_duplicates=kwargs["resample_duplicates"], rcond=kwargs["rcond"], elm_bias_start=kwargs["min_input"], elm_bias_end=kwargs["max_input"], start_model_random_seed=START_MODEL_RANDOM_SEED)

    elm_models = []; uswim_models = []; aswim_models = []; swim_models = []
    elm_test_errors = []; uswim_test_errors = []; aswim_test_errors = []; swim_test_errors = []
    elm_train_times = []; uswim_train_times = []; aswim_train_times = []; swim_train_times = []

    print(f"-> starting experiment with\ndomain params: {domain_params}\nmodel params: {model_params}")

    MODEL_RANDOM_SEED = START_MODEL_RANDOM_SEED
    DATA_RANDOM_SEED = START_DATA_RANDOM_SEED
    for run_index in range(domain_params.repeat):
        train_set, test_set = get_train_test_set(1, single_pendulum, domain_params.train_size, domain_params.test_size, domain_params.q_lims, domain_params.p_lims, rng = np.random.default_rng(DATA_RANDOM_SEED))

        ( (train_inputs, train_dt_truths, train_H_truths, train_H_grad_truths), (train_x_0, train_x_0_H_truth) ) = train_set
        ( test_inputs, test_dt_truths, test_H_truths, test_H_grad_truths ) = test_set

        # ELM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.ELM, ParameterSampler.A_PRIORI)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU)
        time_end = time()

        elm_models.append(model); elm_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        elm_test_errors.append(error_H);

        # U-SWIM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.U_SWIM, ParameterSampler.TANH)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU)
        time_end = time()

        uswim_models.append(model); uswim_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        uswim_test_errors.append(error_H)

        # A-SWIM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.A_SWIM, ParameterSampler.TANH)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU)
        time_end = time()

        aswim_models.append(model); aswim_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        aswim_test_errors.append(error_H)

        # SWIM
        model = S_HNN(input_dim=2, hidden_dim=model_params.network_width, activation=ActivationType.TANH, resample_duplicates=model_params.resample_duplicates, rcond=model_params.rcond, random_seed=MODEL_RANDOM_SEED, elm_bias_start=model_params.elm_bias_start, elm_bias_end=model_params.elm_bias_end)
        sampler = Sampler(SampledNetworkType.SWIM, ParameterSampler.TANH)
        time_begin = time()
        sampler.train(model, train_inputs, train_dt_truths, train_x_0, train_x_0_H_truth, DeviceType.CPU, train_H_truths)
        time_end = time()

        swim_models.append(model); swim_train_times.append(time_end-time_begin)

        error_H = model.evaluate_H(test_inputs, test_H_truths); assert isinstance(error_H, float)
        swim_test_errors.append(error_H)

        MODEL_RANDOM_SEED += 1
        DATA_RANDOM_SEED += 1
        print("model random seed is now", MODEL_RANDOM_SEED)
        print("data random seed is now", DATA_RANDOM_SEED)


    models = SampledModels(elm_models, uswim_models, aswim_models, swim_models)
    results = SampledModelResults(
            elm_test_errors, uswim_test_errors, aswim_test_errors, swim_test_errors,
            elm_train_times, uswim_train_times, aswim_train_times, swim_train_times,
    )
    experiment = SampledExperiment(domain_params, model_params, results)

    dump(models, os.path.join(kwargs["save_dir"], f"freq_20_models_train_size_{train_size}.pkl"))
    dump(experiment, os.path.join(kwargs["save_dir"], f"freq_20_experiment_train_size_{train_size}.pkl"))
    print(f"-> finished freq 10 train size {train_size}")
