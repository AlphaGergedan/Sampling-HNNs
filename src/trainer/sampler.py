from typing import Any
import numpy as np

from model.sampled_network_type import SampledNetworkType
from trainer.param_sampler import ParameterSampler
from .base import BaseTrainer

class Sampler(BaseTrainer):
    sampling_type: SampledNetworkType
    param_sampler: ParameterSampler

    def __init__(self, sampling_type, param_sampler, **kwargs):
        super(Sampler, self).__init__()

        # for ELM, A_PRIORI distribution should be set for parameter sampler, and vice versa
        assert ((sampling_type is SampledNetworkType.ELM and param_sampler is ParameterSampler.A_PRIORI) or
               (sampling_type is not SampledNetworkType.ELM and param_sampler is not ParameterSampler.A_PRIORI))

        self.sampling_type = sampling_type
        self.param_sampler = param_sampler

    def train(self, model, train_inputs, train_dt_truths, train_input_x_0, train_input_x_0_H_truth, device, train_H_truths=None):
        super(Sampler, self).train(model, train_inputs, train_dt_truths, train_input_x_0, train_input_x_0_H_truth, device)

        from model.s_mlp import S_MLP
        from model.s_hnn import S_HNN

        # device parameter is ignored in sampling, it is only relevant for traditional network training
        # sampler only utilizes the cpu
        assert isinstance(model, S_MLP) or isinstance(model, S_HNN)

        if isinstance(model, S_MLP):
            # odenet directly outputs time derivatives
            model.pipeline.fit(train_inputs, train_dt_truths)

        elif isinstance(model, S_HNN):
            assert len(model.mlp.pipeline) == 2 # only support shallow networks for now

            # set the parameter sampler in the dense layer
            dense_layer: Any = model.mlp.pipeline[0]
            assert dense_layer.sample_uniformly
            dense_layer.parameter_sampler = self.param_sampler.value
            dense_layer.__post_init__() # init is required to use the actual param sampler, not the string

            # sample hidden layer weights (unsupervised if not SWIM, and with uniform sampling of the inputs)
            if self.sampling_type is SampledNetworkType.SWIM:
                assert train_H_truths is not None
                dense_layer.sample_uniformly = False
                dense_layer.fit(train_inputs, train_H_truths)
            else:
                dense_layer.fit(train_inputs)

            grad_last_hidden = model.mlp.compute_grad_last_hidden_wrt_input(train_inputs)
            last_hidden_out_x_0 = dense_layer.transform(train_input_x_0)

            # solve the linear layer weights using the linear system (here we incorporate Hamiltonian equations into the fitting)
            c = self.__fit_HNN_linear_layer(grad_last_hidden, last_hidden_out_x_0, train_dt_truths, train_input_x_0_H_truth, rcond=model.mlp.rcond).reshape(-1,1)

            linear_layer: Any = model.H_last_layer()
            linear_layer.weights = c[:-1].reshape((-1,1))
            linear_layer.biases = c[-1].reshape((1,1))
            linear_layer.layer_width = linear_layer.weights.shape[1]
            linear_layer.n_parameters = linear_layer.weights.size + linear_layer.biases.size

            # case: A-SWIM
            if self.sampling_type is SampledNetworkType.A_SWIM:
                # approximate the Hamiltonian values (target function values) which we need in other sampling methods
                train_preds = model.H(train_inputs)

                # resample with approximate values, for this we disable uniform sampling of inputs
                dense_layer.sample_uniformly = False

                # sample hidden layer weights (supervised with approximate values)
                dense_layer.fit(train_inputs, train_preds)

                grad_last_hidden = model.mlp.compute_grad_last_hidden_wrt_input(train_inputs)
                last_hidden_out_x_0 = dense_layer.transform(train_input_x_0)

                # solve again to fit the linear layer
                c = self.__fit_HNN_linear_layer(grad_last_hidden, last_hidden_out_x_0, train_dt_truths, train_input_x_0_H_truth, rcond=model.mlp.rcond).reshape(-1,1)

                linear_layer.weights = c[:-1].reshape((-1,1))
                linear_layer.biases = c[-1].reshape((1,1))

    def __fit_HNN_linear_layer(self, grad_last_hidden, last_hidden_out_x_0, train_dt_truths, train_input_x_0_truth, rcond):
        """
        Fits the last layer of the model by solving least squares,
        builds the matrix A and vector b and solves the linear equation for x (weights)

        @param grad_last_hidden         : gradients of hidden layer output w.r.t. input (K,D,M)
        @param last_hidden_out_x_0      : hidden layer output of x0 (1,M)
        @param y_train_derivs_true      : derivatives of target function w.r.t. X (K*D)
        @param train_input_x_0_truth    : true function value at input x0
        @param rcond                    : how approximately to solve the least squares

        @return c                       : solved x (weights of the final linear layer)
        """
        grad_last_hidden_q_part, grad_last_hidden_p_part = np.split(grad_last_hidden, 2, axis=1)

        (num_points, dof, last_hidden_width) = grad_last_hidden_q_part.shape
        assert num_points == grad_last_hidden_p_part.shape[0]
        assert dof == grad_last_hidden_p_part.shape[1]
        assert last_hidden_width == grad_last_hidden_p_part.shape[2]

        grad_last_hidden_q_part = grad_last_hidden_q_part.reshape(num_points*dof, last_hidden_width)
        grad_last_hidden_p_part = grad_last_hidden_p_part.reshape(num_points*dof, last_hidden_width)

        # for dof>1 we use reshape
        # grad_last_hidden_q_part = np.squeeze(grad_last_hidden_q_part, axis=1)
        # grad_last_hidden_p_part = np.squeeze(grad_last_hidden_p_part, axis=1)

        # Hamilton's Equations
        A = np.concatenate(( grad_last_hidden_p_part, -grad_last_hidden_q_part ), axis=0)
        A = np.concatenate(( A, last_hidden_out_x_0 ), axis=0)
        A = np.column_stack((A, np.concatenate(( np.zeros(A.shape[0] - 1), np.ones(1) ), axis=0) )) # for the bias term

        q_dot_truths, p_dot_truths = np.split(train_dt_truths, 2, axis=1)
        b = np.concatenate((
            q_dot_truths.ravel(),
            p_dot_truths.ravel(),
            train_input_x_0_truth.ravel(),
        ))

        # (ND + 1, M + 1)
        c = np.linalg.lstsq(A, b, rcond=rcond)[0]

        return c.reshape(-1, 1) # final shape (M+1, 1) == [weights, bias] of shapes (M,1) and (1,1)
