import torch; import numpy as np; from numpy import ndarray

from hamiltonian import BaseHamiltonian

from .grid import generate_uniform_train_test_set
from .flow_map import flow_map_rk45


def get_batch(x, step, batch_size, requires_grad, dtype,device):
    """
    Simple batching for traditionally trained networks

    @param x            : data
    @param step         : training step
    @param batch_size   : number of sampled per batch
    @param device       : put the batch into cpu or gpu

    @return             : torch tensor
    """
    # helper function for moving batches of train_x to/from GPU
    x_size, _ = x.shape

    i_begin = (step * batch_size) % x_size
    x_batch = x[i_begin:i_begin + batch_size, :]  # select next batch

    return torch.tensor(x_batch, requires_grad=requires_grad, dtype=dtype, device=device)

# ( (train_inputs, train_dt_truths, train_H_truths, train_H_grad_truths ), (train_x_0, train_x_0_H_truth) )
train_set_type = tuple[ tuple[tuple[ndarray, ndarray | None], ndarray, ndarray, ndarray], tuple[ndarray, ndarray] ]

# ( test_inputs, test_dt_truths, test_H_truths, test_H_grad_truths )
test_set_type = tuple[ ndarray, ndarray, ndarray, ndarray ]

def get_train_test_set(dof, target: BaseHamiltonian, train_size, test_size, q_lims, p_lims, rng=None, use_fd=False, dt_true=1e-4, dt_obs=1e-1) -> tuple[train_set_type, test_set_type]:
    """
    Given degree of freedom, and train and test sizes, sample train and test data

    @param dof          : degree of freedom
    @param target       : target Hamiltonian function
    @param train_size   : train set size, defaults to 10000
    @param test_size    : test set size, defaults to 2000
    @param q_lims       : domain limits for the "positions" q
    @param p_lims       : domain limits for the "momenta" p
    @param rng          : random number generator, defaults to None
    @param use_fd       : use finite differences, simulated a flow map to get the observations data
                          (limited data of (x, x_next) where the gradient is computed using finite differences)
    @param dt_true      : the time step used to simulate the true flow of the Hamiltonian
    @param dt_obs       : time differences between the observations x and x_next. Only important when use_fd is specified

    @return             : train_set, test_set
    """

    train_inputs, test_inputs = generate_uniform_train_test_set(
            dof,

            train_size,
            q_lims,
            p_lims,

            test_size,
            q_lims,
            p_lims,

            rng,
    )

    # prepare the train set

    # simulate exact flow if limited data is specified, gradients are computed using finite differences
    if use_fd:
        train_inputs_next = np.array([flow_map_rk45(x_i, target.H_grad, dt_flow_true=dt_true, dt_obs=dt_obs) for x_i in train_inputs])

        # finite differences to compute time derivatives first
        train_dt_truths = (train_inputs_next - train_inputs) / dt_obs

        J_inv = np.array([[0, -1],
                          [1, 0]])

        # hamilton's equations to compute target function derivatives
        train_H_grad_truths = (J_inv @ train_dt_truths.T).T
    else:
        train_inputs_next = None
        train_dt_truths = target.dt(train_inputs)
        train_H_grad_truths = target.H_grad(train_inputs)

    train_H_truths = target.H(train_inputs)

    # we assume that we know H(x_0)=y_0 for some x_0
    train_x_0 = np.zeros(dof * 2).reshape(1, -1)
    train_x_0_H_truth = target.H(train_x_0).reshape(1, -1)

    # prepare the test set
    test_dt_truths = target.dt(test_inputs)
    test_H_truths = target.H(test_inputs)
    test_H_grad_truths = target.H_grad(test_inputs)


    train_set = ( ((train_inputs, train_inputs_next), train_dt_truths, train_H_truths, train_H_grad_truths), (train_x_0, train_x_0_H_truth) )
    test_set = ( test_inputs, test_dt_truths, test_H_truths, test_H_grad_truths )

    return (train_set, test_set)
