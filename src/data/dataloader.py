import torch; import numpy as np; from numpy import ndarray

from hamiltonian import BaseHamiltonian

from .grid import generate_uniform_train_test_set


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
train_set_type = tuple[ tuple[ndarray, ndarray, ndarray, ndarray], tuple[ndarray, ndarray] ]

# ( test_inputs, test_dt_truths, test_H_truths, test_H_grad_truths )
test_set_type = tuple[ ndarray, ndarray, ndarray, ndarray ]

def get_train_test_set(dof, target: BaseHamiltonian, train_size, test_size, q_lims, p_lims, rng=None) -> tuple[train_set_type, test_set_type]:
    """
    Given degree of freedom, and train and test sizes, sample train and test data

    @param dof          : degree of freedom
    @param target       : target Hamiltonian function
    @param train_size   : train set size, defaults to 10000
    @param test_size    : test set size, defaults to 2000
    @param q_lims       : domain limits for the "positions" q
    @param p_lims       : domain limits for the "momenta" p
    @param rng          : random number generator, defaults to None

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
    train_dt_truths = target.dt(train_inputs)
    train_H_truths = target.H(train_inputs)
    train_H_grad_truths = target.H_grad(train_inputs)

    # we assume that we know H(x_0)=y_0 for some x_0
    train_x_0 = np.zeros(dof * 2).reshape(1, -1)
    train_x_0_H_truth = target.H(train_x_0).reshape(1, -1)

    # prepare the test set
    test_dt_truths = target.dt(test_inputs)
    test_H_truths = target.H(test_inputs)
    test_H_grad_truths = target.H_grad(test_inputs)


    train_set = ( (train_inputs, train_dt_truths, train_H_truths, train_H_grad_truths), (train_x_0, train_x_0_H_truth) )
    test_set = ( test_inputs, test_dt_truths, test_H_truths, test_H_grad_truths )

    return (train_set, test_set)
