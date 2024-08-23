from torch import Tensor; import numpy as np; from numpy import ndarray

from util import is_zero_vector

T = Tensor | ndarray


def l2_error_rel(y_true: T, y_pred: T) -> float | TypeError:
    """
    Returns relative L2 error given numpy.ndarrays or torch.tensors.

    @param y_true: ndarray or Tensor
    @param y_pred: ndarray or Tensor

    @return: Relative L2 error between the inputs
    """
    assert y_true.shape == y_pred.shape

    if isinstance(y_true, Tensor) and isinstance(y_pred, Tensor):
        error = (((y_true - y_pred)**2).sum()).sqrt()
        if not is_zero_vector(y_true):
            error /= ((y_true**2).sum()).sqrt()

        return error.detach().cpu().numpy().item()

    if isinstance(y_true, ndarray) and isinstance(y_pred, ndarray):
        # L2 error or relative L2 error
        error = np.sqrt(((y_true - y_pred)**2).sum())
        if not is_zero_vector(y_true):
            error /= np.sqrt((y_true**2).sum())
        return float(error)

    return TypeError(f"Inputs are of different types: y_true={type(y_true)} and y_pred={y_pred}")
