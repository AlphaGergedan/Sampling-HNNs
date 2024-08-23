import torch; from torch import Tensor; import numpy as np; from numpy import ndarray

T = Tensor | ndarray

def is_zero_vector(x: T, atol=1e-15) -> bool:
    """
    @param x: either torch.Tensor or numpy.ndarray
    """
    if isinstance(x, Tensor):
        return torch.allclose(x, torch.zeros_like(x), atol=atol)
    else:
        return np.allclose(x, 0, atol=atol)
