import torch; from torch import Tensor

from .device_type import DeviceType

def to_tensor(x, requires_grad=False, dtype=torch.float32, device=DeviceType.CPU) -> Tensor:
    """
    Converts given input into a tensor if it is not
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, requires_grad=requires_grad, dtype=dtype, device=device)

    return x
