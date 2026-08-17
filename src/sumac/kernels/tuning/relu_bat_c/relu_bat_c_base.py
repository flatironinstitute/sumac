import torch
from torch import Tensor

@torch.compile(mode="max-autotune-no-cudagraphs")
def relu_bat_c_fallback(A: Tensor, B: Tensor, C: Tensor) -> Tensor:
    return torch.relu(B @ A.T) @ C

