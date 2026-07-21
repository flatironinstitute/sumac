import torch 
import random
import numpy as np

# TODO: Consider merging with cuda_utils
from sumac.kernels.cuda_utils import cuda_is_available
from sumac.config.options import SumacConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_is_available():
        torch.cuda.manual_seed(seed)


def memory_stats():
    if not cuda_is_available():
        print("CUDA is not available")
        return
    print(f'allocated memory {torch.cuda.memory_allocated()/1024**2}')
    print(f'reserved memory {torch.cuda.memory_reserved()/1024**2}')


def resolve_sumac_device(
    S_index: torch.Tensor,
    S_value: torch.Tensor,
    config: SumacConfig
) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)

    if S_index.device == S_value.device:
        return S_value.device
    if S_value.device.type != "cpu" and S_index.device.type == "cpu":
        return S_value.device
    if S_index.device.type != "cpu" and S_value.device.type == "cpu":
        return S_index.device
    raise ValueError(
        "S_index and S_value are on different non-CPU devices. "
        "Pass device=... explicitly to choose the SUMAC training device."
    )
