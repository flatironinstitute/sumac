import torch

from enum import Enum

from sumac.kernels.cuda_utils import cuda_is_available
from sumac.kernels.tuning import active_kernel_autotune_options, KernelAutotuneOptions, AutotuneMode
from sumac.kernels.tuning.autotune import grid_size
from sumac.kernels.tuning.tuning_types import make_choices
from sumac.config import SumacConfig

from .relu_bat_c_base import relu_bat_c_fallback
from .relu_bat_c_fp32 import AutotuneReluBatCFP32, relu_bat_c_fp32_tune_config
from .relu_bat_c_tf32_sync import AutotuneReluBatCTf32Sync, relu_bat_c_tf32_sync_tune_config
from .relu_bat_c_tf32_wgmma import (
    AutotuneReluBatCTf32Wgmma,
    relu_bat_c_tf32_wgmma_tune_config,
    relu_bat_c_tf32_wgmma_available
)


class _KernelModes(Enum):
    FP32 = "fp32_cuda"
    TF32_SYNC = "tf32_mma_sync"
    TF32_WGMMA = "tf32_wgmma"
    FALLBACK = "fallback"


def _select_relu_bat_c_kernel_mode(
    allow_tf32: bool,
    device,
    dtype: torch.dtype = torch.float32,
) -> _KernelModes:
    device = torch.device(device)
    if dtype != torch.float32 or device.type != "cuda" or not cuda_is_available():
        return _KernelModes.FALLBACK
    if not allow_tf32:
        return _KernelModes.FP32

    props = torch.cuda.get_device_properties(device)
    if props.major == 9:
        return _KernelModes.TF32_WGMMA
    if props.major >= 8:
        return _KernelModes.TF32_SYNC
    return _KernelModes.FP32


# NOTE CHANGE: actually, let caller decide whether we need a new tuned kernel
T_KernelTuner = AutotuneReluBatCTf32Wgmma | AutotuneReluBatCTf32Sync | AutotuneReluBatCFP32
def get_tunable_kernel(cfg: SumacConfig):
    autotune_opts = active_kernel_autotune_options()
    mode = _select_relu_bat_c_kernel_mode(cfg.allow_tf32, cfg.device, cfg.dtype)

    if mode == _KernelModes.TF32_WGMMA:
        tune_config = relu_bat_c_tf32_wgmma_tune_config(cfg.rank)
        return AutotuneReluBatCTf32Wgmma(
            configs = tune_config,
            cache_path = "relu_bat_c_tf32_wgmma_mode_autotune.json",
            n_trials = grid_size(make_choices(tune_config)),
            autotune_options = autotune_opts
        )

    if mode == _KernelModes.TF32_SYNC:
        tune_config = relu_bat_c_tf32_sync_tune_config(cfg.rank)
        return AutotuneReluBatCTf32Sync(
            configs = tune_config,
            cache_path = "relu_bat_c_tf32_mma_autotune.json",
            n_trials = grid_size(make_choices(tune_config)),
            autotune_options = autotune_opts
        )

    assert mode in [_KernelModes.FP32, _KernelModes.FALLBACK]
    if mode == _KernelModes.FALLBACK:
        autotune_opts = KernelAutotuneOptions(mode = AutotuneMode.FALLBACK)
    return AutotuneReluBatCFP32(
        configs = relu_bat_c_fp32_tune_config,
        cache_path = "relu_bat_c_jit_autotune.json",
        autotune_options = autotune_opts
    )
