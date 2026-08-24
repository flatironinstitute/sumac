import torch

from enum import Enum

from sumac.kernels.cuda_utils import cuda_is_available
from sumac.kernels.relu_batc_jit_amd.api import (
    relu_bat_c_fp32_mfma_available,
)
from sumac.kernels.relu_batc_tf32_jit_amd.api import (
    relu_bat_c_tf32_mfma_available,
)
from sumac.kernels.tuning import active_kernel_autotune_options, KernelAutotuneOptions, AutotuneMode
from sumac.kernels.tuning.autotune import grid_size
from sumac.kernels.tuning.tuning_types import make_choices
from sumac.config import SumacConfig

from .relu_bat_c_base import relu_bat_c_fallback
from .relu_bat_c_fp32 import AutotuneReluBatCFP32, relu_bat_c_fp32_tune_config
from .relu_bat_c_fp32_mfma_amd import (
    AutotuneReluBatCFP32MfmaAMD,
    relu_bat_c_fp32_mfma_tune_config,
)
from .relu_bat_c_tf32_mfma_amd import (
    AutotuneReluBatCTf32MfmaAMD,
    relu_bat_c_tf32_mfma_tune_config,
)
from .relu_bat_c_tf32_sync import AutotuneReluBatCTf32Sync, relu_bat_c_tf32_sync_tune_config
from .relu_bat_c_tf32_wgmma import (
    AutotuneReluBatCTf32Wgmma,
    relu_bat_c_tf32_wgmma_tune_config,
    relu_bat_c_tf32_wgmma_available
)


class _KernelModes(Enum):
    FP32 = "fp32_cuda"
    FP32_AMD = "fp32_amd"
    TF32_AMD = "tf32_amd"
    TF32_SYNC = "tf32_mma_sync"
    TF32_WGMMA = "tf32_wgmma"
    FALLBACK = "fallback"


def _select_relu_bat_c_kernel_mode(
    allow_tf32: bool,
    device,
    dtype: torch.dtype = torch.float32,
    rank: int | None = None,
) -> _KernelModes:
    device = torch.device(device)
    if dtype != torch.float32 or device.type != "cuda" or not cuda_is_available():
        return _KernelModes.FALLBACK
    if getattr(torch.version, "hip", None) is not None:
        if allow_tf32 and relu_bat_c_tf32_mfma_available(device, rank):
            return _KernelModes.TF32_AMD
        if relu_bat_c_fp32_mfma_available(device, rank):
            return _KernelModes.FP32_AMD
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
T_KernelTuner = (
    AutotuneReluBatCTf32Wgmma
    | AutotuneReluBatCTf32Sync
    | AutotuneReluBatCFP32
    | AutotuneReluBatCFP32MfmaAMD
    | AutotuneReluBatCTf32MfmaAMD
)


def get_tunable_kernel(cfg: SumacConfig):
    autotune_opts = active_kernel_autotune_options()
    mode = _select_relu_bat_c_kernel_mode(
        cfg.allow_tf32,
        cfg.device,
        cfg.dtype,
        cfg.rank,
    )

    if mode == _KernelModes.FP32_AMD:
        tune_config = relu_bat_c_fp32_mfma_tune_config(cfg.rank)
        return AutotuneReluBatCFP32MfmaAMD(
            configs=tune_config,
            cache_path="relu_bat_c_hip_fp32_mfma_runtime_autotune.json",
            n_trials=grid_size(make_choices(tune_config)),
            autotune_options=autotune_opts,
        )

    if mode == _KernelModes.TF32_AMD:
        tune_config = relu_bat_c_tf32_mfma_tune_config(cfg.rank)
        return AutotuneReluBatCTf32MfmaAMD(
            configs=tune_config,
            cache_path="relu_bat_c_hip_tf32_mfma_runtime_autotune.json",
            n_trials=grid_size(make_choices(tune_config)),
            autotune_options=autotune_opts,
        )

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
        if getattr(torch.version, "hip", None) is not None:
            if cfg.allow_tf32:
                tune_config = relu_bat_c_tf32_mfma_tune_config(cfg.rank)
                tuner_type = AutotuneReluBatCTf32MfmaAMD
                cache_path = "relu_bat_c_hip_tf32_mfma_runtime_autotune.json"
            else:
                tune_config = relu_bat_c_fp32_mfma_tune_config(cfg.rank)
                tuner_type = AutotuneReluBatCFP32MfmaAMD
                cache_path = "relu_bat_c_hip_fp32_mfma_runtime_autotune.json"
            return tuner_type(
                configs=tune_config,
                cache_path=cache_path,
                n_trials=grid_size(make_choices(tune_config)),
                autotune_options=autotune_opts,
            )
    return AutotuneReluBatCFP32(
        configs = relu_bat_c_fp32_tune_config,
        cache_path = "relu_bat_c_jit_autotune.json",
        autotune_options = autotune_opts
    )
