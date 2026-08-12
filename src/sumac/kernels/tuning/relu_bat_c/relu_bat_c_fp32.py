import torch

from sumac.kernels.tuning.autotune import AutotuneCudaKernel
from sumac.kernels.relu_batc_jit.api import relu_bat_c_fused
from sumac.kernels.tuning.tuning_types import ReluBatCFp32TuneConfig, T_ReluBatCParams, T_ReluBatCReturn, make_config_list
from sumac.kernels.tuning.relu_bat_c.relu_bat_c_base import relu_bat_c_fallback

default_config = ReluBatCFp32TuneConfig(BM = 32, BK = 16, num_ms = 1)
relu_bat_c_fp32_tune_config = make_config_list(
    default_config,
    {
        'BM': [32, 64, 128, 256],
        'BK': [16, 32, 64],
        'num_ms': [1, 2, 4, 6]
    }
)


class AutotuneReluBatCFP32(AutotuneCudaKernel[ReluBatCFp32TuneConfig, T_ReluBatCParams, T_ReluBatCReturn]):
    def __init__(self, *args, **kwargs):
        kwargs['wrapped_fn_name'] = relu_bat_c_fused.__name__
        kwargs['wrapped_fn_module'] = relu_bat_c_fused.__module__
        super().__init__(*args, **kwargs)


    def _candidate_fn(self, params: T_ReluBatCParams, config: ReluBatCFp32TuneConfig):
        (A, B, C) = params
        return relu_bat_c_fused(A, B, C, BM=config.BM, BK=config.BK, MS=config.num_ms)


    def _fallback(self, params: T_ReluBatCParams) -> T_ReluBatCReturn:
        (A, B, C) = params
        return relu_bat_c_fallback(A, B, C)


    def _constraint(self, params: T_ReluBatCParams, config: ReluBatCFp32TuneConfig) -> bool:
        (A, _, _) = params
        _, D = A.shape

        if D >= 32 and config.num_ms > 2:
            return False
        if D >= 64 and config.num_ms > 1:
            return False

        props = torch.cuda.get_device_properties(A.device)
        if config.BM > getattr(props, "max_threads_per_block", 1024):
            return False
        shared_memory_per_block = getattr(props, "shared_memory_per_block", 0)
        return shared_memory_per_block >= 8 * config.BK * D
