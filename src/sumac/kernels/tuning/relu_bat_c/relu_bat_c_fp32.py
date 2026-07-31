import torch

from sumac.kernels.tuning.autotune import AutotuneCudaKernel
from sumac.kernels.relu_batc_jit.api import relu_bat_c_fused
from sumac.kernels.tuning.tuning_types import ReluBatCFp32TuneConfig, T_ReluBatCParams, T_ReluBatCReturn
from sumac.kernels.tuning.relu_bat_c import relu_bat_c_fallback


relu_bat_c_fp32_tune_config = ReluBatCFp32TuneConfig(
    BM = [32, 64, 128, 256],
    BK = [16, 32, 64],
    num_ms = [1, 2, 4, 6]
)


def _unpack_config(cfg: ReluBatCFp32TuneConfig):
    BM = cfg.BM[0]
    BK = cfg.BK[0]
    num_ms = cfg.num_ms[0]
    return (BM, BK, num_ms)


class AutotuneReluBatCFP32(AutotuneCudaKernel[ReluBatCFp32TuneConfig, T_ReluBatCParams, T_ReluBatCReturn]):
    def __init__(self, *args, **kwargs):
        kwargs['wrapped_fn_name'] = relu_bat_c_fused.__name__
        kwargs['wrapped_fn_module'] = relu_bat_c_fused.__module__
        super().__init__(*args, **kwargs)


    def _candidate_fn(self, params: T_ReluBatCParams, config: ReluBatCFp32TuneConfig):
        (A, B, C) = params
        (BM, BK, num_ms) = _unpack_config(config)
        return relu_bat_c_fused(A, B, C, BM=BM, BK=BK, MS=num_ms)


    def _fallback(self, params: T_ReluBatCParams) -> T_ReluBatCReturn:
        (A, B, C) = params
        return relu_bat_c_fallback(A, B, C)


    def _constraint(self, params: T_ReluBatCParams, config: ReluBatCFp32TuneConfig) -> bool:
        (A, _, _) = params
        (BM, BK, num_ms) = _unpack_config(config)

        if A.shape[1] >= 32 and num_ms > 2:
            return False
        if A.shape[1] >= 64 and num_ms > 1:
            return False

        props = torch.cuda.get_device_properties(A.device)
        if BM > getattr(props, "max_threads_per_block", 1024):
            return False
        shared_memory_per_block = getattr(props, "shared_memory_per_block", 0)
        return shared_memory_per_block >= 8 * BK * A.shape[1]
