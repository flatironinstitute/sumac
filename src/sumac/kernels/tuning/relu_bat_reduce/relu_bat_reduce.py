import torch
from torch import Tensor

from sumac.utils import is_power_of_two
from sumac.kernels.relu_bat_reduce_jit.api import relu_bat_reduce_fused
from sumac.kernels.tuning.autotune import AutotuneCudaKernel
from sumac.kernels.tuning.tuning_types import (
    KernelAutotuneOptions,
    ReluBatReduceTuneConfig,
    T_ReluBatReduceParams,
    T_ReluBatReduceReturn,
    make_config_list,
)


@torch.compile
def relu_bat_reduce_fallback(
    A: Tensor,
    B: Tensor,
) -> tuple[Tensor, Tensor]:
    A64 = A.to(torch.float64)
    B64 = B.to(torch.float64)
    Sr = torch.relu(A64 @ B64.T)

    sum_sr = Sr.sum()
    sum_sr2 = (Sr * Sr).sum()
    return sum_sr, sum_sr2


default_config = ReluBatReduceTuneConfig(BM = 32, BK = 16, num_ms = 1)
relu_bat_reduce_tune_config = make_config_list(
    default_config,
    {
        'BM': [32, 64, 128, 256],
        'BK': [16, 32, 64, 128],
        'num_ms': [1, 2, 4, 6]    
    }
)


class AutotuneReluBatReduce(AutotuneCudaKernel[ReluBatReduceTuneConfig, T_ReluBatReduceParams, T_ReluBatReduceReturn]):
    def __init__(self, *args, **kwargs):
        default_cache_path = "relu_bat_reduce_jit_autotune.json"
        kwargs['wrapped_fn_name'] = relu_bat_reduce_fused.__name__
        kwargs['wrapped_fn_module'] = relu_bat_reduce_fused.__module__
        kwargs['configs'] = kwargs.get('configs', relu_bat_reduce_tune_config) or relu_bat_reduce_tune_config
        kwargs['cache_path'] = kwargs.get('cache_path', default_cache_path) or default_cache_path
        super().__init__(*args, **kwargs)


    def resolve_decision(self, params: T_ReluBatReduceParams):
        if getattr(torch.version, "hip", None) is not None:
            self._set_interface_fn()
            return
        return super().resolve_decision(params)


    def _candidate_fn(self, params: T_ReluBatReduceParams, config: ReluBatReduceTuneConfig):
        (A, B) = params
        return relu_bat_reduce_fused(A, B, config.BM, config.BK, config.num_ms)


    def _fallback(self, params: T_ReluBatReduceParams) -> T_ReluBatReduceReturn:
        (A, B) = params
        return relu_bat_reduce_fallback(A, B)


    def _constraint(self, params: T_ReluBatReduceParams, config: ReluBatReduceTuneConfig) -> bool:
        (A, _) = params
        _, D = A.shape
        props = torch.cuda.get_device_properties(A.device)

        if D >= 32 and config.num_ms > 4: return False
        if D >= 64 and config.num_ms > 2: return False
        if not is_power_of_two(config.BM): return False
        if config.BM > getattr(props, "max_threads_per_block", 1024): return False

        shared_memory_per_block = int(getattr(props, "shared_memory_per_block", 0))
        smem_bytes = 4 * config.BK * D + 2 * config.BM * 8
        return smem_bytes <= shared_memory_per_block


def make_relu_bat_reduce(
    autotune_opts: KernelAutotuneOptions | None = None,
):
    if getattr(torch.version, "hip", None) is not None:
        from .relu_bat_reduce_fp32_mfma_amd import (
            AutotuneReluBatReduceMfmaAMD,
        )

        return AutotuneReluBatReduceMfmaAMD(
            autotune_options=autotune_opts,
        )
    return AutotuneReluBatReduce(
        autotune_options=autotune_opts,
    )
