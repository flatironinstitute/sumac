from __future__ import annotations

from functools import lru_cache

import torch

from .cuda_utils import cuda_is_available
from .tuning import KernelAutotuneOptions


class FallbackReluBatReduce:
    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return relu_bat_reduce_fallback(A, B)

    def resolve_params(self, *args, **kwargs) -> dict:
        return {}


def relu_bat_reduce_fallback_launcher() -> FallbackReluBatReduce:
    return FallbackReluBatReduce()


def autotune_deps():
    from .tuning import autotune_cuda_kernel, relu_bat_reduce_key

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "CUDA relu_bat_reduce kernels require optional dependency 'optuna'. "
            "Install the CUDA/autotune extras to use custom CUDA kernels."
        ) from exc

    return autotune_cuda_kernel, relu_bat_reduce_key, optuna


def grid_size(config: dict) -> int:
    size = 1
    for values in config.values():
        size *= len(values)
    return size


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def relu_bat_reduce_tune_config() -> dict:
    return {
        "BM": [32, 64, 128, 256],
        "BK": [16, 32, 64, 128],
        "num_ms": [1, 2, 4, 6],
    }


@torch.compile
def relu_bat_reduce_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    A64 = A.to(torch.float64)
    B64 = B.to(torch.float64)
    Sr = torch.relu(A64 @ B64.T)

    sum_sr = Sr.sum()
    sum_sr2 = (Sr * Sr).sum()
    return sum_sr, sum_sr2


def relu_bat_reduce_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BK: int,
    num_ms: int,
) -> bool:
    _, D = A.shape
    props = torch.cuda.get_device_properties(A.device)

    if D >= 32 and num_ms > 4:
        return False
    if D >= 64 and num_ms > 2:
        return False
    if not is_power_of_two(BM):
        return False
    if BM > getattr(props, "max_threads_per_block", 1024):
        return False

    smem_bytes = 4 * BK * D + 2 * BM * 8
    return smem_bytes <= props.shared_memory_per_block


@lru_cache(maxsize=None)
def relu_bat_reduce_launcher(
    autotune_options: KernelAutotuneOptions | None = None,
):
    if not cuda_is_available():
        return relu_bat_reduce_fallback_launcher()

    autotune_cuda_kernel, relu_bat_reduce_key, optuna = autotune_deps()
    from .relu_bat_reduce_jit.api import relu_bat_reduce_fused

    tune_config = relu_bat_reduce_tune_config()

    @autotune_cuda_kernel(
        configs=tune_config,
        fallback_fn=relu_bat_reduce_fallback,
        constraint_fn=relu_bat_reduce_constraints,
        key_fn=relu_bat_reduce_key,
        cache_path="relu_bat_reduce_jit_autotune.json",
        n_trials=1000,
        warmup=1,
        rep=5,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
        autotune_options=autotune_options,
    )
    def relu_bat_reduce(
        A: torch.Tensor,
        B: torch.Tensor,
        BM: int,
        BK: int,
        num_ms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return relu_bat_reduce_fused(A, B, BM, BK, num_ms)

    return relu_bat_reduce
