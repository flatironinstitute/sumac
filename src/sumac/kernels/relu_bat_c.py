from __future__ import annotations

from functools import lru_cache

import torch

from .cuda_utils import cuda_is_available
from .tuning import KernelAutotuneOptions


class FallbackReluBatC:
    def __call__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        return relu_bat_c_fallback(A, B, C)

    def resolve_params(self, *args, **kwargs) -> dict:
        return {}


def relu_bat_c_fallback_launcher() -> FallbackReluBatC:
    return FallbackReluBatC()


def autotune_deps():
    from .tuning import autotune_cuda_kernel, relu_bat_c_key

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "CUDA relu_bat_c kernels require optional dependency 'optuna'. "
            "Install the CUDA/autotune extras to use custom CUDA kernels."
        ) from exc

    return autotune_cuda_kernel, relu_bat_c_key, optuna


def grid_size(config: dict) -> int:
    size = 1
    for values in config.values():
        size *= len(values)
    return size


def max_dynamic_smem_bytes(props) -> int:
    candidates = [
        int(getattr(props, "shared_memory_per_block", 0) or 0),
        int(getattr(props, "shared_memory_per_block_optin", 0) or 0),
    ]
    if getattr(props, "major", 0) == 9:
        candidates.append(227 * 1024)
    return max(candidates)


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def relu_bat_c_fp32_tune_config() -> dict:
    return {
        "BM": [32, 64, 128, 256],
        "BK": [16, 32, 64],
        "num_ms": [1, 2, 4, 6],
    }


def relu_bat_c_tf32_sync_tune_config(D: int) -> dict:
    if D == 64:
        return {
            "BM": [128, 256],
            "BN": [32, 64, 128],
            "M_TILES": [1, 2, 4],
            "num_stages": [1, 2, 3],
        }
    if D == 128:
        return {
            "BM": [64, 128, 256],
            "BN": [8, 16, 32],
            "M_TILES": [1, 2, 4],
            "num_stages": [1, 2, 3],
        }
    if D == 256:
        return {
            "BM": [64, 128, 256],
            "BN": [8, 16],
            "M_TILES": [1, 2],
            "num_stages": [1, 2],
        }

    return {
        "BM": [128, 256],
        "BN": [16, 32, 64],
        "M_TILES": [2, 4],
        "num_stages": [1, 2, 3],
    }


def relu_bat_c_tf32_wgmma_tune_config(D: int) -> dict:
    if D <= 16:
        return {
            "BM": [192, 256, 320],
            "BN": [64, 128, 256],
            "WGMMA_S_N": [32, 64],
            "WGMMA_Y_N": [16],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }
    if D <= 32:
        return {
            "BM": [192, 256, 320],
            "BN": [64, 128, 256],
            "WGMMA_S_N": [32, 64],
            "WGMMA_Y_N": [16, 32],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }
    if D <= 72:
        return {
            "BM": [128, 192, 256],
            "BN": [64, 128, 256],
            "WGMMA_S_N": [32, 64, 128],
            "WGMMA_Y_N": [32, 64],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }
    if D <= 144:
        return {
            "BM": [128, 192, 256],
            "BN": [32, 64, 128],
            "WGMMA_S_N": [32, 64, 128],
            "WGMMA_Y_N": [64, 128],
            "num_stages": [2],
            "wgmma_mode": ["RS", "SS"],
        }

    return {
        "BM": [64, 128, 192, 256],
        "BN": [16, 32, 64, 128],
        "WGMMA_S_N": [16, 32, 64],
        "WGMMA_Y_N": [64, 128],
        "num_stages": [2],
        "wgmma_mode": ["SS"],
    }


def relu_bat_c_fp32_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BK: int,
    num_ms: int,
) -> bool:
    if A.shape[1] >= 32 and num_ms > 2:
        return False
    if A.shape[1] >= 64 and num_ms > 1:
        return False

    props = torch.cuda.get_device_properties(A.device)
    if BM > getattr(props, "max_threads_per_block", 1024):
        return False
    return props.shared_memory_per_block >= 8 * BK * A.shape[1]


def relu_bat_c_tf32_sync_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
    num_stages: int,
) -> bool:
    _, D = A.shape
    props = torch.cuda.get_device_properties(A.device)

    if props.major < 8:
        return False
    if D < 1:
        return False
    if BN % 8 != 0:
        return False
    if num_stages < 1:
        return False

    warp_m_rows = M_TILES * 16
    if BM % warp_m_rows != 0:
        return False

    compute_warps = BM // warp_m_rows
    if compute_warps < 1 or compute_warps > 8:
        return False

    max_smem = getattr(
        props,
        "shared_memory_per_block_optin",
        props.shared_memory_per_block,
    )
    smem_bytes = 2 * num_stages * BN * round_up(D, 8) * 4 + 127
    return smem_bytes <= max_smem


def relu_bat_c_tf32_wgmma_available(A: torch.Tensor) -> bool:
    props = torch.cuda.get_device_properties(A.device)
    return props.major == 9


def relu_bat_c_tf32_wgmma_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    WGMMA_S_N: int,
    WGMMA_Y_N: int,
    num_stages: int,
    wgmma_mode: str,
) -> bool:
    _, D = A.shape
    props = torch.cuda.get_device_properties(A.device)

    if props.major != 9:
        return False
    if D < 1:
        return False
    if WGMMA_S_N not in (16, 32, 64, 128):
        return False
    if WGMMA_Y_N not in (16, 32, 64, 128):
        return False
    if BN % WGMMA_S_N != 0:
        return False
    if num_stages not in (1, 2, 3):
        return False
    if wgmma_mode not in ("RS", "SS"):
        return False
    if BM % 64 != 0:
        return False

    compute_warpgroups = BM // 64
    threads_per_block = (compute_warpgroups + 1) * 128
    if compute_warpgroups < 1:
        return False
    if threads_per_block > getattr(props, "max_threads_per_block", 1024):
        return False

    D_k_pad = round_up(D, 8)
    D_y_pad = round_up(D, WGMMA_Y_N)
    smem_elems = num_stages * BN * (D_k_pad + D_y_pad)
    if wgmma_mode == "SS":
        smem_elems += BM * D_k_pad
    smem_bytes = smem_elems * 4 + 127
    return smem_bytes <= max_dynamic_smem_bytes(props)


@torch.compile(mode="max-autotune-no-cudagraphs")
def relu_bat_c_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    return torch.relu(B @ A.T) @ C


@lru_cache(maxsize=None)
def relu_bat_c_cuda_launcher(
    autotune_options: KernelAutotuneOptions | None = None,
):
    if not cuda_is_available():
        return relu_bat_c_fallback_launcher()

    autotune_cuda_kernel, relu_bat_c_key, optuna = autotune_deps()
    from .relu_batc_jit.api import relu_bat_c_fused

    tune_config = relu_bat_c_fp32_tune_config()

    @autotune_cuda_kernel(
        configs=tune_config,
        fallback_fn=relu_bat_c_fallback,
        constraint_fn=relu_bat_c_fp32_constraints,
        key_fn=relu_bat_c_key,
        cache_path="relu_bat_c_jit_autotune.json",
        n_trials=1000,
        warmup=1,
        rep=5,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
        autotune_options=autotune_options,
    )
    def relu_batc(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BK: int,
        num_ms: int,
    ) -> torch.Tensor:
        return relu_bat_c_fused(
            A,
            B,
            C,
            BM=BM,
            BK=BK,
            MS=num_ms,
        )

    return relu_batc


@lru_cache(maxsize=None)
def relu_bat_c_tf32_sync_launcher(
    D: int,
    autotune_options: KernelAutotuneOptions | None = None,
):
    if not cuda_is_available():
        return relu_bat_c_fallback_launcher()

    autotune_cuda_kernel, relu_bat_c_key, optuna = autotune_deps()
    from .relu_batc_tf32_jit.api import (
        relu_bat_c_tf32_mma_sync,
    )

    tune_config = relu_bat_c_tf32_sync_tune_config(D)

    @autotune_cuda_kernel(
        configs=tune_config,
        fallback_fn=relu_bat_c_fallback,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_tf32_sync_constraints,
        cache_path="relu_bat_c_tf32_mma_autotune.json",
        n_trials=grid_size(tune_config),
        warmup=1,
        rep=5,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
        autotune_options=autotune_options,
    )
    def relu_bat_c_tf32_sync_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BN: int,
        M_TILES: int,
        num_stages: int,
    ) -> torch.Tensor:
        return relu_bat_c_tf32_mma_sync(
            A,
            B,
            C,
            BM=BM,
            BN=BN,
            M_TILES=M_TILES,
            num_stages=num_stages,
        )

    return relu_bat_c_tf32_sync_cuda


@lru_cache(maxsize=None)
def relu_bat_c_tf32_wgmma_launcher(
    D: int,
    autotune_options: KernelAutotuneOptions | None = None,
):
    if not cuda_is_available():
        return relu_bat_c_fallback_launcher()

    autotune_cuda_kernel, relu_bat_c_key, optuna = autotune_deps()
    from .relu_batc_tf32_jit.api import (
        relu_bat_c_tf32_wgmma,
    )

    tune_config = relu_bat_c_tf32_wgmma_tune_config(D)

    @autotune_cuda_kernel(
        configs=tune_config,
        fallback_fn=relu_bat_c_fallback,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_tf32_wgmma_constraints,
        cache_path="relu_bat_c_tf32_wgmma_mode_autotune.json",
        n_trials=grid_size(tune_config),
        warmup=1,
        rep=5,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
        autotune_options=autotune_options,
    )
    def relu_bat_c_tf32_wgmma_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BN: int,
        WGMMA_S_N: int,
        WGMMA_Y_N: int,
        num_stages: int,
        wgmma_mode: str,
    ) -> torch.Tensor:
        return relu_bat_c_tf32_wgmma(
            A,
            B,
            C,
            BM=BM,
            BN=BN,
            WGMMA_S_N=WGMMA_S_N,
            WGMMA_Y_N=WGMMA_Y_N,
            num_stages=num_stages,
            wgmma_mode=wgmma_mode,
        )

    return relu_bat_c_tf32_wgmma_cuda


def select_relu_bat_c_kernel_mode(
    allow_tf32: bool,
    device,
    dtype: torch.dtype = torch.float32,
) -> str:
    device = torch.device(device)
    if dtype != torch.float32 or device.type != "cuda" or not cuda_is_available():
        return "fallback"
    if not allow_tf32:
        return "fp32_cuda"

    props = torch.cuda.get_device_properties(device)
    if props.major == 9:
        return "tf32_wgmma"
    if props.major >= 8:
        return "tf32_mma_sync"
    return "fp32_cuda"
