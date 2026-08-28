from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from ..compile_utils import compile_hip_kernel, hip_device_arch


_KERNEL_PATH_MFMA = Path(__file__).with_name("kernel_mfma.cpp")
_KERNEL_MARKER = "// KERNEL_START"

_MFMA_FP32_WAVE64_ARCHES = {"gfx90a", "gfx942", "gfx950"}
_MFMA_FP32_WAVE64_ARCHES_DESCRIPTION = "gfx90a, gfx942, or gfx950"


def _make_header_mfma(
    BM: int,
    BN: int,
    M_TILES: int,
    D_f: int,
    kernel_name: str,
) -> str:
    return f"""
#define BM {BM}
#define BN {BN}
#define M_TILES {M_TILES}
#define D_f {D_f}
#define REDUCE_MFMA_FP32_KERNEL_NAME {kernel_name}
"""


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _base_gpu_arch(gpu_arch: str) -> str:
    return gpu_arch.split(":", 1)[0]


def _mfma_dynamic_smem_bytes(
    *,
    BN: int,
    D_f: int,
    threads_per_block: int,
) -> int:
    panel_words = BN * D_f
    reduction_words = 2 * threads_per_block
    return (panel_words + reduction_words) * 4


def _mfma_register_floor(*, M_TILES: int, D_f: int) -> int:
    a_words = M_TILES * D_f // 4
    score_words = 4 * M_TILES
    reduction_words = 2
    return a_words + score_words + reduction_words


def _max_shared_memory_per_block(properties) -> int:
    default_limit = int(
        getattr(properties, "shared_memory_per_block", 0) or 0
    )
    optin_limit = int(
        getattr(properties, "shared_memory_per_block_optin", None)
        or default_limit
    )
    return max(default_limit, optin_limit)


def _split_kernel_file(path: Path) -> tuple[str, str, int]:
    source = path.read_text()
    marker_index = source.find(_KERNEL_MARKER)
    if marker_index < 0:
        raise RuntimeError(f"Missing marker {_KERNEL_MARKER!r} in {path}")

    marker_end = marker_index + len(_KERNEL_MARKER)
    remainder = source[marker_end:]
    kernel_source = remainder.lstrip()
    kernel_index = marker_end + len(remainder) - len(kernel_source)
    kernel_source_line = source.count("\n", 0, kernel_index) + 1
    return source[:marker_index], kernel_source, kernel_source_line


@lru_cache(maxsize=None)
def get_relu_bat_reduce_kernel_mfma(
    BM: int,
    BN: int,
    M_TILES: int,
    D_f: int,
    gpu_arch: str,
):
    if _base_gpu_arch(gpu_arch) not in _MFMA_FP32_WAVE64_ARCHES:
        raise RuntimeError(
            "The FP32 MFMA reduction kernel requires "
            f"{_MFMA_FP32_WAVE64_ARCHES_DESCRIPTION}; "
            f"got {gpu_arch}"
        )
    kernel_name = f"relu_bat_reduce_fp32_mfma_d{D_f}"
    header_code, kernel_source, kernel_source_line = _split_kernel_file(
        _KERNEL_PATH_MFMA
    )

    header_code = (
        _make_header_mfma(BM, BN, M_TILES, D_f, kernel_name)
        + "\n"
        + f'#line 1 "{_KERNEL_PATH_MFMA}"\n'
        + header_code
        + "\n"
        + f'#line {kernel_source_line} "{_KERNEL_PATH_MFMA}"\n'
    )
    return compile_hip_kernel(
        kernel_source,
        kernel_name=kernel_name,
        header_code=header_code,
        gpu_arch=gpu_arch,
    )


def _validate_inputs(A: torch.Tensor, B: torch.Tensor) -> None:
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError(
            "The AMD relu_bat_reduce kernel requires a ROCm build of PyTorch"
        )
    if not (A.is_cuda and B.is_cuda):
        raise ValueError("A and B must be HIP tensors")
    if A.device != B.device:
        raise ValueError("A and B must be on the same HIP device")
    if not (A.dtype == B.dtype == torch.float32):
        raise ValueError("A and B must be float32")
    if not (A.is_contiguous() and B.is_contiguous()):
        raise ValueError("A and B must be contiguous")
    if not (A.ndim == B.ndim == 2):
        raise ValueError("A and B must be 2D")

    _, D = A.shape
    _, DB = B.shape
    if DB != D:
        raise ValueError(f"Inner dimensions must match, got {D} and {DB}")


def relu_bat_reduce_fp32_mfma_available(
    device=None,
    D: int | None = None,
) -> bool:
    if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
        return False
    try:
        gpu_arch = hip_device_arch(device)
        if _base_gpu_arch(gpu_arch) not in _MFMA_FP32_WAVE64_ARCHES:
            return False
        if D is None or D == 0:
            return True
        if D < 0:
            return False

        properties = torch.cuda.get_device_properties(device)
        D_f = _round_up(D, 4)
        if _mfma_register_floor(M_TILES=1, D_f=D_f) > 256:
            return False
        minimum_smem = _mfma_dynamic_smem_bytes(
            BN=16,
            D_f=D_f,
            threads_per_block=64,
        )
        return minimum_smem <= _max_shared_memory_per_block(properties)
    except Exception:
        return False


def relu_bat_reduce_fp32_mfma(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(A, B)
    if BM <= 0 or BN <= 0 or M_TILES <= 0:
        raise ValueError(
            "BM, BN, and M_TILES must be positive, got "
            f"{BM}, {BN}, {M_TILES}"
        )
    if BN % 16 != 0:
        raise ValueError("BN must be divisible by the MFMA tile width 16")

    wave_m_rows = M_TILES * 16
    if BM % wave_m_rows != 0:
        raise ValueError("BM must be divisible by M_TILES * 16")
    waves_per_block = BM // wave_m_rows
    threads_per_block = waves_per_block * 64
    if threads_per_block > 1024:
        raise ValueError(
            "MFMA launch requires at most 1024 threads, got "
            f"{threads_per_block}"
        )

    M, D = A.shape
    N = B.shape[0]
    out_sum = torch.zeros((1,), device=A.device, dtype=torch.float64)
    out_sum2 = torch.zeros((1,), device=A.device, dtype=torch.float64)
    if M == 0 or N == 0 or D == 0:
        return out_sum.float(), out_sum2.float()

    gpu_arch = hip_device_arch(A.device)
    if _base_gpu_arch(gpu_arch) not in _MFMA_FP32_WAVE64_ARCHES:
        raise RuntimeError(
            "The FP32 MFMA reduction kernel requires "
            f"{_MFMA_FP32_WAVE64_ARCHES_DESCRIPTION}; "
            f"got {gpu_arch}"
        )

    properties = torch.cuda.get_device_properties(A.device)
    max_threads = int(
        getattr(properties, "max_threads_per_block", 1024) or 1024
    )
    if threads_per_block > max_threads:
        raise ValueError(
            f"MFMA launch requires {threads_per_block} threads, device limit is "
            f"{max_threads}"
        )

    D_f = _round_up(D, 4)
    register_floor = _mfma_register_floor(M_TILES=M_TILES, D_f=D_f)
    if register_floor > 256:
        raise ValueError(
            "MFMA launch has a minimum per-lane ArchVGPR footprint of "
            f"{register_floor}, exceeding the limit of 256"
        )
    smem_bytes = _mfma_dynamic_smem_bytes(
        BN=BN,
        D_f=D_f,
        threads_per_block=threads_per_block,
    )
    max_smem = _max_shared_memory_per_block(properties)
    if smem_bytes > max_smem:
        raise ValueError(
            f"MFMA launch requires {smem_bytes} bytes of LDS, device limit is "
            f"{max_smem}"
        )

    kernel = get_relu_bat_reduce_kernel_mfma(
        BM,
        BN,
        M_TILES,
        D_f,
        gpu_arch,
    )
    default_smem = int(
        getattr(properties, "shared_memory_per_block", 0) or 0
    )
    if smem_bytes > default_smem:
        kernel.set_shared_memory_config(smem_bytes)

    grid = (
        (M + BM - 1) // BM,
        (N + BN - 1) // BN,
        1,
    )
    kernel(
        grid=grid,
        block=(threads_per_block, 1, 1),
        shared_mem=smem_bytes,
        args=[A, B, out_sum, out_sum2, M, N, D],
    )
    return out_sum.float(), out_sum2.float()


__all__ = [
    "get_relu_bat_reduce_kernel_mfma",
    "relu_bat_reduce_fp32_mfma",
    "relu_bat_reduce_fp32_mfma_available",
]
