from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch
from ..compile_utils import compile_cuda_kernel
from torch.utils.cpp_extension import include_paths

DEBUG = False
DEFAULT_DYNAMIC_SMEM_LIMIT_BYTES = 48 * 1024
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

_KERNEL_MARKER = "// KERNEL_START"
_KERNEL_PATH_MMA_SYNC_TF32 = Path(__file__).with_name(
    "kernel_mma_sync_tf32.cu"
)


def _split_kernel_file(path: Path) -> tuple[str, str, int]:
    src = path.read_text()
    if _KERNEL_MARKER not in src:
        raise RuntimeError(f"Missing marker {_KERNEL_MARKER!r} in {path}")

    marker_offset = src.index(_KERNEL_MARKER)
    marker_line = src[:marker_offset].count("\n") + 1
    header_code, kernel_source = src.split(_KERNEL_MARKER, 1)
    kernel_source = kernel_source.lstrip()

    # The marker is followed by a blank line in this file.
    kernel_source_start_line = marker_line + 2
    return header_code, kernel_source, kernel_source_start_line


def _make_header_mma_sync_tf32(
    BM: int,
    BN: int,
    D_f: int,
    M_TILES: int,
    num_stages: int,
    padded_d: bool,
    kernel_name: str,
    pack_kernel_name: str,
) -> str:
    return f"""
#define BM {BM}
#define BN {BN}
#define D_f {D_f}
#define M_TILES {M_TILES}
#define MMA_SYNC_TF32_STAGES {num_stages}
#define MMA_SYNC_TF32_PADDED_D {1 if padded_d else 0}
#define MMA_SYNC_TF32_KERNEL_NAME {kernel_name}
#define MMA_SYNC_TF32_PACK_KERNEL_NAME {pack_kernel_name}
"""


def _dynamic_smem_bytes(
    *,
    BM: int,
    BN: int,
    D: int,
    M_TILES: int,
    num_stages: int,
) -> int:
    operand_pipeline_bytes = 2 * num_stages * BN * D * 4
    compute_warps = BM // (M_TILES * 16)
    b_stage_bytes = compute_warps * 512
    return max(operand_pipeline_bytes, b_stage_bytes) + 127


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _packed_tile_elems(*, BN: int, D: int) -> int:
    return BN * D


def _packed_tile_pairs(*, BN: int, D: int) -> int:
    return _packed_tile_elems(BN=BN, D=D) // 2


def _kernel_name_mma_sync_tf32(D_f: int) -> str:
    return f"relu_bat_c_tf32_mma_sync_d{D_f}"


def _pack_kernel_name_mma_sync_tf32(D_f: int) -> str:
    return f"relu_bat_c_tf32_mma_sync_pack_d{D_f}"


def _maybe_opt_in_dynamic_smem(kernel, smem_bytes: int) -> None:
    if smem_bytes < DEFAULT_DYNAMIC_SMEM_LIMIT_BYTES:
        return

    set_shared_memory_config = getattr(kernel, "set_shared_memory_config", None)
    if set_shared_memory_config is not None:
        set_shared_memory_config(smem_bytes)
        return

    from torch.cuda._utils import _check_cuda, _get_cuda_library

    libcuda = _get_cuda_library()
    libcuda.cuFuncSetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    libcuda.cuFuncSetAttribute.restype = ctypes.c_int

    _check_cuda(
        libcuda.cuFuncSetAttribute(
            kernel.func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes,
        )
    )


@lru_cache(maxsize=None)
def get_relu_bat_c_kernel_mma_sync_tf32(
    BM: int,
    BN: int,
    D_f: int,
    M_TILES: int,
    num_stages: int,
    padded_d: bool,
    *,
    device: int,
):
    kernel_name = _kernel_name_mma_sync_tf32(D_f)
    pack_kernel_name = _pack_kernel_name_mma_sync_tf32(D_f)
    header_code, kernel_source, kernel_source_start_line = _split_kernel_file(
        _KERNEL_PATH_MMA_SYNC_TF32
    )

    header_code = (
        f'#line 1 "{_KERNEL_PATH_MMA_SYNC_TF32}"\n'
        + _make_header_mma_sync_tf32(
            BM,
            BN,
            D_f,
            M_TILES,
            num_stages,
            padded_d,
            kernel_name,
            pack_kernel_name,
        )
        + "\n"
        + header_code
        + "\n"
        + f'#line {kernel_source_start_line} '
          f'"{_KERNEL_PATH_MMA_SYNC_TF32}"\n'
    )

    if DEBUG:
        print(
            "compiling relu_bat_c TF32 cp.async kernel "
            f"with BM={BM}, BN={BN}, D_f={D_f}, "
            f"M_TILES={M_TILES}, num_stages={num_stages}, "
            f"padded_d={padded_d}"
        )

    return compile_cuda_kernel(
        kernel_source,
        kernel_name=kernel_name,
        header_code=header_code,
        device=device,
        cuda_include_dirs=include_paths("cuda"),
        nvcc_options=["-lineinfo"],
    )


@lru_cache(maxsize=None)
def get_relu_bat_c_pack_kernel_mma_sync_tf32(
    BM: int,
    BN: int,
    D_f: int,
    M_TILES: int,
    *,
    device: int,
):
    kernel_name = _kernel_name_mma_sync_tf32(D_f)
    pack_kernel_name = _pack_kernel_name_mma_sync_tf32(D_f)
    header_code, kernel_source, kernel_source_start_line = _split_kernel_file(
        _KERNEL_PATH_MMA_SYNC_TF32
    )

    header_code = (
        f'#line 1 "{_KERNEL_PATH_MMA_SYNC_TF32}"\n'
        + _make_header_mma_sync_tf32(
            BM,
            BN,
            D_f,
            M_TILES,
            2,
            False,
            kernel_name,
            pack_kernel_name,
        )
        + "\n"
        + header_code
        + "\n"
        + f'#line {kernel_source_start_line} '
          f'"{_KERNEL_PATH_MMA_SYNC_TF32}"\n'
    )

    if DEBUG:
        print(
            "compiling relu_bat_c TF32 mma.sync A/C pack kernel "
            f"with BM={BM}, BN={BN}, D_f={D_f}, M_TILES={M_TILES}"
        )

    return compile_cuda_kernel(
        kernel_source,
        kernel_name=pack_kernel_name,
        header_code=header_code,
        device=device,
        cuda_include_dirs=include_paths("cuda"),
        nvcc_options=["-lineinfo"],
    )


def relu_bat_c_tf32_mma_sync_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    BM: int,
    BN: int,
    M_TILES: int,
    num_stages: int = 2,
) -> torch.Tensor:
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, and C must be CUDA tensors")
    if not (A.device == B.device == C.device):
        raise ValueError("A, B, and C must be on the same CUDA device")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("A, B, and C must be float32 tensors")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("A, B, and C must be contiguous")
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        raise ValueError("A, B, and C must be rank-2 tensors")

    N, D = A.shape
    M, DB = B.shape
    NC, DC = C.shape

    if DB != D:
        raise ValueError(f"B has D={DB}, expected {D}")
    if NC != N or DC != D:
        raise ValueError(f"C must have shape [{N}, {D}], got {tuple(C.shape)}")
    if D < 1:
        raise ValueError("D must be >= 1")
    if BN % 8 != 0:
        raise ValueError("BN must be divisible by MMA_N=8")
    if num_stages < 1 or num_stages > 3:
        raise ValueError("num_stages must be in [1, 3]")

    warp_m_rows = M_TILES * 16
    if BM % warp_m_rows != 0:
        raise ValueError("BM must be divisible by M_TILES * 16")

    compute_warps_per_block = BM // warp_m_rows
    threads_per_block = compute_warps_per_block * 32

    num_tiles = (N + BN - 1) // BN
    D_f = _round_up(D, 8)
    Y = torch.empty((M, D_f), device=A.device, dtype=torch.float32)

    packed_shape = (num_tiles, _packed_tile_elems(BN=BN, D=D_f))
    A_packed = torch.empty(packed_shape, device=A.device, dtype=torch.int32)
    C_packed = torch.empty(packed_shape, device=A.device, dtype=torch.int32)
    device = int(A.device.index)

    if num_tiles > 0:
        pack_kernel = get_relu_bat_c_pack_kernel_mma_sync_tf32(
            BM,
            BN,
            D_f,
            M_TILES,
            device=device,
        )
        pack_threads = 256
        tile_pairs = _packed_tile_pairs(BN=BN, D=D_f)
        pack_grid = (
            (tile_pairs + pack_threads - 1) // pack_threads,
            num_tiles,
            1,
        )
        pack_kernel(
            grid=pack_grid,
            block=(pack_threads, 1, 1),
            args=[
                A,
                C,
                A_packed,
                C_packed,
                int(N),
                int(D),
            ],
        )

    kernel = get_relu_bat_c_kernel_mma_sync_tf32(
        BM,
        BN,
        D_f,
        M_TILES,
        num_stages,
        D != D_f,
        device=device,
    )
    smem_bytes = _dynamic_smem_bytes(
        BM=BM,
        BN=BN,
        D=D_f,
        M_TILES=M_TILES,
        num_stages=num_stages,
    )
    _maybe_opt_in_dynamic_smem(kernel, smem_bytes)

    grid = ((M + BM - 1) // BM, 1, 1)
    block = (threads_per_block, 1, 1)

    kernel(
        grid=grid,
        block=block,
        shared_mem=smem_bytes,
        args=[
            A_packed,
            C_packed,
            B,
            Y,
            int(N),
            int(M),
            int(D),
        ],
    )

    return Y[:, :D]
