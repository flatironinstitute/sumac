from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import torch

from torch.utils.cpp_extension import include_paths
DEBUG = False

_KERNEL_PATH = Path(__file__).with_name("kernel.cu")
_KERNEL_PATH_MIXED = Path(__file__).with_name("kernel_mixed.cu")
_KERNEL_MARKER = "// KERNEL_START"

KERNEL_SOURCE_START_LINE = 10 #this is only used for source-line->instruction correlation in nsight-compute
KERNEL_SOURCE_START_LINE_MIXED = 85

def _make_header(BK, V, MS):
    return f"""
#define BK {BK}
#define V  {V}
#define MS {MS}
"""
def _make_header_mixed(BK, V, R, MS):
    return f"""
#define BK {BK}
#define V  {V}
#define R  {R}
#define MS {MS}
"""

def _split_kernel_file(path: Path) -> tuple[str, str]:
    src = path.read_text()
    if _KERNEL_MARKER not in src:
        raise RuntimeError(f"Missing marker {_KERNEL_MARKER!r} in {path}")
    header_code, kernel_source = src.split(_KERNEL_MARKER, 1)
    kernel_source = kernel_source.lstrip()
    return header_code, kernel_source


@lru_cache(maxsize=None)
def get_relu_bat_reduce_kernel_float4(BK, V, MS):
    header_code, kernel_source = _split_kernel_file(_KERNEL_PATH)

    header_code = (
        f'#line 1 "{_KERNEL_PATH}"\n'
        + header_code
        + "\n"
        + _make_header(BK, V, MS)
        + "\n"
        + f'#line {KERNEL_SOURCE_START_LINE} "{_KERNEL_PATH}"\n'
    )
    if DEBUG:
        print(f"compiling relu_bat_reduce with BK={BK}, V={V}, MS={MS}")

    return torch.cuda._compile_kernel(
        kernel_source,
        kernel_name="relu_bat_reduce_kernel_float4_sync",
        header_code=header_code,
        cuda_include_dirs=include_paths("cuda"),
        nvcc_options = ["-lineinfo"]
    )

@lru_cache(maxsize=None)
def get_relu_bat_reduce_kernel_mixed(BK, V, R, MS):
    header_code, kernel_source = _split_kernel_file(_KERNEL_PATH_MIXED)
    header_code = (
        f'#line 1 "{_KERNEL_PATH_MIXED}"\n'
        + _make_header_mixed(BK, V, R, MS)
        + "\n"
        + header_code
        + "\n"
        + f'#line {KERNEL_SOURCE_START_LINE_MIXED} "{_KERNEL_PATH_MIXED}"\n'
    )
    if DEBUG:
        print(f"compiling relu_bat_reduce (mixed variant) with BK={BK}, V={V}, R={R}, MS={MS}")

    return torch.cuda._compile_kernel(
        kernel_source=kernel_source,
        kernel_name="relu_bat_reduce_kernel_mixed_sync",
        header_code=header_code,
        cuda_include_dirs=include_paths("cuda"),
        nvcc_options = ["-lineinfo"]
    )


def relu_bat_reduce_fused(
        A: torch.Tensor,
        B: torch.Tensor,
        BM: int,
        BK: int,
        MS: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.cuda.nvtx.range_push("Prep kernel")
    if not (A.is_cuda and B.is_cuda):
        raise ValueError("A and B must be CUDA tensors")
    if not (A.dtype == B.dtype == torch.float32):
        raise ValueError("A and B must be float32")
    if not (A.is_contiguous() and B.is_contiguous()):
        raise ValueError("A and B must be contiguous")
    if not (A.ndim == B.ndim == 2):
        raise ValueError("A and B must be 2D")

    M, D = A.shape
    N, DB = B.shape
    if D != DB:
        raise ValueError(f"Inner dimensions must match, got {D} and {DB}")
    V = D // 4
    R = D % 4

    dynamic_shared_mem_bytes = 2 * BM * 4
    out_sum = torch.zeros(1, device=A.device, dtype=torch.float64)
    out_sum2 = torch.zeros(1, device=A.device, dtype=torch.float64)
    kernel = get_relu_bat_reduce_kernel_float4(BK, V, MS) if R==0 else get_relu_bat_reduce_kernel_mixed(BK, V, R, MS)
    torch.cuda.nvtx.range_pop()
    
    grid = (
        (M + BM * MS - 1) // (BM * MS),
        (N + BK - 1) // BK,
        1,
    )
    block = (BM, 1, 1)
    
    with torch.cuda.nvtx.range("launch kernel"):
        kernel(grid, block, (A, B, out_sum, out_sum2, M, N, D), shared_mem=dynamic_shared_mem_bytes)

    return out_sum.float(), out_sum2.float()