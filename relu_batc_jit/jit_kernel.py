from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import torch

from torch.utils.cpp_extension import include_paths
DEBUG = False


_KERNEL_PATH = Path(__file__).with_name("kernel.cu")
_KERNEL_PATH_MIXED = Path(__file__).with_name("kernel_mixed.cu")
_KERNEL_MARKER = "// KERNEL_START"

KERNEL_SOURCE_START_LINE = 19
KERNEL_SOURCE_START_LINE_MIXED = 127 #Using these markers to ~mostly~ fix the source-line->instruction correlation in nsight-compute

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
def get_relu_bat_c_kernel_float4(BK, V, MS):
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
        print(f"compiling relu_bat_c (float4 variant) with BK={BK}, V={V}, MS={MS}")
    
    return torch.cuda._compile_kernel(
        kernel_source,
        kernel_name="relu_bat_c_fused_kernel_float4_sync",
        header_code=header_code,
        cuda_include_dirs=include_paths("cuda"),
        nvcc_options = ["-lineinfo"]
    )

@lru_cache(maxsize=None)
def get_relu_bat_c_kernel_mixed(BK, V, R, MS):
    header_code, kernel_source = _split_kernel_file(_KERNEL_PATH_MIXED)
    header_code = (
        f'#line 1 "{_KERNEL_PATH_MIXED}"\n'
        + _make_header_mixed(BK, V, R, MS) #This ordering matters since header_code uses them
        + "\n"
        + header_code
        + "\n"
        + f'#line {KERNEL_SOURCE_START_LINE_MIXED} "{_KERNEL_PATH_MIXED}"\n'
    )
    if DEBUG:
        print(f"compiling relu_bat_c (mixed variant) with BK={BK}, V={V}, R={R}, MS={MS}")

    return torch.cuda._compile_kernel(
        kernel_source,
        kernel_name="relu_bat_c_fused_kernel_mixed_sync",
        header_code= header_code,
        cuda_include_dirs=include_paths("cuda"),
        nvcc_options = ["-lineinfo"]
    )

def relu_bat_c_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BK: int,
    MS: int,
    BM: int,
) -> torch.Tensor:
    torch.cuda.nvtx.range_push("Prep kernel")
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A, B, and C must be CUDA tensors")
    if not (A.dtype == B.dtype == C.dtype == torch.float32):
        raise ValueError("A, B, and C must be float32")
    if not (A.is_contiguous() and B.is_contiguous() and C.is_contiguous()):
        raise ValueError("A, B, and C must be contiguous")
    if not (A.ndim == B.ndim == C.ndim == 2):
        raise ValueError("A, B, and C must be 2D")

    N, D = A.shape
    M, DB = B.shape
    NC, DC = C.shape
    V = D//4
    R = D % 4
    if DB != D:
        raise ValueError(f"B.shape[1] must equal A.shape[1], got {DB} vs {D}")
    if (NC, DC) != (N, D):
        raise ValueError(f"C must have shape {(N, D)}, got {tuple(C.shape)}")
   


    Y = torch.empty((M, D), device=A.device, dtype=A.dtype)
    kernel = get_relu_bat_c_kernel_float4(BK, V, MS) if R==0 else get_relu_bat_c_kernel_mixed(BK, V, R, MS)
    torch.cuda.nvtx.range_pop()
    
    grid = ((M + BM * MS - 1) // (BM * MS), 1, 1)  
    block = (BM, 1, 1)
    
    with torch.cuda.nvtx.range("launch kernel"):
        kernel(grid, block, (A, B, C, Y, N, M, D))

    return Y