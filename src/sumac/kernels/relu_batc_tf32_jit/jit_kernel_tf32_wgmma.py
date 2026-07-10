from __future__ import annotations

import ctypes
import sys
from functools import lru_cache
from pathlib import Path

import torch
from ..compile_utils import compile_cuda_kernel
from torch.utils.cpp_extension import include_paths

DEBUG = False
DEFAULT_DYNAMIC_SMEM_LIMIT_BYTES = 48 * 1024
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

_KERNEL_MARKER = "// KERNEL_START"
_KERNEL_PATH_WGMMA_TF32 = Path(__file__).with_name(
    "kernel_wgmma_tf32_tma.cu"
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


def _make_header_wgmma_tf32(
    BM: int,
    BN: int,
    D_f: int,
    WGMMA_S_N: int,
    WGMMA_Y_N: int,
    num_stages: int,
    kernel_name: str,
    pack_kernel_name: str,
    wgmma_mode: str = "RS",
    pack_only: bool = False,
    D_y_f: int | None = None,
) -> str:
    wgmma_mode = _normalize_wgmma_mode(wgmma_mode)
    D_k_f = D_f
    if D_y_f is None:
        D_y_f = D_f
    return f"""
#define BM {BM}
#define BN {BN}
#define D_f {D_y_f}
#define D_K_F {D_k_f}
#define D_Y_F {D_y_f}
#define WGMMA_S_N_SHAPE {WGMMA_S_N}
#define WGMMA_Y_N_SHAPE {WGMMA_Y_N}
#define SMEM_COPY_STAGES {num_stages}
#define WGMMA_TF32_KERNEL_NAME {kernel_name}
#define WGMMA_TF32_PACK_KERNEL_NAME {pack_kernel_name}
#define WGMMA_TF32_PACK_ONLY {1 if pack_only else 0}
#define WGMMA_FIRST_MMA_SS {1 if wgmma_mode == "SS" else 0}
"""


def _dynamic_smem_bytes(
    *,
    BM: int,
    BN: int,
    D: int | None = None,
    D_k: int | None = None,
    D_y: int | None = None,
    num_stages: int,
    wgmma_mode: str,
) -> int:
    wgmma_mode = _normalize_wgmma_mode(wgmma_mode)
    if D_k is None:
        D_k = D
    if D_y is None:
        D_y = D
    if D_k is None or D_y is None:
        raise ValueError("Either D or both D_k and D_y must be provided")

    elems = num_stages * BN * (D_k + D_y)
    if wgmma_mode == "SS":
        elems += BM * D_k
    return elems * 4 + 127


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _packed_panel_elems(*, BN: int, D: int) -> int:
    return BN * D


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


def _require_sm90a(device: torch.device) -> None:
    major, minor = torch.cuda.get_device_capability(device)
    if major != 9:
        raise ValueError(
            "TF32 WGMMA path requires an SM90-class GPU such as H100/H200; "
            f"got compute capability {major}.{minor}"
        )


def _kernel_name_wgmma_tf32() -> str:
    return "relu_bat_c_tf32_wgmma"


def _pack_kernel_name_wgmma_tf32() -> str:
    return "relu_bat_c_tf32_wgmma_pack"


def _check_wgmma_n_shape(name: str, value: int) -> None:
    if value not in (16, 32, 64, 128):
        raise ValueError(f"{name} must be one of 16, 32, 64, or 128")


def _check_num_stages(num_stages: int) -> None:
    if num_stages not in (1, 2, 3):
        raise ValueError("num_stages must be 1, 2, or 3")


def _normalize_wgmma_mode(wgmma_mode: str) -> str:
    mode = str(wgmma_mode).upper()
    if mode not in ("RS", "SS"):
        raise ValueError("wgmma_mode must be 'RS' or 'SS'")
    return mode


def _compile_or_print_full_error(
    kernel_source: str,
    *,
    kernel_name: str,
    header_code: str,
    debug_context: str,
):
    try:
        return compile_cuda_kernel(
            kernel_source,
            kernel_name=kernel_name,
            compute_capability="90a",
            header_code=header_code,
            cuda_include_dirs=include_paths("cuda"),
            nvcc_options=["-lineinfo"],
        )
    except Exception as exc:
        print(
            "\n=== WGMMA NVRTC compile failure ===",
            file=sys.stderr,
            flush=True,
        )
        print(debug_context, file=sys.stderr, flush=True)
        print(str(exc), file=sys.stderr, flush=True)
        print(
            "=== end WGMMA NVRTC compile failure ===\n",
            file=sys.stderr,
            flush=True,
        )
        raise


@lru_cache(maxsize=None)
def get_relu_bat_c_kernel_wgmma_tf32(
    BM: int,
    BN: int,
    D_k_f: int,
    D_y_f: int,
    WGMMA_S_N: int,
    WGMMA_Y_N: int,
    num_stages: int,
    wgmma_mode: str,
):
    _check_wgmma_n_shape("WGMMA_S_N", WGMMA_S_N)
    _check_wgmma_n_shape("WGMMA_Y_N", WGMMA_Y_N)
    _check_num_stages(num_stages)
    wgmma_mode = _normalize_wgmma_mode(wgmma_mode)

    kernel_name = _kernel_name_wgmma_tf32()
    pack_kernel_name = _pack_kernel_name_wgmma_tf32()
    header_code, kernel_source, kernel_source_start_line = _split_kernel_file(
        _KERNEL_PATH_WGMMA_TF32
    )

    header_code = (
        _make_header_wgmma_tf32(
            BM,
            BN,
            D_k_f,
            WGMMA_S_N,
            WGMMA_Y_N,
            num_stages,
            kernel_name,
            pack_kernel_name,
            wgmma_mode,
            False,
            D_y_f=D_y_f,
        )
        + "\n"
        + f'#line 1 "{_KERNEL_PATH_WGMMA_TF32}"\n'
        + header_code
        + "\n"
        + f'#line {kernel_source_start_line} '
          f'"{_KERNEL_PATH_WGMMA_TF32}"\n'
    )

    if DEBUG:
        print(
            "compiling relu_bat_c TF32 WGMMA direct kernel "
            f"with BM={BM}, BN={BN}, D_k_f={D_k_f}, D_y_f={D_y_f}, "
            f"WGMMA_S_N={WGMMA_S_N}, WGMMA_Y_N={WGMMA_Y_N}, "
            f"num_stages={num_stages}, wgmma_mode={wgmma_mode}, "
            "prepacked A/C, producer warpgroup"
        )

    return _compile_or_print_full_error(
        kernel_source,
        kernel_name=kernel_name,
        header_code=header_code,
        debug_context=(
            "compute kernel "
            f"BM={BM} BN={BN} D_k_f={D_k_f} D_y_f={D_y_f} "
            f"WGMMA_S_N={WGMMA_S_N} WGMMA_Y_N={WGMMA_Y_N} "
            f"num_stages={num_stages} "
            f"wgmma_mode={wgmma_mode} "
            f"kernel_name={kernel_name}"
        ),
    )


@lru_cache(maxsize=None)
def get_relu_bat_c_pack_kernel_wgmma_tf32(
    BM: int,
    BN: int,
    D_k_f: int,
    D_y_f: int,
    WGMMA_S_N: int,
    WGMMA_Y_N: int,
):
    _check_wgmma_n_shape("WGMMA_S_N", WGMMA_S_N)
    _check_wgmma_n_shape("WGMMA_Y_N", WGMMA_Y_N)

    kernel_name = _kernel_name_wgmma_tf32()
    pack_kernel_name = _pack_kernel_name_wgmma_tf32()
    header_code, kernel_source, kernel_source_start_line = _split_kernel_file(
        _KERNEL_PATH_WGMMA_TF32
    )

    header_code = (
        _make_header_wgmma_tf32(
            BM,
            BN,
            D_k_f,
            WGMMA_S_N,
            WGMMA_Y_N,
            2,
            kernel_name,
            pack_kernel_name,
            "RS",
            True,
            D_y_f=D_y_f,
        )
        + "\n"
        + f'#line 1 "{_KERNEL_PATH_WGMMA_TF32}"\n'
        + header_code
        + "\n"
        + f'#line {kernel_source_start_line} '
          f'"{_KERNEL_PATH_WGMMA_TF32}"\n'
    )

    if DEBUG:
        print(
            "compiling relu_bat_c TF32 WGMMA A/C pack kernel "
            f"with BN={BN}, D_k_f={D_k_f}, D_y_f={D_y_f}, "
            f"WGMMA_S_N={WGMMA_S_N}, WGMMA_Y_N={WGMMA_Y_N}"
        )

    return _compile_or_print_full_error(
        kernel_source,
        kernel_name=pack_kernel_name,
        header_code=header_code,
        debug_context=(
            "pack kernel "
            f"BM={BM} BN={BN} D_k_f={D_k_f} D_y_f={D_y_f} "
            f"WGMMA_S_N={WGMMA_S_N} WGMMA_Y_N={WGMMA_Y_N} "
            f"kernel_name={pack_kernel_name}"
        ),
    )


def relu_bat_c_tf32_wgmma_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    BM: int,
    BN: int,
    WGMMA_N: int = 16,
    WGMMA_S_N: int | None = None,
    WGMMA_Y_N: int | None = None,
    num_stages: int = 2,
    wgmma_mode: str = "RS",
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

    _require_sm90a(A.device)

    N, D = A.shape
    M, DB = B.shape
    NC, DC = C.shape

    if DB != D:
        raise ValueError(f"B has D={DB}, expected {D}")
    if NC != N or DC != D:
        raise ValueError(f"C must have shape [{N}, {D}], got {tuple(C.shape)}")

    if WGMMA_S_N is None:
        WGMMA_S_N = WGMMA_N
    if WGMMA_Y_N is None:
        WGMMA_Y_N = WGMMA_N

    _check_wgmma_n_shape("WGMMA_S_N", WGMMA_S_N)
    _check_wgmma_n_shape("WGMMA_Y_N", WGMMA_Y_N)
    _check_num_stages(num_stages)
    wgmma_mode = _normalize_wgmma_mode(wgmma_mode)

    if D < 1:
        raise ValueError("D must be >= 1")
    if BN % WGMMA_S_N != 0:
        raise ValueError(
            "TF32 WGMMA S path requires "
            f"BN % WGMMA_S_N == 0, got BN={BN}, WGMMA_S_N={WGMMA_S_N}"
        )

    if BM % 64 != 0:
        raise ValueError("BM must be divisible by 64")

    compute_warpgroups_per_block = BM // 64
    if compute_warpgroups_per_block < 1:
        raise ValueError("BM must cover at least one warpgroup")
    threads_per_block = (compute_warpgroups_per_block + 1) * 128
    if threads_per_block > 1024:
        raise ValueError(
            "A CTA cannot contain more than 1024 threads"
        )
    D_k_pad = _round_up(D, 8)
    D_y_pad = _round_up(D, WGMMA_Y_N)
    Y = torch.empty((M, D_y_pad), device=A.device, dtype=torch.float32)
    num_panels = (N + BN - 1) // BN

    A_packed_shape = (num_panels, _packed_panel_elems(BN=BN, D=D_k_pad))
    C_packed_shape = (num_panels, _packed_panel_elems(BN=BN, D=D_y_pad))
    A_packed = torch.empty(A_packed_shape, device=A.device, dtype=torch.int32)
    C_packed = torch.empty(C_packed_shape, device=A.device, dtype=torch.int32)

    if num_panels > 0:
        pack_kernel = get_relu_bat_c_pack_kernel_wgmma_tf32(
            BM,
            BN,
            D_k_pad,
            D_y_pad,
            WGMMA_S_N,
            WGMMA_Y_N,
        )
        pack_threads = 256
        panel_elems = max(
            _packed_panel_elems(BN=BN, D=D_k_pad),
            _packed_panel_elems(BN=BN, D=D_y_pad),
        )
        pack_grid = (
            (panel_elems + pack_threads - 1) // pack_threads,
            num_panels,
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

    kernel = get_relu_bat_c_kernel_wgmma_tf32(
        BM,
        BN,
        D_k_pad,
        D_y_pad,
        WGMMA_S_N,
        WGMMA_Y_N,
        num_stages,
        wgmma_mode,
    )
    smem_bytes = _dynamic_smem_bytes(
        BM=BM,
        BN=BN,
        D_k=D_k_pad,
        D_y=D_y_pad,
        num_stages=num_stages,
        wgmma_mode=wgmma_mode,
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
