from __future__ import annotations

import torch

from .jit_kernel_tf32_sync import relu_bat_c_tf32_mma_sync_impl
from .jit_kernel_tf32_wgmma import relu_bat_c_tf32_wgmma_impl


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def validate_relu_bat_c_shapes(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> tuple[int, int]:
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        raise RuntimeError("expected rank-2 tensors")

    N, D = A.shape
    M, DB = B.shape
    NC, DC = C.shape
    if DB != D or NC != N or DC != D:
        raise RuntimeError("shape mismatch")

    if D < 1:
        raise RuntimeError("D must be >= 1")

    return M, D


@torch.library.custom_op(
    "sumac::relu_bat_c_tf32_mma_sync",
    mutates_args=(),
    device_types="cuda",
)
def relu_bat_c_tf32_mma_sync_op(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
    num_stages: int,
) -> torch.Tensor:
    return relu_bat_c_tf32_mma_sync_impl(
        A,
        B,
        C,
        BM=BM,
        BN=BN,
        M_TILES=M_TILES,
        num_stages=num_stages,
    )


@relu_bat_c_tf32_mma_sync_op.register_fake
def relu_bat_c_tf32_mma_sync_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
    num_stages: int,
) -> torch.Tensor:
    M, D = validate_relu_bat_c_shapes(A, B, C)
    if BN % 8 != 0:
        raise RuntimeError("BN must be divisible by MMA_N=8")
    if num_stages < 1 or num_stages > 3:
        raise RuntimeError("num_stages must be in [1, 3]")

    warp_m_rows = M_TILES * 16
    if BM % warp_m_rows != 0:
        raise RuntimeError("BM must be divisible by M_TILES * 16")

    D_pad = round_up(D, 8)
    return A.new_empty_strided((M, D), (D_pad, 1))


@torch.library.custom_op(
    "sumac::relu_bat_c_tf32_wgmma",
    mutates_args=(),
    device_types="cuda",
)
def relu_bat_c_tf32_wgmma_op(
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
    return relu_bat_c_tf32_wgmma_impl(
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


@relu_bat_c_tf32_wgmma_op.register_fake
def relu_bat_c_tf32_wgmma_fake(
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
    M, D = validate_relu_bat_c_shapes(A, B, C)
    if WGMMA_S_N not in (16, 32, 64, 128):
        raise RuntimeError("WGMMA_S_N must be one of 16, 32, 64, or 128")
    if WGMMA_Y_N not in (16, 32, 64, 128):
        raise RuntimeError("WGMMA_Y_N must be one of 16, 32, 64, or 128")
    if BN % WGMMA_S_N != 0:
        raise RuntimeError("BN must be divisible by WGMMA_S_N")
    if num_stages not in (1, 2, 3):
        raise RuntimeError("num_stages must be 1, 2, or 3")
    mode = str(wgmma_mode).upper()
    if mode not in ("RS", "SS"):
        raise RuntimeError("wgmma_mode must be 'RS' or 'SS'")
    if BM % 64 != 0:
        raise RuntimeError("BM must be divisible by 64")

    compute_warpgroups_per_block = BM // 64
    if compute_warpgroups_per_block < 1:
        raise RuntimeError("BM must cover at least one warpgroup")
    if (compute_warpgroups_per_block + 1) * 128 > 1024:
        raise RuntimeError("A CTA cannot contain more than 1024 threads")

    D_y_pad = round_up(D, WGMMA_Y_N)
    return A.new_empty_strided((M, D), (D_y_pad, 1))
