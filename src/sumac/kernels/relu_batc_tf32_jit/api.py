from __future__ import annotations

import torch

from .custom_op import relu_bat_c_tf32_mma_sync_op, relu_bat_c_tf32_wgmma_op


def normalize_wgmma_mode(wgmma_mode: str) -> str:
    mode = str(wgmma_mode).upper()
    if mode not in ("RS", "SS"):
        raise ValueError("wgmma_mode must be 'RS' or 'SS'")
    return mode


def relu_bat_c_tf32_mma_sync(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    BM: int,
    BN: int,
    M_TILES: int,
    num_stages: int = 2,
) -> torch.Tensor:
    return relu_bat_c_tf32_mma_sync_op(
        A,
        B,
        C,
        BM,
        BN,
        M_TILES,
        num_stages,
    )


def relu_bat_c_tf32_wgmma(
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
    if WGMMA_S_N is None:
        WGMMA_S_N = WGMMA_N
    if WGMMA_Y_N is None:
        WGMMA_Y_N = WGMMA_N

    return relu_bat_c_tf32_wgmma_op(
        A,
        B,
        C,
        BM,
        BN,
        WGMMA_S_N,
        WGMMA_Y_N,
        num_stages,
        normalize_wgmma_mode(wgmma_mode),
    )
