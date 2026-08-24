from __future__ import annotations

import torch

from .custom_op import relu_bat_reduce_fp32_mfma_amd_op
from .jit_kernel import relu_bat_reduce_fp32_mfma_available


def relu_bat_reduce_fp32_mfma(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    BM: int,
    BN: int,
    M_TILES: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return relu_bat_reduce_fp32_mfma_amd_op(
        A,
        B,
        BM,
        BN,
        M_TILES,
    )


__all__ = [
    "relu_bat_reduce_fp32_mfma",
    "relu_bat_reduce_fp32_mfma_available",
]
