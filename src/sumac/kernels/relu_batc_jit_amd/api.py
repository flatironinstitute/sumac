from __future__ import annotations

import torch

from .custom_op import relu_bat_c_fp32_mfma_amd_op
from .jit_kernel import relu_bat_c_fp32_mfma_available


def relu_bat_c_fp32_mfma(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    BM: int,
    BN: int,
    M_TILES: int,
    num_stages: int = 2,
) -> torch.Tensor:
    return relu_bat_c_fp32_mfma_amd_op(
        A, B, C, BM, BN, M_TILES, num_stages
    )


__all__ = [
    "relu_bat_c_fp32_mfma",
    "relu_bat_c_fp32_mfma_available",
]
