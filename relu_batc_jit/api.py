from __future__ import annotations

import torch
from .custom_op import relu_bat_c_fused_op


def relu_bat_c_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    BM: int,
    BK: int,
    MS: int,
) -> torch.Tensor:
    return relu_bat_c_fused_op(A, B, C, BM, BK, MS)
