from __future__ import annotations

import torch
from .custom_op import relu_bat_reduce_fused_op


def relu_bat_reduce_fused(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BK: int,
    MS: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return relu_bat_reduce_fused_op(A, B, BM, BK, MS)
