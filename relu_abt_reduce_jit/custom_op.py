from __future__ import annotations
import torch

from .jit_kernel import relu_abt_reduce_fused as _relu_abt_reduce_fused_impl


@torch.library.custom_op("sumac::relu_abt_reduce_fused", mutates_args=(), device_types="cuda")
def relu_abt_reduce_fused_op(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BK: int,
    MS: int,
) -> tuple[torch.Tensor,torch.Tensor]:
    return _relu_abt_reduce_fused_impl(A, B, BM, BK, MS)

@relu_abt_reduce_fused_op.register_fake
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BK: int,
    MS: int,
) -> tuple[torch.Tensor,torch.Tensor]:
    if A.dim() != 2 or B.dim() != 2:
        raise RuntimeError("expected 2D tensors")
    M, D = A.shape
    N, DB = B.shape
    if DB != D:
        raise RuntimeError("shape mismatch")
    return A.new_empty((1)), A.new_empty((1))