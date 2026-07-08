from __future__ import annotations
import torch

from .jit_kernel import relu_bat_c_fused as _relu_bat_c_fused_impl


@torch.library.custom_op("sumac::relu_bat_c_fused", mutates_args=(), device_types="cuda")
def relu_bat_c_fused_op(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BK: int,
    MS: int,
) -> torch.Tensor:
    return _relu_bat_c_fused_impl(
        A,
        B,
        C,
        BM=BM,
        BK=BK,
        MS=MS,
    )


@relu_bat_c_fused_op.register_fake
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BK: int,
    MS: int,
) -> torch.Tensor:
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        raise RuntimeError("expected 2D tensors")
    N, D = A.shape
    M, DB = B.shape
    NC, DC = C.shape
    if DB != D or NC != N or DC != D:
        raise RuntimeError("shape mismatch")
    return A.new_empty((M, D))
