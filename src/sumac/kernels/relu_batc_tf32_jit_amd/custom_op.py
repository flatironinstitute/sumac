from __future__ import annotations

import torch

from .jit_kernel import (
    relu_bat_c_tf32_mfma as _relu_bat_c_tf32_mfma_impl,
)


@torch.library.custom_op(
    "sumac::relu_bat_c_tf32_mfma_amd",
    mutates_args=(),
    device_types="cuda",
)
def relu_bat_c_tf32_mfma_amd_op(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
) -> torch.Tensor:
    return _relu_bat_c_tf32_mfma_impl(
        A,
        B,
        C,
        BM=BM,
        BN=BN,
        M_TILES=M_TILES,
    )


@relu_bat_c_tf32_mfma_amd_op.register_fake
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
) -> torch.Tensor:
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        raise RuntimeError("expected 2D tensors")
    N, D = A.shape
    M, DB = B.shape
    NC, DC = C.shape
    if DB != D or NC != N or DC != D:
        raise RuntimeError("shape mismatch")
    if BM <= 0 or BN <= 0 or M_TILES <= 0:
        raise RuntimeError("BM, BN, and M_TILES must be positive")
    if BN % 16 != 0:
        raise RuntimeError("BN must be divisible by 16")
    if BM % (M_TILES * 16) != 0:
        raise RuntimeError("BM must be divisible by M_TILES * 16")
    return A.new_empty((M, D))


__all__ = ["relu_bat_c_tf32_mfma_amd_op"]
