from __future__ import annotations

import torch

from .jit_kernel import (
    relu_bat_reduce_fp32_mfma as _relu_bat_reduce_fp32_mfma_impl,
)


@torch.library.custom_op(
    "sumac::relu_bat_reduce_fp32_mfma_amd",
    mutates_args=(),
    device_types="cuda",
)
def relu_bat_reduce_fp32_mfma_amd_op(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _relu_bat_reduce_fp32_mfma_impl(
        A,
        B,
        BM=BM,
        BN=BN,
        M_TILES=M_TILES,
    )


@relu_bat_reduce_fp32_mfma_amd_op.register_fake
def _(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if A.dim() != 2 or B.dim() != 2:
        raise RuntimeError("expected 2D tensors")
    _, D = A.shape
    _, DB = B.shape
    if DB != D:
        raise RuntimeError("shape mismatch")
    if BM <= 0 or BN <= 0 or M_TILES <= 0:
        raise RuntimeError("BM, BN, and M_TILES must be positive")
    if BN % 16 != 0:
        raise RuntimeError("BN must be divisible by 16")
    if BM % (M_TILES * 16) != 0:
        raise RuntimeError("BM must be divisible by M_TILES * 16")
    return A.new_empty((1,)), A.new_empty((1,))


def _setup_context(ctx, inputs, output) -> None:
    A, B = inputs[:2]
    ctx.save_for_backward(A, B)


def _backward(
    ctx,
    grad_sum_sr: torch.Tensor | None,
    grad_sum_sr2: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    None,
    None,
    None,
]:
    A, B = ctx.saved_tensors

    needs_A_grad = ctx.needs_input_grad[0]
    needs_B_grad = ctx.needs_input_grad[1]
    grad_A = None
    grad_B = None
    relu_values = None

    def dense_relu_values() -> torch.Tensor:
        nonlocal relu_values
        if relu_values is None:
            relu_values = torch.relu(A @ B.T)
        return relu_values

    if grad_sum_sr2 is not None:
        scale = 2.0 * grad_sum_sr2.to(dtype=A.dtype)
        if needs_A_grad:
            grad_A = scale * (dense_relu_values() @ B)
        if needs_B_grad:
            grad_B = scale * (dense_relu_values().T @ A)

    if grad_sum_sr is not None and (needs_A_grad or needs_B_grad):
        grad_sum_sr = grad_sum_sr.to(dtype=A.dtype)
        relu_mask = (dense_relu_values() > 0).to(dtype=A.dtype)
        if needs_A_grad:
            grad_A_sum = grad_sum_sr * (relu_mask @ B)
            grad_A = grad_A_sum if grad_A is None else grad_A + grad_A_sum
        if needs_B_grad:
            grad_B_sum = grad_sum_sr * (relu_mask.T @ A)
            grad_B = grad_B_sum if grad_B is None else grad_B + grad_B_sum

    return grad_A, grad_B, None, None, None


relu_bat_reduce_fp32_mfma_amd_op.register_autograd(
    _backward,
    setup_context=_setup_context,
)


__all__ = ["relu_bat_reduce_fp32_mfma_amd_op"]
