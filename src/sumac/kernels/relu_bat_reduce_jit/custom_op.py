from __future__ import annotations
import torch

from .jit_kernel import relu_bat_reduce_fused as _relu_bat_reduce_fused_impl


@torch.library.custom_op("sumac::relu_bat_reduce_fused", mutates_args=(), device_types="cuda")
def relu_bat_reduce_fused_op(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BK: int,
    MS: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _relu_bat_reduce_fused_impl(A, B, BM, BK, MS)

@relu_bat_reduce_fused_op.register_fake
def relu_bat_reduce_fused_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BK: int,
    MS: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if A.dim() != 2 or B.dim() != 2:
        raise RuntimeError("expected 2D tensors")
    M, D = A.shape
    N, DB = B.shape
    if DB != D:
        raise RuntimeError("shape mismatch")
    return A.new_empty((1)), A.new_empty((1))


def relu_bat_reduce_fused_setup_context(ctx, inputs, output) -> None:
    A, B = inputs[:2]
    ctx.save_for_backward(A, B)


def relu_bat_reduce_fused_backward(
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
    # The backward products have an unfavorable shape for the custom relu_bat_c,
    # so we use plain torch here.
    relu_values = None

    def dense_relu_values() -> torch.Tensor:
        nonlocal relu_values
        if relu_values is None:
            relu_values = torch.relu(A @ B.T)
        return relu_values

    if grad_sum_sr2 is not None:
        scale = 2.0 * grad_sum_sr2.to(dtype=A.dtype)
        if needs_A_grad:
            # d/dA sum(ReLU(A @ B.T)^2) = 2 * ReLU(A @ B.T) @ B
            grad_A = scale * (dense_relu_values() @ B)
        if needs_B_grad:
            # d/dB sum(ReLU(A @ B.T)^2) = 2 * ReLU(A @ B.T).T @ A
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


relu_bat_reduce_fused_op.register_autograd(
    relu_bat_reduce_fused_backward,
    setup_context=relu_bat_reduce_fused_setup_context,
)
