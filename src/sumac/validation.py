import torch
from torch import Tensor

from sumac.config import SumacConfig


def _validate_shape(shape: tuple[int, int]) -> None:
    if not all(dim > 0 for dim in shape):
        raise ValueError(f"shape dimensions must be positive integers, got {shape!r}")


def _validate_matrix_input(
    S_index: Tensor,
    S_value: Tensor,
    *,
    shape: tuple[int, int],
) -> None:
    if S_index.ndim != 2 or S_index.shape[0] != 2:
        raise ValueError(f"S_index must have shape (2, nnz), got {tuple(S_index.shape)}")
    if S_index.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"S_index must have an integer index dtype, got {S_index.dtype}")
    if S_value.ndim != 1:
        raise ValueError(f"S_value must have shape (nnz,), got {tuple(S_value.shape)}")
    if S_value.shape[0] != S_index.shape[1]:
        raise ValueError(
            "S_index and S_value nnz dimensions must match, got "
            f"{S_index.shape[1]} and {S_value.shape[0]}"
        )
    if not S_value.is_floating_point():
        raise TypeError(f"S_value must have a floating-point dtype, got {S_value.dtype}")
    if S_value.numel() == 0:
        raise ValueError("S_value must contain at least one nonzero entry")
    if not torch.isfinite(S_value).all().item():
        raise ValueError("S_value must contain only finite values")
    if (S_value < 0).any().item():
        raise ValueError("S_value must contain only nonnegative values")
    if (S_value == 0).any().item():
        raise ValueError("S_value must contain only nonzero COO values")

    m, n = shape
    rows, cols = S_index
    if (rows < 0).any().item() or (rows >= m).any().item():
        raise ValueError(f"S_index row indices must be in [0, {m})")
    if (cols < 0).any().item() or (cols >= n).any().item():
        raise ValueError(f"S_index column indices must be in [0, {n})")
    if torch.unique(S_index.T, dim=0).shape[0] != S_index.shape[1]:
        raise ValueError("S_index contains duplicate matrix coordinates")


def _validate_warm_start_matrices(
    A_init: Tensor | None,
    B_init: Tensor | None,
    *,
    shape: tuple[int, int],
    config: SumacConfig,
) -> None:
    if (A_init is None) != (B_init is None):
        raise ValueError("A_init and B_init must either both be provided or both be None")
    if A_init is None or B_init is None:
        return

    m, n = shape
    expected_A_shape = (m, config.rank)
    expected_B_shape = (n, config.rank)
    if tuple(A_init.shape) != expected_A_shape:
        raise ValueError(
            f"A_init must have shape {expected_A_shape}, got {tuple(A_init.shape)}"
        )
    if tuple(B_init.shape) != expected_B_shape:
        raise ValueError(
            f"B_init must have shape {expected_B_shape}, got {tuple(B_init.shape)}"
        )
    if A_init.dtype != config.dtype or B_init.dtype != config.dtype:
        raise TypeError(
            "A_init and B_init must have the configured dtype "
            f"({config.dtype})"
        )
    if not torch.isfinite(A_init).all().item() or not torch.isfinite(B_init).all().item():
        raise ValueError("A_init and B_init must contain only finite values")


def validate_sumac_inputs(
    *,
    S_index: Tensor,
    S_value: Tensor,
    shape: tuple[int, int],
    A_init: Tensor | None,
    B_init: Tensor | None,
    config: SumacConfig,
) -> tuple[int, int]:
    _validate_shape(shape)
    _validate_matrix_input(S_index, S_value, shape=shape)
    _validate_warm_start_matrices(
        A_init,
        B_init,
        shape=shape,
        config=config,
    )
    return shape
