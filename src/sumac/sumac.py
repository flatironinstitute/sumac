from contextlib import contextmanager
import math
import warnings
import torch
from torch import Tensor

from .kernels.tuning import kernel_autotune_options

from sumac.config import AutotuneMode, OptimizerName, SumacConfig, SumacMethod
from sumac.data import prune_zero_rows_cols, restore_zero_rows_cols
from sumac.kernels.cuda_utils import (
    cuda_device_count,
    nvtx_range_pop,
    nvtx_range_push,
)
from sumac.training.salsa import salsa_loop
from sumac.training.gd import GD_loop
from sumac.utils import resolve_sumac_device


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_finite_number(name: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


@contextmanager
def _matmul_precision(allow_tf32: bool):
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    with warnings.catch_warnings():
        if not allow_tf32:
            warnings.filterwarnings(
                "ignore",
                message=(
                    "TensorFloat32 tensor cores for float32 matrix multiplication "
                    "available but not enabled.*"
                ),
                category=UserWarning,
                module=r"torch\._inductor\.compile_fx",
            )
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(previous_precision)


def _validate_sumac_inputs(
    *,
    S_index: Tensor,
    S_value: Tensor,
    shape: object,
    A_init: Tensor | None,
    B_init: Tensor | None,
    config: SumacConfig,
) -> tuple[int, int]:
    if not isinstance(config, SumacConfig):
        raise TypeError("config must be a SumacConfig")
    if not isinstance(S_index, Tensor):
        raise TypeError("S_index must be a torch.Tensor")
    if not isinstance(S_value, Tensor):
        raise TypeError("S_value must be a torch.Tensor")
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

    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError(f"shape must contain exactly two dimensions, got {shape!r}")
    if not all(_is_int(dim) and dim > 0 for dim in shape):
        raise ValueError(f"shape dimensions must be positive integers, got {shape!r}")
    m, n = shape

    if not isinstance(config.method, SumacMethod):
        raise ValueError(f"config.method must be a SumacMethod, got {config.method!r}")
    if not isinstance(config.optimizer, OptimizerName):
        raise ValueError(f"config.optimizer must be an OptimizerName, got {config.optimizer!r}")
    if not isinstance(config.autotune, AutotuneMode):
        raise ValueError(f"config.autotune must be an AutotuneMode, got {config.autotune!r}")
    if not _is_int(config.rank) or config.rank <= 0:
        raise ValueError(f"config.rank must be a positive integer, got {config.rank!r}")
    if not _is_int(config.max_iterations) or config.max_iterations < 0:
        raise ValueError(
            "config.max_iterations must be a nonnegative integer, "
            f"got {config.max_iterations!r}"
        )
    if config.num_blocks is not None and (
        not _is_int(config.num_blocks) or config.num_blocks < 1
    ):
        raise ValueError(
            "config.num_blocks must be a positive integer or None, "
            f"got {config.num_blocks!r}"
        )
    _validate_finite_number("config.cache_mb", config.cache_mb)
    if config.cache_mb <= 0:
        raise ValueError(f"config.cache_mb must be positive, got {config.cache_mb!r}")
    if config.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "config.dtype must be torch.float32 or torch.float64, "
            f"got {config.dtype!r}"
        )
    if config.dtype == torch.float64 and config.allow_tf32:
        raise ValueError(
            "config.dtype=torch.float64 and config.allow_tf32=True are mutually exclusive"
        )
    if config.seed is not None and not _is_int(config.seed):
        raise ValueError(f"config.seed must be an integer or None, got {config.seed!r}")
    if not _is_int(config.eval_interval) or config.eval_interval <= 0:
        raise ValueError(
            "config.eval_interval must be a positive integer, "
            f"got {config.eval_interval!r}"
        )
    if not _is_int(config.batch_blocks) or config.batch_blocks <= 0:
        raise ValueError(
            "config.batch_blocks must be a positive integer, "
            f"got {config.batch_blocks!r}"
        )

    _validate_finite_number("config.learning_rate", config.learning_rate)
    if config.learning_rate < 0:
        raise ValueError(
            "config.learning_rate must be nonnegative, "
            f"got {config.learning_rate!r}"
        )
    _validate_finite_number("config.momentum", config.momentum)
    if not 0 <= config.momentum < 1:
        raise ValueError(
            "config.momentum must satisfy 0 <= momentum < 1, "
            f"got {config.momentum!r}"
        )
    if config.method == SumacMethod.GD and config.optimizer in (
        OptimizerName.ADAM,
        OptimizerName.ADAMW,
    ):
        _validate_finite_number("config.adam_eps", config.adam_eps)
        if config.adam_eps <= 0:
            raise ValueError(f"config.adam_eps must be positive, got {config.adam_eps!r}")
        if not isinstance(config.adam_betas, (tuple, list)) or len(config.adam_betas) != 2:
            raise ValueError(
                "config.adam_betas must contain two values, "
                f"got {config.adam_betas!r}"
            )
        for beta in config.adam_betas:
            _validate_finite_number("config.adam_betas value", beta)
            if not 0 <= beta < 1:
                raise ValueError(
                    "config.adam_betas values must satisfy 0 <= beta < 1, "
                    f"got {config.adam_betas!r}"
                )

    if (A_init is None) != (B_init is None):
        raise ValueError("A_init and B_init must either both be provided or both be None")
    if A_init is not None and B_init is not None:
        if not isinstance(A_init, Tensor) or not isinstance(B_init, Tensor):
            raise TypeError("A_init and B_init must be torch.Tensor instances")
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

    rows, cols = S_index
    if (rows < 0).any().item() or (rows >= m).any().item():
        raise ValueError(f"S_index row indices must be in [0, {m})")
    if (cols < 0).any().item() or (cols >= n).any().item():
        raise ValueError(f"S_index column indices must be in [0, {n})")
    if torch.unique(S_index.T, dim=0).shape[0] != S_index.shape[1]:
        raise ValueError("S_index contains duplicate matrix coordinates")

    return m, n


def sumac_factorize(
    *,
    S_index: Tensor,
    S_value: Tensor,
    shape: tuple[int, int],
    A_init: Tensor | None = None,
    B_init: Tensor | None = None,
    config: SumacConfig
):
    """
    Factorize a sparse nonnegative matrix with SUMAC.

    Args:
      S_index (Tensor): sparse index of shape (2, nnz)
      S_value (Tensor): values at the indices
      shape (tuple[int, int]): matrix shape (m, n)
      A_init (Tensor): Optional initial value for factor matrix A
      B_init (Tensor): Optional initial value for factor matrix B
      config (SumacConfig): Configuration option object
    """

    m, n = _validate_sumac_inputs(
        S_index=S_index,
        S_value=S_value,
        shape=shape,
        A_init=A_init,
        B_init=B_init,
        config=config,
    )

    with _matmul_precision(config.allow_tf32):
        config.device = resolve_sumac_device(S_index, S_value, config)
        S_index = S_index.to(config.device)
        S_value = S_value.to(device=config.device, dtype=config.dtype)

        S_index, row_mask, col_mask, m_eff, n_eff = prune_zero_rows_cols(
            S_index,
            shape=(m, n),
        )
        config.set_block_sizes(m_eff, n_eff)
        assert config.num_blocks is not None
        assert config.eval_interval is not None
        max_num_blocks = min(m_eff, n_eff)
        if not 1 <= config.num_blocks <= max_num_blocks:
            raise ValueError(
                "config.num_blocks must satisfy "
                f"1 <= num_blocks <= min(effective shape)={max_num_blocks}, "
                f"got {config.num_blocks}"
            )

        if A_init is not None and B_init is not None:
            A_init = A_init.to(config.device)
            B_init = B_init.to(config.device)
            if row_mask is not None:
                A_init = A_init[row_mask]
            if col_mask is not None:
                B_init = B_init[col_mask]

        config.print_prefactor_report(S_value, m, n, cuda_device_count())

        nvtx_range_push("core SUMAC loop")
        with kernel_autotune_options(
            mode=config.autotune.value,
            cache_dir=config.autotune_cache_dir,
            verbose=config.autotune_verbose,
        ):
            if config.method == SumacMethod.GD:
                A, B, costs = GD_loop(S_index, S_value, m_eff, n_eff, config, A_init, B_init)
            elif config.method == SumacMethod.SALSA:
                A, B, costs = salsa_loop(S_index, S_value, m_eff, n_eff, config, A_init, B_init)
            else:
                raise NotImplementedError("method must be chosen from GD or SALSA")
        nvtx_range_pop()

        A_ori, B_ori = restore_zero_rows_cols(A, B, m, n, config.rank, row_mask, col_mask)

        if config.verbose:
            print(f'SUMAC finished after {config.max_iterations} iterations.')
        return A_ori, B_ori, costs
