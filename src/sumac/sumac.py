import torch
from torch import Tensor

from .kernels.tuning import kernel_autotune_options

from sumac.config.options import SumacConfig, SumacMethod
from sumac.data import prune_zero_rows_cols, restore_zero_rows_cols
from sumac.kernels.cuda_utils import (
    cuda_device_count,
    nvtx_range_pop,
    nvtx_range_push,
)
from sumac.training.salsa import salsa_loop
from sumac.training.gd import GD_loop
from sumac.utils import resolve_sumac_device


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

    torch.set_float32_matmul_precision('high' if config.allow_tf32 else 'highest')
    config.device = resolve_sumac_device(S_index, S_value, config)
    S_index = S_index.to(config.device)
    S_value = S_value.to(device=config.device, dtype=config.dtype)
    m, n = shape
    config.set_block_sizes(m, n)
    assert config.num_blocks is not None
    assert config.eval_interval is not None

    # verify input nonnegativity
    if (S_value < 0).any().item():
        raise ValueError("sumac: the input matrix should be nonnegative.")

    config.print_prefactor_report(S_value, m, n, cuda_device_count())

    S_index, row_mask, col_mask, m_eff, n_eff = prune_zero_rows_cols(S_index, shape=(m,n))

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

    print(f'SUMAC finished after {config.max_iterations} iterations.')
    return A_ori, B_ori, costs
