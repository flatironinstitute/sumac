import torch
from torch import Tensor
from torch.utils.data import DataLoader
import math
import time
import random
from pathlib import Path
from .data import prune_zero_rows_cols

from .kernels.cuda_utils import (
    cuda_device_count,
    nvtx_range_pop,
    nvtx_range_push,
    synchronize_if_cuda,
)
from .datasets import collate_blocks, StochasticRowBlockDataset
from .training.gd import TrainConfig, make_optimizer, apply_clip_and_step
from .training.salsa import configure_kernel_prec, init_salsa_factors, update_factor_salsa
from .kernels.tuning import kernel_autotune_options
from .eval import block_loss_and_pred, eval

from sumac.training.salsa import salsa_loop
from sumac.config.options import SumacConfig, SumacMethod, OptimizerName


def resolve_sumac_device(
    S_index: Tensor,
    S_value: Tensor,
    config: SumacConfig
) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)

    if S_index.device == S_value.device:
        return S_value.device
    if S_value.device.type != "cpu" and S_index.device.type == "cpu":
        return S_value.device
    if S_index.device.type != "cpu" and S_value.device.type == "cpu":
        return S_index.device
    raise ValueError(
        "S_index and S_value are on different non-CPU devices. "
        "Pass device=... explicitly to choose the SUMAC training device."
    )

    ## TODO: Document this on the SumacConfig itself
    # # #   rank: target rank
    # # #   max_iterations: maximum number of iterations
    # # #   dtype: default to torch.float32, accept torch.float64
    # # #   device: optional training device. If omitted, infer from S_index and S_value.
    # # #   method: optimization routines, choices = ["GD", "SALSA"]
    # # #   optimizer args: adam_, muon_
    # # #   A_init, B_init: allow user-specified factor initialization
    # # #   allow_tf32: If true, allow TF32 matmuls and SALSA TF32 relu_bat_c kernels
    # # #   autotune: "cache", "force", "disable", or "fallback"
    # # #   autotune_cache_dir: directory for kernel autotune JSON files
    # # #   autotune_verbose: print kernel autotune decisions



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

    m, n = shape
    config.device = resolve_sumac_device(S_index, S_value, config)
    S_index = S_index.to(config.device)
    S_value = S_value.to(device=config.device, dtype=config.dtype)
    torch.set_float32_matmul_precision('high' if config.allow_tf32 else 'highest')
    config.set_cols_per_block(m, n)
    config.print_prefactor_report(S_value, m, n, cuda_device_count())

    # verify input nonnegativity
    if (S_value < 0).any().item():
        raise ValueError("sumac: the input matrix should be nonnegative.")

    ## NEW: remove all-zero rows and/or columns
    S_index, row_mask, col_mask = prune_zero_rows_cols(S_index, shape=(m,n))
    m_eff = row_mask.sum().item() if row_mask is not None else m
    n_eff = col_mask.sum().item() if col_mask is not None else n

    # set random seed for reproducibility
    if config.seed is not None:
        print(f"seed = {config.seed}")
        random.seed(config.seed)
        gen = torch.Generator(device=config.device)
        gen.manual_seed(config.seed)
        torch.manual_seed(config.seed)

    assert config.num_blocks is not None
    assert config.eval_interval is not None

    # call the core SUMAC loop 
    nvtx_range_push("core SUMAC loop")
    with kernel_autotune_options(
        mode=config.autotune.value,
        cache_dir=config.autotune_cache_dir,
        verbose=config.autotune_verbose,
    ):
        if config.method == SumacMethod.GD:
            cfg = TrainConfig(
                config.rank,
                num_blocks=config.num_blocks,
                epochs=config.max_iterations,
                lr=config.learning_rate,
                optimizer=config.optimizer,
                momentum=config.momentum,
                adam_betas=config.adam_betas,
                adam_eps=config.adam_eps,
                muon_momentum=config.muon_momentum,
                eval_interval=config.eval_interval,
                device=config.device,
            )
            A, B, costs = GD_loop(S_index, S_value, m_eff, n_eff, cfg, gen, A_init, B_init)
        elif config.method == SumacMethod.SALSA:
            A, B, costs = salsa_loop(S_index, S_value, m_eff, n_eff, config, A_init, B_init)
        else:
            raise NotImplementedError("method must be chosen from GD or SALSA")
    nvtx_range_pop()
    
    # NEW: restore zero rows and columns
    if row_mask is not None:
        A_ori = torch.zeros((m, config.rank), dtype=A.dtype, device=A.device)
        A_ori[row_mask] = A 
    else:
        A_ori = A
    if col_mask is not None:
        B_ori = torch.zeros((n, config.rank), dtype=B.dtype, device=B.device)
        B_ori[col_mask] = B 
    else:
        B_ori = B
    print(f'SUMAC finished after {config.max_iterations} iterations.')
    return A_ori, B_ori, costs


def GD_loop(
    S_index: Tensor,
    S_value: Tensor,
    m: int,
    n: int,
    cfg: TrainConfig,
    gen: torch.Generator,
    A_init: Tensor | None = None,
    B_init: Tensor | None = None
):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Move data to the training device.
    S_index = S_index.to(device)
    S_value = S_value.to(device)
    if A_init is None or B_init is None:
        scale = 0.5 * math.sqrt(S_value.mean().item() / cfg.d)
        A = torch.nn.Parameter(torch.rand(m, cfg.d, device=device) * scale)
        B = torch.nn.Parameter(torch.rand(n, cfg.d, device=device) * scale)
    else:
        A = torch.nn.Parameter(A_init.to(device))
        B = torch.nn.Parameter(B_init.to(device))
    opt = make_optimizer(
        cfg.optimizer,
        [A, B],
        lr=cfg.lr,
        weight_decay=0.0,
        momentum=cfg.momentum,
        adam_betas=cfg.adam_betas,
        adam_eps=cfg.adam_eps,
        muon_momentum=cfg.muon_momentum
    )

    # DataLoader
    ds = StochasticRowBlockDataset(S_index, S_value, m, cfg.num_blocks, gen)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_blocks,
        shuffle=cfg.shuffle_blocks,
        collate_fn=collate_blocks
    )

    history = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        nvtx_range_push("epoch: " + str(epoch))
        total_loss, sumSr, num_jacc = 0.0, 0.0, 0.0
        t_start = time.time()

        for blocks in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device, dtype=A.dtype)
            sumSr_batch = torch.zeros((), device=device, dtype=A.dtype)
            num_jacc_batch = torch.zeros((), device=device, dtype=A.dtype)

            for block_id, edge_idx, row_indices in blocks:
                edge_idx = edge_idx.to(device).view(-1)
                mse_block, sumSr_block, jacc_num_block, errZ_block = block_loss_and_pred(
                    A,
                    B,
                    block_id=int(block_id),
                    num_blocks=cfg.num_blocks,
                    m=m,
                    n=n,
                    S_index=S_index,
                    S_value=S_value,
                    edge_idx=edge_idx,
                    row_indices=row_indices,
                    errZ_obj=True,
                )
                loss_block = errZ_block if errZ_block is not None else mse_block
                loss = loss + loss_block
                sumSr_batch = sumSr_batch + torch.as_tensor(sumSr_block, device=device, dtype=A.dtype)
                num_jacc_batch = num_jacc_batch + jacc_num_block

            loss.backward()
            apply_clip_and_step(opt, A, B, cfg)

            total_loss += float(loss.detach().item())
            sumSr += float(sumSr_batch.detach().item())
            num_jacc += float(num_jacc_batch.detach().item())

        time_step = time.time() - t_start
        denom_jacc = float(torch.sum(S_value).item()) + sumSr - num_jacc
        jacc = 1.0 - num_jacc / denom_jacc
        S_norm = float(torch.norm(S_value).item())
        rmse = math.sqrt(total_loss) / (S_norm + 1e-16)
        log = f"[epoch {epoch}/{cfg.epochs}]: rmse={rmse:.6f}, jacc={jacc:.6f}, factor_step ={time_step:6.4f}"
        print(log)
        history.append(log)
        nvtx_range_pop()
    total = time.time() - t0
    print(f"\nTotal elapsed time: {total:.2f} sec")

    rmse, jacc, errZ = eval(
        A,
        B,
        S_index,
        S_value,
        m,
        n,
        num_blocks=cfg.num_blocks,
        full_block_loader=loader,
        device=A.device,
        errZ_obj=True
    )
    print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")

    return A.detach(), B.detach(), history
