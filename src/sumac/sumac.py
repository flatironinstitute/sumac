import torch
from torch import Tensor
from torch.utils.data import DataLoader
import math
import time
import random
from .data import prune_zero_rows_cols
from typing import Literal

from .kernels.cuda_utils import (
    cuda_device_count,
    nvtx_range_pop,
    nvtx_range_push,
    synchronize_if_cuda,
)
from .datasets import collate_blocks, StochasticRowBlockDataset
from .training.gd import TrainConfig, make_optimizer, apply_clip_and_step
from .training.salsa import configure_kernel_prec, init_salsa_factors, update_factor_salsa
from .eval import block_loss_and_pred, eval

SumacMethod = Literal["SALSA", "GD"]
OptimizerName = Literal["adam", "adamw", "sgd", "muon"]
SUMAC_METHODS: tuple[SumacMethod, ...] = ("SALSA", "GD")
OPTIMIZER_NAMES: tuple[OptimizerName, ...] = ("adam", "adamw", "sgd", "muon")


def resolve_sumac_device(
    S_index: Tensor,
    S_value: Tensor,
    device: torch.device | str | None,
) -> torch.device:
    if device is not None:
        return torch.device(device)
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


def sumac_factorize(
    *,
    S_index: Tensor,
    S_value: Tensor,
    shape: tuple[int, int],
    rank: int,
    method: SumacMethod = "SALSA",
    max_iterations: int = 25,
    num_blocks: int | None = None,
    cache_mb: int = 5000,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    momentum: float = 0.7,
    learning_rate: float = 1e-1,
    optimizer: OptimizerName = "adam",
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    muon_momentum: float = 0.95,
    seed: int = 0,
    eval_interval: int | None = None,
    verbose: bool = True,
    A_init: Tensor | None = None,
    B_init: Tensor | None = None,
    allow_tf32: bool = False,
):
    """
    Factorize a sparse nonnegative matrix with SUMAC.

    Args:
      S_index: sparse index of shape (2, nnz)
      S_value: values at the indices
      shape: matrix shape (m, n)
      rank: target rank
      max_iterations: maximum number of iterations
      dtype: default to torch.float32, accept torch.float64
      device: optional training device. If omitted, infer from S_index and S_value.
      method: optimization routines, choices = ["GD", "SALSA"]
      optimizer args: adam_, muon_
      A_init, B_init: allow user-specified factor initialization
      allow_tf32: If true, allow TF32 matmuls and SALSA TF32 relu_bat_c kernels
    """

    if method not in SUMAC_METHODS:
        raise ValueError(f"method must be one of {SUMAC_METHODS}, got {method!r}")
    if optimizer not in OPTIMIZER_NAMES:
        raise ValueError(f"optimizer must be one of {OPTIMIZER_NAMES}, got {optimizer!r}")
    m, n = shape

    training_device = resolve_sumac_device(S_index, S_value, device)
    S_index = S_index.to(training_device)
    S_value = S_value.to(device=training_device, dtype=dtype)
    training_device = S_value.device

    # verify input nonnegativity
    if (S_value < 0).any().item():
        raise ValueError("sumac: the input matrix should be nonnegative.")

    if eval_interval is None:
        eval_interval = 100 if method == "GD" else 10

    options = {
        'max_iterations': max_iterations,
        'dtype': dtype,
        'method': method,
        'seed': seed,
        'verbose': verbose,
        'cache_mb': cache_mb,
        'momentum': momentum,
        'eval_interval': eval_interval,
        'optimizer': optimizer,
        'allow_tf32': allow_tf32,
        'device': str(training_device),
    }

    torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')

    if num_blocks is None:
        max_bytes = cache_mb * 1e6
        bytes_per_dtype = 8 if dtype == torch.float64 else 4
        cols_per_block = max(1, int(max_bytes // (m * bytes_per_dtype)))
        num_blocks   = math.ceil(n / cols_per_block)
    else:
        cols_per_block = int(n // num_blocks)
    options['num_blocks'] = num_blocks
    options['cols_per_block'] = cols_per_block

    if verbose:
        nnz = len(S_value)
        print(f"\n  Input to SUMAC is a {m}×{n} sparse matrix with {nnz} nonzeros.")
        print(f"  Attempting matrix completion with rank {rank}.")
        print(f"  Available GPUs: {cuda_device_count()}.")
        print("  Options:")
        for k, v in options.items():
            print(f"    {k}: {v}")
        print()

    ## NEW: remove all-zero rows and/or columns
    S_index, row_mask, col_mask = prune_zero_rows_cols(S_index, shape=(m,n))
    m_eff = row_mask.sum().item() if row_mask is not None else m
    n_eff = col_mask.sum().item() if col_mask is not None else n

    # set random seed for reproducibility
    random.seed(seed)
    gen = torch.Generator(device=training_device)
    print(f"seed = {seed}")
    gen.manual_seed(seed)
    torch.manual_seed(seed)


    # call the core SUMAC loop 
    nvtx_range_push("core SUMAC loop")
    if method == 'GD':
        cfg = TrainConfig(
            rank,
            num_blocks=num_blocks,
            epochs=max_iterations,
            lr=learning_rate,
            optimizer=optimizer,
            momentum=momentum,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            muon_momentum=muon_momentum,
            eval_interval=eval_interval,
            device=training_device,
        )
        A, B, costs = GD_loop(S_index, S_value, m_eff, n_eff, cfg, gen, A_init, B_init)
    elif method == 'SALSA':
        A, B, costs = salsa_loop(
            S_index,
            S_value,
            m_eff,
            n_eff,
            rank,
            options,
            learning_rate,
            gen,
            training_device,
            A_init,
            B_init,
        )
    else:
        raise NotImplementedError("method must be chosen from GD or SALSA")
    nvtx_range_pop()
    # NEW: restore zero rows and columns
    if row_mask is not None:
        A_ori = torch.zeros((m, rank), dtype=A.dtype, device=A.device)
        A_ori[row_mask] = A 
    else:
        A_ori = A
    if col_mask is not None:
        B_ori = torch.zeros((n, rank), dtype=B.dtype, device=B.device)
        B_ori[col_mask] = B 
    else:
        B_ori = B
    print(f'SUMAC finished after {max_iterations} iterations.')
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


def salsa_loop(
    S_index: Tensor,
    S_value: Tensor,
    m: int,
    n: int,
    d: int,
    opts: dict,
    lrate: float | Tensor,
    gen: torch.Generator,
    device: torch.device,
    A_init: Tensor | None = None,
    B_init: Tensor | None = None
):
    """
    Minimal PyTorch version of the SALSA loop, reusing helpers from sumac.py.
    """
    configure_kernel_prec(
        allow_tf32=(
            bool(opts.get('allow_tf32', False)) and
            opts.get('dtype', torch.float32) == torch.float32
        ),
        device=device,
        D=d,
    )
    S_index = S_index.to(device)
    S_value = S_value.to(device)
    
    gen_rows = torch.Generator(device=device)
    gen_rows.manual_seed(opts['seed'] + 1)
    gen_cols = torch.Generator(device=device)
    gen_cols.manual_seed(opts['seed'] + 2)

    if A_init is None or B_init is None:
        nvtx_range_push("init_salsa_factors")
        A, B = init_salsa_factors(S_index, S_value, m, n, d, gen=gen)
        nvtx_range_pop()
    else:
        A, B = A_init.to(device), B_init.to(device)

    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)


    # Datasets for row and column blocks

    nvtx_range_push("StochasitcRowBlockDataset rows")
    ds_rows = StochasticRowBlockDataset(S_index, S_value, m, opts['num_blocks'], gen=gen_rows)
    S_index_T = S_index[[1, 0], :] 
    nvtx_range_pop()

    nvtx_range_push("StochasticRowBlockDataset cols")
    ds_cols = StochasticRowBlockDataset(S_index_T, S_value, n, opts['num_blocks'], gen=gen_cols)
    nvtx_range_pop()
    ##init evaluation

    nvtx_range_push("eval_loader init")
    eval_loader = DataLoader(ds_rows, batch_size=1, shuffle=False, collate_fn=collate_blocks)
    rmse, jacc, errZ = eval(A.to(device), B.to(device), S_index, S_value, m, n, opts['num_blocks'], 
                            eval_loader, device=device, errZ_obj=True)
    print(f"iter = 0000, rmse = {rmse:.6f}, jacc = {jacc:.6f}, errZ = {errZ:.6}")
    nvtx_range_pop()
    rmse_hist = []
    jacc_hist = []
    time_hist = []
    lrate = torch.tensor(lrate, device=A.device, dtype=A.dtype)
    t_start_loop = time.time()    
    
    momentum = torch.tensor(opts.get("momentum", 0.7), device=A.device, dtype=A.dtype)
    t_start = time.time()
    for iter_idx in range(1, opts['max_iterations'] + 1):
        nvtx_range_push("Iteration " + str(iter_idx))

        # Truly stochastic sampling: reshuffle partitions every epoch
        nvtx_range_push("reshuffle")
        ds_rows.reshuffle()
        ds_cols.reshuffle()
        block_order = list(range(opts['num_blocks']))
        nvtx_range_pop()
        #random.shuffle(block_order) -- only used for deterministic minibatch
        
        for mb_idx, block_id in enumerate(block_order):
            stepnum: int = mb_idx + 1 + (iter_idx - 1) * opts['num_blocks']
            unbias = 1 - (momentum ** stepnum)
            nvtx_range_push("update_factor_salsa B")

            # --- Update B ---
            B, dB = update_factor_salsa(S_index, S_value, ds_rows, block_id, A, B, dB, momentum, unbias, lrate)
            nvtx_range_pop()

            nvtx_range_push("update_factor_salsa A")
            # --- Update A ---
            A, dA = update_factor_salsa(S_index_T, S_value, ds_cols, block_id, B, A, dA, momentum, unbias, lrate)
            nvtx_range_pop()

        # Metrics and Reporting
        if iter_idx % opts['eval_interval'] == 0:
            eval_loader = DataLoader(ds_rows, batch_size=1, shuffle=False, collate_fn=collate_blocks)
            rmse, jacc, errZ = eval(
                A.to(device),
                B.to(device),
                S_index,
                S_value,
                m,
                n,
                opts['num_blocks'],
                eval_loader,
                device=device,
                errZ_obj=True,
            )

            if device.type == "cuda":
                synchronize_if_cuda(device)
            elapsed = time.time() - t_start
            rmse_hist.append(rmse)
            jacc_hist.append(jacc)
            time_hist.append(elapsed)
            
            if opts['verbose']:
                print(f"iter = {iter_idx:04d}, rmse = {rmse:.6f}, jacc = {jacc:.6f}, errZ = {errZ:.6}, time = {elapsed:.2f}s")
            t_start = time.time()
        nvtx_range_pop()
    # WRAP UP
    costs = {
        'rmse': rmse_hist,
        'jacc': jacc_hist,
        'time': time_hist
    }

    if opts['verbose']:
        total = time.time() - t_start_loop
        print(f"\nTotal elapsed time: {total:.2f} sec")

    return A, B, costs
