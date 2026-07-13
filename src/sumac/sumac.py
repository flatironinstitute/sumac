import torch
from torch import Tensor
from torch.utils.data import DataLoader
import math
import time
from pathlib import Path
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
from .kernels.tuning import (
    AutotuneMode,
    default_kernel_autotune_cache_dir,
    kernel_autotune_options,
    normalize_autotune_mode,
)
from .eval import block_loss_and_pred, eval

SumacMethod = Literal["SALSA", "GD"]
OptimizerName = Literal["adam", "adamw", "sgd", "muon"]
SUMAC_METHODS: tuple[SumacMethod, ...] = ("SALSA", "GD")
OPTIMIZER_NAMES: tuple[OptimizerName, ...] = ("adam", "adamw", "sgd", "muon")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_sumac_inputs(
    *,
    S_index: Tensor,
    S_value: Tensor,
    shape,
    rank: int,
    max_iterations: int,
    num_blocks: int | None,
    cache_mb: int,
    method: str,
    momentum: float,
    learning_rate: float,
    optimizer: str,
    adam_betas,
    adam_eps: float,
    muon_momentum: float,
    seed: int,
    eval_interval: int | None,
    A_init: Tensor | None,
    B_init: Tensor | None,
) -> tuple[int, int]:
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
    if S_value.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "S_value must have dtype torch.float32 or torch.float64, "
            f"got {S_value.dtype}"
        )
    if not torch.isfinite(S_value).all().item():
        raise ValueError("S_value must contain only finite values")
    if S_value.numel() == 0:
        raise ValueError("S_value must contain at least one nonzero entry")
    if (S_value < 0).any().item():
        raise ValueError("S_value must contain only nonnegative values")
    if (S_value == 0).any().item():
        raise ValueError("S_value must contain only nonzero COO values")

    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError(f"shape must contain exactly two dimensions, got {shape!r}")
    if not all(_is_int(dim) and dim > 0 for dim in shape):
        raise ValueError(f"shape dimensions must be positive integers, got {shape!r}")
    m, n = shape

    if not _is_int(rank) or rank <= 0:
        raise ValueError(f"rank must be a positive integer, got {rank!r}")
    if not _is_int(max_iterations) or max_iterations < 0:
        raise ValueError(f"max_iterations must be a nonnegative integer, got {max_iterations!r}")
    if num_blocks is not None and (not _is_int(num_blocks) or num_blocks < 1):
        raise ValueError(f"num_blocks must be a positive integer or None, got {num_blocks!r}")
    if (
        not isinstance(cache_mb, (int, float))
        or isinstance(cache_mb, bool)
        or not math.isfinite(cache_mb)
        or cache_mb <= 0
    ):
        raise ValueError(f"cache_mb must be finite and positive, got {cache_mb!r}")
    if not _is_int(seed):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if eval_interval is not None and (not _is_int(eval_interval) or eval_interval <= 0):
        raise ValueError(f"eval_interval must be a positive integer or None, got {eval_interval!r}")

    def validate_finite_number(name, value):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number, got {value!r}")

    validate_finite_number("learning_rate", learning_rate)
    if learning_rate < 0:
        raise ValueError(f"learning_rate must be nonnegative, got {learning_rate!r}")

    if method == "SALSA" or optimizer == "sgd":
        validate_finite_number("momentum", momentum)
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must satisfy 0 <= momentum < 1, got {momentum!r}")

    if method == "GD" and optimizer in ("adam", "adamw"):
        validate_finite_number("adam_eps", adam_eps)
        if adam_eps <= 0:
            raise ValueError(f"adam_eps must be positive, got {adam_eps!r}")
        if not isinstance(adam_betas, (tuple, list)) or len(adam_betas) != 2:
            raise ValueError(f"adam_betas must contain two values, got {adam_betas!r}")
        for beta in adam_betas:
            if (
                not isinstance(beta, (int, float))
                or isinstance(beta, bool)
                or not math.isfinite(beta)
                or not 0 <= beta < 1
            ):
                raise ValueError(
                    "adam_betas values must satisfy "
                    f"0 <= beta < 1, got {adam_betas!r}"
                )

    if method == "GD" and optimizer == "muon":
        validate_finite_number("muon_momentum", muon_momentum)
        if not 0 <= muon_momentum < 1:
            raise ValueError(
                "muon_momentum must satisfy "
                f"0 <= muon_momentum < 1, got {muon_momentum!r}"
            )

    if (A_init is None) != (B_init is None):
        raise ValueError("A_init and B_init must either both be provided or both be None")
    if A_init is not None and B_init is not None:
        if not isinstance(A_init, Tensor) or not isinstance(B_init, Tensor):
            raise TypeError("A_init and B_init must be torch.Tensor instances")
        expected_A_shape = (m, rank)
        expected_B_shape = (n, rank)
        if tuple(A_init.shape) != expected_A_shape:
            raise ValueError(
                f"A_init must have shape {expected_A_shape}, got {tuple(A_init.shape)}"
            )
        if tuple(B_init.shape) != expected_B_shape:
            raise ValueError(
                f"B_init must have shape {expected_B_shape}, got {tuple(B_init.shape)}"
            )
        if not (A_init.is_floating_point() and B_init.is_floating_point()):
            raise TypeError("A_init and B_init must have floating-point dtypes")
        if A_init.dtype != S_value.dtype or B_init.dtype != S_value.dtype:
            raise TypeError(
                "A_init and B_init must have the same dtype as S_value "
                f"({S_value.dtype})"
            )
        if not torch.isfinite(A_init).all().item() or not torch.isfinite(B_init).all().item():
            raise ValueError("A_init and B_init must contain only finite values")

    if S_index.numel():
        rows, cols = S_index
        if (rows < 0).any().item() or (rows >= m).any().item():
            raise ValueError(f"S_index row indices must be in [0, {m})")
        if (cols < 0).any().item() or (cols >= n).any().item():
            raise ValueError(f"S_index column indices must be in [0, {n})")

    return m, n


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
    autotune: AutotuneMode = "cache",
    autotune_cache_dir: str | Path | None = None,
    autotune_verbose: bool = False,
):
    """
    Factorize a sparse nonnegative matrix with SUMAC.

    Args:
      S_index: sparse index of shape (2, nnz)
      S_value: values at the indices
      shape: matrix shape (m, n)
      rank: target rank
      max_iterations: maximum number of iterations
      S_value dtype: determines computation dtype; accepts float32 or float64
      device: optional training device. If omitted, infer from S_index and S_value.
      method: optimization routines, choices = ["GD", "SALSA"]
      optimizer args: adam_, muon_
      A_init, B_init: optional factor initialization; provide both or neither
      allow_tf32: If true, allow TF32 matmuls and SALSA TF32 relu_bat_c kernels
      autotune: "cache", "force", "disable", or "fallback"
      autotune_cache_dir: directory for kernel autotune JSON files
      autotune_verbose: print kernel autotune decisions
    """

    if method not in SUMAC_METHODS:
        raise ValueError(f"method must be one of {SUMAC_METHODS}, got {method!r}")
    if method == "GD" and optimizer not in OPTIMIZER_NAMES:
        raise ValueError(f"optimizer must be one of {OPTIMIZER_NAMES}, got {optimizer!r}")
    m, n = _validate_sumac_inputs(
        S_index=S_index,
        S_value=S_value,
        shape=shape,
        rank=rank,
        max_iterations=max_iterations,
        num_blocks=num_blocks,
        cache_mb=cache_mb,
        method=method,
        momentum=momentum,
        learning_rate=learning_rate,
        optimizer=optimizer,
        adam_betas=adam_betas,
        adam_eps=adam_eps,
        muon_momentum=muon_momentum,
        seed=seed,
        eval_interval=eval_interval,
        A_init=A_init,
        B_init=B_init,
    )
    autotune = normalize_autotune_mode(autotune)

    training_device = resolve_sumac_device(S_index, S_value, device)
    S_index = S_index.to(training_device)
    S_value = S_value.to(device=training_device)
    training_device = S_value.device

    coordinates = S_index.T
    if torch.unique(coordinates, dim=0).shape[0] != coordinates.shape[0]:
        raise ValueError("sumac: S_index contains duplicate matrix coordinates.")

    if eval_interval is None:
        eval_interval = 100 if method == "GD" else 10

    options = {
        'max_iterations': max_iterations,
        'dtype': S_value.dtype,
        'method': method,
        'seed': seed,
        'verbose': verbose,
        'cache_mb': cache_mb,
        'momentum': momentum,
        'eval_interval': eval_interval,
        'optimizer': optimizer,
        'allow_tf32': allow_tf32,
        'device': str(training_device),
        'autotune': autotune,
        'autotune_cache_dir': str(
            autotune_cache_dir
            if autotune_cache_dir is not None
            else default_kernel_autotune_cache_dir()
        ),
        'autotune_verbose': autotune_verbose,
    }

    automatic_num_blocks = num_blocks is None
    S_index, row_mask, col_mask = prune_zero_rows_cols(
        S_index, shape=(m, n), verbose=verbose
    )
    m_eff = row_mask.sum().item() if row_mask is not None else m
    n_eff = col_mask.sum().item() if col_mask is not None else n

    max_num_blocks = min(m_eff, n_eff)
    if automatic_num_blocks:
        max_bytes = cache_mb * 1e6
        bytes_per_dtype = S_value.element_size()
        cols_per_block = max(1, int(max_bytes // (m_eff * bytes_per_dtype)))
        num_blocks = min(math.ceil(n_eff / cols_per_block), max_num_blocks)
    else:
        cols_per_block = n_eff // num_blocks

    if not 1 <= num_blocks <= max_num_blocks:
        raise ValueError(
            "num_blocks must satisfy "
            f"1 <= num_blocks <= min(effective shape)={max_num_blocks}, got {num_blocks}"
        )
    options['num_blocks'] = num_blocks
    options['cols_per_block'] = cols_per_block

    if verbose:
        nnz = len(S_value)
        print(f"\n  Input to SUMAC is a {m}×{n} sparse matrix with {nnz} nonzeros.")
        print(f"  Effective matrix shape after pruning is {m_eff}×{n_eff}.")
        print(f"  Attempting matrix completion with rank {rank}.")
        print(f"  Available GPUs: {cuda_device_count()}.")
        print("  Options:")
        for k, v in options.items():
            print(f"    {k}: {v}")
        print()

    if A_init is not None and B_init is not None:
        A_init = A_init.to(device=training_device)
        B_init = B_init.to(device=training_device)
        if row_mask is not None:
            A_init = A_init[row_mask]
        if col_mask is not None:
            B_init = B_init[col_mask]

    gen = torch.Generator(device=training_device)
    if verbose:
        print(f"seed = {seed}")
    gen.manual_seed(seed)


    # call the core SUMAC loop 
    previous_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')
    nvtx_range_push("core SUMAC loop")
    try:
        with kernel_autotune_options(
            mode=autotune,
            cache_dir=autotune_cache_dir,
            verbose=autotune_verbose and verbose,
        ):
            if method == 'GD':
                cfg = TrainConfig(
                    rank,
                    num_blocks=num_blocks,
                    epochs=max_iterations,
                    lr=learning_rate,
                    seed=seed,
                    optimizer=optimizer,
                    momentum=momentum,
                    adam_betas=adam_betas,
                    adam_eps=adam_eps,
                    muon_momentum=muon_momentum,
                    eval_interval=eval_interval,
                    device=training_device,
                )
                A, B, costs = GD_loop(
                    S_index, S_value, m_eff, n_eff, cfg, gen, A_init, B_init,
                    verbose=verbose,
                )
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
    finally:
        try:
            nvtx_range_pop()
        finally:
            torch.set_float32_matmul_precision(previous_matmul_precision)
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
    if verbose:
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
    B_init: Tensor | None = None,
    verbose: bool = True,
):
    device = torch.device(cfg.device)

    # Move data to the training device.
    S_index = S_index.to(device)
    S_value = S_value.to(device)
    if A_init is None or B_init is None:
        scale = 0.5 * math.sqrt(S_value.mean().item() / cfg.d)
        A = torch.nn.Parameter(
            torch.rand(
                m,
                cfg.d,
                device=device,
                dtype=S_value.dtype,
                generator=gen,
            ) * scale
        )
        B = torch.nn.Parameter(
            torch.rand(
                n,
                cfg.d,
                device=device,
                dtype=S_value.dtype,
                generator=gen,
            ) * scale
        )
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
    loader_gen = torch.Generator()
    loader_gen.manual_seed(cfg.seed)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_blocks,
        shuffle=cfg.shuffle_blocks,
        collate_fn=collate_blocks,
        generator=loader_gen,
    )

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
    )
    rmse_hist = [rmse]
    jacc_hist = [jacc]
    time_hist = [0.0]
    if verbose:
        print(
            f"epoch = 0000, rmse = {rmse:.6f}, "
            f"jacc = {jacc:.6f}, errZ = {errZ:.6f}"
        )
    t0 = time.time()
    t_last_eval = t0

    for epoch in range(1, cfg.epochs + 1):
        nvtx_range_push("epoch: " + str(epoch))
        for blocks in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device, dtype=A.dtype)

            for block_id, edge_idx, row_indices in blocks:
                edge_idx = edge_idx.to(device).view(-1)
                mse_block, _, _, errZ_block = block_loss_and_pred(
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
                )
                loss_block = errZ_block if errZ_block is not None else mse_block
                loss = loss + loss_block

            loss.backward()
            apply_clip_and_step(opt, A, B, cfg)

        if epoch % cfg.eval_interval == 0:
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
            )
            elapsed = time.time() - t_last_eval
            rmse_hist.append(rmse)
            jacc_hist.append(jacc)
            time_hist.append(elapsed)
            if verbose:
                print(
                    f"epoch = {epoch:04d}, rmse = {rmse:.6f}, "
                    f"jacc = {jacc:.6f}, errZ = {errZ:.6f}, time = {elapsed:.2f}s"
                )
            t_last_eval = time.time()
        nvtx_range_pop()
    if verbose:
        total = time.time() - t0
        print(f"\nTotal elapsed time: {total:.2f} sec")

    history = {
        'rmse': rmse_hist,
        'jacc': jacc_hist,
        'time': time_hist,
    }
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
        dtype=opts.get('dtype', torch.float32),
    )
    S_index = S_index.to(device)
    S_value = S_value.to(device)
    
    gen_rows = torch.Generator(device=device)
    gen_rows.manual_seed(opts['seed'] + 1)
    gen_cols = torch.Generator(device=device)
    gen_cols.manual_seed(opts['seed'] + 2)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(opts['seed'] + 3)

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
    eval_loader = DataLoader(
        ds_rows,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_blocks,
        generator=loader_gen,
    )
    rmse, jacc, errZ = eval(
        A.to(device), B.to(device), S_index, S_value, m, n,
        opts['num_blocks'], eval_loader, device=device,
    )
    if opts['verbose']:
        print(f"iter = 0000, rmse = {rmse:.6f}, jacc = {jacc:.6f}, errZ = {errZ:.6}")
    nvtx_range_pop()
    rmse_hist = [rmse]
    jacc_hist = [jacc]
    time_hist = [0.0]
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
            eval_loader = DataLoader(
                ds_rows,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_blocks,
                generator=loader_gen,
            )
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
