import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from sumac.config import SumacConfig
from sumac.datasets import collate_blocks, StochasticRowBlockDataset
from sumac.eval import eval
from sumac.kernels.cuda_utils import nvtx_range_push, nvtx_range_pop, synchronize_if_cuda
from sumac.kernels.tuning import get_tunable_kernel, T_KernelTuner, AutotuneReluBatReduce


# TODO:
# - review
# - tighten
# - check inter-line todos
# - remove performance instrumentation


def refactor(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """
    Refactor A and B so A @ B.T is unchanged while both factors share singular values.
    """
    Qa, Ra = torch.linalg.qr(A, mode="reduced")
    Qb, Rb = torch.linalg.qr(B, mode="reduced")

    U, S, Vh = torch.linalg.svd(Ra @ Rb.T, full_matrices=False)
    sqrtS = torch.diag(torch.sqrt(S))

    Ar = Qa @ U @ sqrtS
    Br = Qb @ Vh.T @ sqrtS

    return Ar, Br


def init_salsa_factors(
    S_index: Tensor,
    S_value: Tensor,
    m: int,
    n: int,
    d: int,
    gen: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """
    Initialize with A > 0 and B < 0, then rescale.
    """
    device = S_value.device
    dtype = S_value.dtype

    S = torch.sparse_coo_tensor(
        S_index.to(device=device, dtype=torch.long),
        S_value,
        size=(m, n),
        device=device,
        dtype=dtype,
    ).coalesce()
    R_A = torch.rand((n, d), device=device, dtype=dtype, generator=gen)
    R_B = torch.rand((m, d), device=device, dtype=dtype, generator=gen)
    A = torch.sqrt(torch.sparse.mm(S, R_A))
    B = -torch.sqrt(torch.sparse.mm(S.T, R_B))

    Lij = torch.sum(A[S_index[0], :] * B[S_index[1], :], dim=1)

    ssqS = torch.sum(S_value ** 2)
    ssqL = torch.sum(Lij ** 2)
    SdotL = torch.sum(S_value * Lij)

    a = ssqL
    b = -2.0 * SdotL
    c = -3.0 * ssqS

    alpha = (-b + torch.sqrt(b**2 - 4 * a * c)) / (2 * a)
    scale = torch.clamp(torch.sqrt(alpha), 0.0, 1.0)
    A = scale * A
    B = scale * B

    return refactor(A, B)


# TODO: CONFIRM THAT COMPILATION STILL WORKS FOR THIS CASE
# TODO: Consider reducing parameters passed into here that
# are only used after the values are returned
def lsq_update_single_gpu(
    kernel: T_KernelTuner,
    Ar_dev: Tensor,
    B_blk_dev: Tensor,
    pinvAt_dev: Tensor,
    dB_blk_dev: Tensor,
    edge_i: Tensor,
    edge_j: Tensor,
    blk_vals: Tensor,  
    momentum: Tensor,
    unbias: Tensor,
    lrate: Tensor | float,
) -> tuple[Tensor, Tensor]:

    stepM_blk = kernel((Ar_dev, B_blk_dev, pinvAt_dev))

    Lij_blk = torch.sum(Ar_dev[edge_i, :] * B_blk_dev[edge_j, :], dim=1)
    Mij_blk = torch.relu(Lij_blk)
    Ct_vals = blk_vals - Lij_blk + Mij_blk

    
    stepC_blk = torch.zeros_like(B_blk_dev)
    
    stepC_blk.index_add_(
        0,
        edge_j,
        Ct_vals[:, None] * pinvAt_dev[edge_i, :],
    )

    lsqB_blk = B_blk_dev - stepM_blk + stepC_blk
    dB_blk_new = (lsqB_blk - B_blk_dev) * (1 - momentum) + dB_blk_dev * momentum
    B_blk_new = B_blk_dev + lrate * dB_blk_new / unbias
    return B_blk_new, dB_blk_new


@torch.compile(mode='max-autotune-no-cudagraphs')
def batch_update_single_gpu(
    kernel: T_KernelTuner,
    S_idx_full: Tensor,
    S_val_full: Tensor,
    edge_idx: Tensor,
    Factor_fixed: Tensor,
    row_indices: Tensor,
    B: Tensor,
    dB: Tensor,
    momentum: Tensor,
    unbias: Tensor,
    lrate: float | Tensor,
    m_fixed: int,
):
    dev = B.device

    m_batch = row_indices.shape[0]

    Ar_dev = Factor_fixed[row_indices, :]
    GramA = Ar_dev.T @ Ar_dev
    pinvAt_dev = torch.linalg.solve(GramA, Ar_dev.T).T

    blk_idx = S_idx_full[:, edge_idx]
    blk_vals = S_val_full[edge_idx]

    local_map = torch.zeros(m_fixed, dtype=torch.long, device=dev)

    local_map[row_indices] = torch.arange(m_batch, device=dev)
    edge_i = local_map[blk_idx[0]]
    edge_j = blk_idx[1]
    

    B_new, dB_new = lsq_update_single_gpu(
        kernel=kernel,
        Ar_dev=Ar_dev,
        B_blk_dev=B,
        pinvAt_dev=pinvAt_dev,
        dB_blk_dev=dB,
        edge_i=edge_i,
        edge_j=edge_j,
        blk_vals=blk_vals,
        momentum=momentum,
        unbias=unbias,
        lrate=lrate,
    )

    return B_new, dB_new


def update_factor_salsa(
    kernel: T_KernelTuner,
    S_idx_full: Tensor,
    S_val_full: Tensor,
    dataset: StochasticRowBlockDataset,
    block_id: int,
    Factor_fixed: Tensor,
    Factor_update: Tensor,
    dFactor: Tensor,
    momentum: Tensor,
    unbias: Tensor,
    lrate: float | Tensor,
):
    m_fixed = Factor_fixed.shape[0]
    _, edge_idx, row_indices = dataset[block_id]

    # Need to resolve CUDA autotune params outside of the compiled region.
    kernel.resolve_decision((
        Factor_fixed[row_indices, :],
        Factor_update,
        Factor_fixed[row_indices, :],
    ))

    nextF, dF = batch_update_single_gpu(
        kernel=kernel,
        S_idx_full=S_idx_full,
        S_val_full=S_val_full,
        edge_idx=edge_idx,
        Factor_fixed=Factor_fixed,
        row_indices=row_indices,
        B=Factor_update,
        dB=dFactor,
        momentum=momentum,
        unbias=unbias,
        lrate=lrate,
        m_fixed=m_fixed,
    )
    
    return nextF, dF


# TODO: Ask whether learning rate is still intended to be a scalar
# TODO: Can we break this into fewer than 140 lines maybe
def salsa_loop(
    S_index: Tensor,
    S_value: Tensor,
    m: int,
    n: int,
    cfg: SumacConfig,
    A_init: Tensor | None = None,
    B_init: Tensor | None = None
):
    """
    Minimal PyTorch version of the SALSA loop, reusing helpers from sumac.py.
    """
    assert cfg.num_blocks is not None
    assert cfg.device is not None

    (gen, gen_rows, gen_cols) = cfg.get_generator()
    assert gen_rows is not None
    assert gen_cols is not None


    S_index = S_index.to(cfg.device)
    S_value = S_value.to(cfg.device)
    if A_init is None or B_init is None:
        nvtx_range_push("init_salsa_factors")
        A, B = init_salsa_factors(
            S_index,
            S_value,
            m,
            n,
            cfg.rank,
            gen=gen,
        )
        nvtx_range_pop()
    else:
        A, B = A_init.to(cfg.device), B_init.to(cfg.device)

    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)

    # Datasets for row and column blocks

    nvtx_range_push("StochasitcRowBlockDataset rows")
    ds_rows = StochasticRowBlockDataset(S_index, S_value, m, cfg.num_blocks, gen=gen_rows)
    S_index_T = S_index[[1, 0], :] 
    nvtx_range_pop()

    nvtx_range_push("StochasticRowBlockDataset cols")
    ds_cols = StochasticRowBlockDataset(S_index_T, S_value, n, cfg.num_blocks, gen=gen_cols)
    nvtx_range_pop()
    ##init evaluation

    eval_kernel = AutotuneReluBatReduce()
    nvtx_range_push("eval_loader init")
    eval_loader = DataLoader(ds_rows, batch_size=1, shuffle=False, collate_fn=collate_blocks)
    rmse, jacc, errZ = eval(
        eval_kernel,
        A.to(cfg.device),
        B.to(cfg.device),
        S_index,
        S_value,
        eval_loader,
        device=cfg.device,
    )
    if cfg.verbose:
        print(f"iter = 0000, rmse = {rmse:.6f}, jacc = {jacc:.6f}, errZ = {errZ:.6}")
    nvtx_range_pop()
    rmse_hist = []
    jacc_hist = []
    time_hist = []
    lrate = torch.tensor(cfg.learning_rate, device=A.device, dtype=A.dtype)
    t_start_loop = time.time()    
    
    momentum = torch.tensor(cfg.momentum, device=A.device, dtype=A.dtype)
    kernel = get_tunable_kernel(cfg)
    t_start = time.time()
    for iter_idx in range(1, cfg.max_iterations + 1):
        nvtx_range_push("Iteration " + str(iter_idx))

        # Truly stochastic sampling: reshuffle partitions every epoch
        nvtx_range_push("reshuffle")
        ds_rows.reshuffle()
        ds_cols.reshuffle()
        block_order = list(range(cfg.num_blocks))
        nvtx_range_pop()
        #random.shuffle(block_order) -- only used for deterministic minibatch
        
        for mb_idx, block_id in enumerate(block_order):
            stepnum: int = mb_idx + 1 + (iter_idx - 1) * cfg.num_blocks
            unbias = 1 - (momentum ** stepnum)
            nvtx_range_push("update_factor_salsa B")

            # --- Update B ---
            B, dB = update_factor_salsa(kernel, S_index, S_value, ds_rows, block_id, A, B, dB, momentum, unbias, lrate)
            nvtx_range_pop()

            nvtx_range_push("update_factor_salsa A")
            # --- Update A ---
            A, dA = update_factor_salsa(kernel, S_index_T, S_value, ds_cols, block_id, B, A, dA, momentum, unbias, lrate)
            nvtx_range_pop()

        # Metrics and Reporting
        if cfg.eval_interval is not None and iter_idx % cfg.eval_interval == 0:
            eval_loader = DataLoader(ds_rows, batch_size=1, shuffle=False, collate_fn=collate_blocks)
            rmse, jacc, errZ = eval(
                eval_kernel,
                A.to(cfg.device),
                B.to(cfg.device),
                S_index,
                S_value,
                eval_loader,
                device=cfg.device,
            )

            if cfg.device.type == "cuda":
                synchronize_if_cuda(cfg.device)
            elapsed = time.time() - t_start
            rmse_hist.append(rmse)
            jacc_hist.append(jacc)
            time_hist.append(elapsed)
            
            if cfg.verbose:
                print(f"iter = {iter_idx:04d}, rmse = {rmse:.6f}, jacc = {jacc:.6f}, errZ = {errZ:.6}, time = {elapsed:.2f}s")
            t_start = time.time()
        nvtx_range_pop()
    # WRAP UP
    costs = {
        'rmse': rmse_hist,
        'jacc': jacc_hist,
        'time': time_hist
    }

    if cfg.verbose:
        total = time.time() - t_start_loop
        print(f"\nTotal elapsed time: {total:.2f} sec")

    return A, B, costs
