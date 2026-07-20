import torch

from .kernels.cuda_utils import nvtx_range, nvtx_range_pop, nvtx_range_push
from .kernels.tuning import KernelAutotuneOptions, active_kernel_autotune_options
from .datasets import block_span


relu_bat_tuned = None
relu_bat_reduce_kernel_mode = None
relu_bat_reduce_kernel_autotune_options: KernelAutotuneOptions | None = None


def get_relu_bat_reduce(A: torch.Tensor, B: torch.Tensor):
    global relu_bat_tuned
    global relu_bat_reduce_kernel_mode
    global relu_bat_reduce_kernel_autotune_options

    from .kernels.relu_bat_reduce import (
        relu_bat_reduce_fallback_launcher,
        relu_bat_reduce_launcher,
        select_relu_bat_reduce_kernel_mode,
    )

    autotune_options = active_kernel_autotune_options()
    mode = select_relu_bat_reduce_kernel_mode(A, B)
    if (
        mode == relu_bat_reduce_kernel_mode
        and relu_bat_tuned is not None
        and autotune_options == relu_bat_reduce_kernel_autotune_options
    ):
        return relu_bat_tuned

    if mode == "fallback":
        relu_bat_tuned = relu_bat_reduce_fallback_launcher()
    else:
        relu_bat_tuned = relu_bat_reduce_launcher(autotune_options)

    relu_bat_reduce_kernel_mode = mode
    relu_bat_reduce_kernel_autotune_options = autotune_options
    return relu_bat_tuned


@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def block_loss_no_errz(
    A_block: torch.Tensor,   
    B: torch.Tensor,         
    local_r: torch.Tensor,   
    cols_all: torch.Tensor, 
    vals_all: torch.Tensor,
):
    Sr = torch.relu(A_block @ B.T)

    sum_sr = Sr.sum()
    sum_sr2 = (Sr * Sr).sum()

    sr_obs = Sr[local_r, cols_all]
    dot_obs = (vals_all * sr_obs).sum()
    jacc_num = torch.minimum(vals_all, sr_obs).sum()
    ssq_vals = (vals_all * vals_all).sum()

    #||Sr - S||^2 where S is sparse and zero elsewhere
    mse_full = sum_sr2 - 2.0 * dot_obs + ssq_vals

    errZ_num = mse_full
    return mse_full, sum_sr, jacc_num, errZ_num


@torch.compile(dynamic=True)
def block_loss_errz(
    A_block: torch.Tensor,
    B: torch.Tensor,
    local_r: torch.Tensor,
    cols_all: torch.Tensor,
    vals_all: torch.Tensor,
    sum_sr: torch.Tensor,
    sum_sr2: torch.Tensor,
):
    vals_all64 = vals_all.to(torch.float64)
    A_obs = A_block[local_r].to(torch.float64)      
    B_obs = B[cols_all].to(torch.float64)           
    L_obs = (A_obs * B_obs).sum(dim=1)
    Mij = torch.relu(L_obs)

    obs_sq = (Mij * Mij).sum()
    obs_res_sq = ((vals_all64 - Mij) * (vals_all64 - Mij)).sum()
    mse_full = (sum_sr2.squeeze() - obs_sq) + obs_res_sq

    errZ_num = sum_sr2.squeeze() - obs_sq + ((vals_all64 - L_obs) * (vals_all64 - L_obs)).sum()

    jacc_num = torch.minimum(vals_all64, Mij).sum()
    
    return mse_full.to(torch.float32), sum_sr.squeeze().to(torch.float32), jacc_num.to(torch.float32), errZ_num.to(torch.float32)


def compute_local_rows(
    A: torch.Tensor,
    rows_all: torch.Tensor,
    row_indices: torch.Tensor | None = None,
    start: int | None = None,
):
    if row_indices is not None:
        row_indices = row_indices.to(device=rows_all.device, dtype=torch.long)
        b = row_indices.numel()

        local_map = torch.full(
            (A.shape[0],),
            fill_value=-1,
            dtype=torch.long,
            device=rows_all.device,
        )
        local_map[row_indices] = torch.arange(b, device=rows_all.device)
        local_r = local_map[rows_all]

        return local_r, b

    if start is None:
        raise ValueError("start must be provided when row_indices is None")

    local_r = rows_all - start
    return local_r, None


def block_loss_and_pred(
    A: torch.Tensor,
    B: torch.Tensor,
    block_id: int,
    num_blocks: int,
    m: int,
    n: int,     # NOTE UNUSED
    S_index: torch.Tensor,           # (2, nnz)
    S_value: torch.Tensor,           # (nnz,)
    edge_idx: torch.Tensor,          # indices into S_index/S_value for this block
    row_indices: torch.Tensor | None = None, # NEW: explicit row indices for this block
    errZ_obj: bool = False,          # whether use objective to min ||Z - L|| instead of ||S - Sr||
):
    """
    - Builds full block prediction Sr_I = ReLU(A_I @ B^T) (shape b x n).
    - Target is zero everywhere, except at observed entries (from edge_idx).
    - Loss: RMSE over the entire b*n matrix (sqrt of mean squared error) — used for backprop.
    - Metrics: returns scalar rmse (same as loss) and 1-Jaccard on the block.
    """
    # 1) Dense prediction for the block vs all columns (all-zero negatives)
    # torch.cuda.nvtx.range_push("get factor block")
    edge_idx = edge_idx.view(-1)

    if row_indices is not None:
        row_indices = row_indices.to(device=A.device, dtype=torch.long).view(-1)
        A_block = A[row_indices, :]
        b = row_indices.numel()
        start = None
    else:
        start, end = block_span(block_id, m, num_blocks)
        b = end - start
        A_block = A[start:end, :]

    assert b > 0, "Empty block span"

    rows_all = S_index[0, edge_idx]
    cols_all = S_index[1, edge_idx]
    vals_all = S_value[edge_idx]
    
    rows_all = rows_all.to(device=A.device, dtype=torch.long)
    cols_all = cols_all.to(device=A.device, dtype=torch.long)
    vals_all = vals_all.to(device=A.device)

    local_r, _ = compute_local_rows(
        A=A,
        rows_all=rows_all,
        row_indices=row_indices,
        start=start,
    )

#       torch.cuda.profiler.start()
    if errZ_obj:
        with nvtx_range("block_loss_errz"):
            reduce_kernel = get_relu_bat_reduce(A_block, B)
            reduce_kernel.resolve_params(A_block, B)
            sum_sr, sum_sr2 = reduce_kernel(A_block, B)
            mse_full, sumSr_block, jacc_num_block, errZ_num_block = block_loss_errz(
                A_block,
                B,
                local_r,
                cols_all,
                vals_all,
                sum_sr,
                sum_sr2,
            )
    else:
        with nvtx_range("block_loss_no_errz"):
            mse_full, sumSr_block, jacc_num_block, errZ_num_block = block_loss_no_errz(A_block, B, local_r, cols_all, vals_all)
#    torch.cuda.profiler.stop()
    return mse_full, sumSr_block, jacc_num_block, errZ_num_block


@torch.no_grad()
def eval(
    A: torch.Tensor,
    B: torch.Tensor,
    S_index: torch.Tensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    num_blocks: int,
    full_block_loader,   # yields (block_id, edge_idx) once per block_id # TODO
    device: torch.device | None = None,
    errZ_obj: bool = False,  # whether use objective to min ||Z - L|| instead of ||S - Sr||
):
    if device is None:
        device = A.device

    ssqe = torch.zeros((), device=device, dtype=A.dtype)
    sumSr = torch.zeros((), device=device, dtype=A.dtype)
    num_j = torch.zeros((), device=device, dtype=A.dtype)
    errZ_num = torch.zeros((), device=device, dtype=A.dtype)

    for block in full_block_loader:
        for (block_id, edge_idx, row_indices) in block:
            edge_idx = edge_idx.to(device).view(-1)
            block_id = int(block_id)

            nvtx_range_push("block_loss_and_pred")
            ssqe_b, sumSr_b, num_j_b, errZ_b = block_loss_and_pred(
                A,
                B,
                block_id,
                num_blocks,
                m,
                n,
                S_index,
                S_value,
                edge_idx, 
                row_indices=row_indices,
                errZ_obj=errZ_obj
            )
            nvtx_range_pop()

            ssqe += ssqe_b
            sumSr += sumSr_b
            num_j += num_j_b
            errZ_num += errZ_b

    S_norm = torch.norm(S_value)

    rmse = torch.sqrt(ssqe) / (S_norm + 1e-16)
    denom = S_value.sum() + sumSr - num_j
    jacc = 1.0 - num_j / (denom + 1e-16)
    errZ = torch.sqrt(errZ_num) / (S_norm + 1e-16)

    return float(rmse.item()), float(jacc.item()), float(errZ.item())
