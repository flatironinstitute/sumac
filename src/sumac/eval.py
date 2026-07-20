import torch

from .kernels.cuda_utils import nvtx_range, nvtx_range_pop, nvtx_range_push
from .kernels.tuning import KernelAutotuneOptions, active_kernel_autotune_options


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
    row_indices: torch.Tensor,
):
    row_indices = row_indices.to(device=rows_all.device, dtype=torch.long)
    local_map = torch.full(
        (A.shape[0],),
        fill_value=-1,
        dtype=torch.long,
        device=rows_all.device,
    )
    local_map[row_indices] = torch.arange(row_indices.numel(), device=rows_all.device)
    return local_map[rows_all]


def block_loss_and_pred(
    A: torch.Tensor,
    B: torch.Tensor,
    S_index: torch.Tensor,           # (2, nnz)
    S_value: torch.Tensor,           # (nnz,)
    edge_idx: torch.Tensor,          # indices into S_index/S_value for this block
    row_indices: torch.Tensor,
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

    row_indices = row_indices.to(device=A.device, dtype=torch.long).view(-1)
    A_block = A[row_indices, :]
    b = row_indices.numel()

    assert b > 0, "Empty block span"

    rows_all = S_index[0, edge_idx]
    cols_all = S_index[1, edge_idx]
    vals_all = S_value[edge_idx]
    
    rows_all = rows_all.to(device=A.device, dtype=torch.long)
    cols_all = cols_all.to(device=A.device, dtype=torch.long)
    vals_all = vals_all.to(device=A.device)

    local_r = compute_local_rows(
        A=A,
        rows_all=rows_all,
        row_indices=row_indices,
    )

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
    return mse_full, sumSr_block, jacc_num_block, errZ_num_block


@torch.no_grad()
def eval(
    A: torch.Tensor,
    B: torch.Tensor,
    S_index: torch.Tensor,
    S_value: torch.Tensor,
    full_block_loader,
    device: torch.device | None = None,
):
    if device is None:
        device = A.device

    ssqe = torch.zeros((), device=device, dtype=A.dtype)
    sumSr = torch.zeros((), device=device, dtype=A.dtype)
    num_j = torch.zeros((), device=device, dtype=A.dtype)
    errZ_num = torch.zeros((), device=device, dtype=A.dtype)

    for block in full_block_loader:
        for (_block_id, edge_idx, row_indices) in block:
            edge_idx = edge_idx.to(device).view(-1)

            nvtx_range_push("block_loss_and_pred")
            ssqe_b, sumSr_b, num_j_b, errZ_b = block_loss_and_pred(
                A,
                B,
                S_index,
                S_value,
                edge_idx, 
                row_indices=row_indices,
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
