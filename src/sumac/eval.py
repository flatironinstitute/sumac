import torch

from .kernels.cuda_utils import cuda_is_available, nvtx_range, nvtx_range_pop, nvtx_range_push
from .kernels.tuning import KernelAutotuneOptions, active_kernel_autotune_options
from .datasets import block_span


relu_bat_tuned = None
relu_bat_reduce_kernel_mode = None
relu_bat_reduce_kernel_autotune_options: KernelAutotuneOptions | None = None


def get_relu_bat_reduce(A: torch.Tensor, B: torch.Tensor):
    global relu_bat_tuned
    global relu_bat_reduce_kernel_mode
    global relu_bat_reduce_kernel_autotune_options

    autotune_options = active_kernel_autotune_options()
    if not (
        A.dtype == torch.float32
        and B.dtype == torch.float32
        and A.is_cuda
        and B.is_cuda
        and cuda_is_available()
    ):
        if relu_bat_reduce_kernel_mode != "fallback" or relu_bat_tuned is None:
            from .kernels.relu_bat_reduce import (
                relu_bat_reduce_fallback_launcher,
            )

            relu_bat_tuned = relu_bat_reduce_fallback_launcher()
            relu_bat_reduce_kernel_mode = "fallback"
            relu_bat_reduce_kernel_autotune_options = autotune_options
        return relu_bat_tuned

    if (
        relu_bat_reduce_kernel_mode != "cuda" or
        relu_bat_tuned is None or
        autotune_options != relu_bat_reduce_kernel_autotune_options
    ):
        from .kernels.relu_bat_reduce import relu_bat_reduce_launcher

        relu_bat_tuned = relu_bat_reduce_launcher(autotune_options)
        relu_bat_reduce_kernel_mode = "cuda"
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
    
    output_dtype = A_block.dtype
    return (
        mse_full.to(output_dtype),
        sum_sr.squeeze().to(output_dtype),
        jacc_num.to(output_dtype),
        errZ_num.to(output_dtype),
    )


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

        if (local_r < 0).any().item():
            raise ValueError("rows_all contains rows that are not present in row_indices")

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
    if ((local_r < 0) | (local_r >= b)).any().item():
        raise ValueError("edge_idx contains rows outside the selected row block")

#       torch.cuda.profiler.start()
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
