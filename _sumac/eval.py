import torch

from _sumac.dataset import block_span
from _sumac.relu_bat_reduce_kernel_helper import relu_bat_reduce_launcher


relu_bat_tuned = relu_bat_reduce_launcher()


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
):
    sum_sr, sum_sr2 = relu_bat_tuned(A_block, B) # type: ignore
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


# # #jit helper function for speed up
# # @torch.compile(mode='max-autotune-no-cudagraphs')
# # def relu_AB(A: torch.Tensor, B: torch.Tensor):
# #     return torch.relu(A @ B.T)


# # @torch.compile(mode="max-autotune-no-cudagraphs")
# # def relu_AB_with_Lobs(A: torch.Tensor, B: torch.Tensor,
# #                      local_r: torch.Tensor,
# #                      cols_all: torch.Tensor):
# #     L = A @ B.T
# #     Sr_block = torch.relu(L)
# #     L_obs = L[local_r, cols_all]
# #     return Sr_block, L_obs


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

    ## TODO: Confirm we don't need/won't use this code, should be OK to delete
    # # torch.cuda.nvtx.range_pop()

    # # torch.cuda.nvtx.range_push("map idx to local block")
    # # rows_all = S_index[0][edge_idx]         # global rows
    # # cols_all = S_index[1][edge_idx]         # global cols
    # # vals_all = S_value[edge_idx]            # (E_b,)
    
    # # # Mapping global row indices to local indices [0, b)
    # # if row_indices is not None: 
    # #     local_map = torch.zeros(A.shape[0], dtype=torch.long, device=rows_all.device)
    # #     local_map[row_indices.to(rows_all.device)] = torch.arange(b, device=rows_all.device)
    # #     local_r = local_map[rows_all]
    # # else:
    # #     local_r = rows_all - start
    # # torch.cuda.nvtx.range_pop()
    
    # # torch.cuda.nvtx.range_push("compute pred and target")
    # # if errZ_obj:
    # #     Sr_block, L_obs = relu_AB_with_Lobs(A_block, B, local_r, cols_all)
    # # else:
    # #     Sr_block = relu_AB(A_block, B)
    # # sumSr_block = Sr_block.sum()

    # # target = Sr_block.new_zeros((b, n))      # (b, n)    
    # # target[local_r, cols_all] = vals_all
    # # torch.cuda.nvtx.range_pop()

    # # # MSE/jacc numerator over *all* entries in the block (loss used for backprop)
    # # torch.cuda.nvtx.range_push("compute losses")
    # # mse_full = F.mse_loss(Sr_block, target, reduction="sum")  # sum over all b*n -> dense compute
    # # Sr_obs = Sr_block[local_r, cols_all]
    # # # ssqS_block = (vals_all * vals_all).sum()
    # # # mse_full = ssqS_block + ssqSr_block - 2.0 * (vals_all * Sr_obs).sum() #this turns out to be slightly slower
    # # jacc_num_block = torch.minimum(vals_all, Sr_obs).sum()
    # # errZ_num_block = None
    # # if errZ_obj: #make equivalent errZ objective;
    # #     # L at observed (local) coordinates
    # #     neg_mask_pos = L_obs < 0                  # only entries that were clamped in Sr
    # #     ## TODO -- double-check that there isn't something sneaky happening in the orig
    # #     # (which tested against neg_mask_pos.any() directly)
    # #     masking_happened = False
    # #     if isinstance(neg_mask_pos, torch.Tensor):
    # #         masking_happened = neg_mask_pos.any().item()
    # #     else:
    # #         masking_happened = neg_mask_pos
    # #     if masking_happened:
    # #         S_obs = vals_all[neg_mask_pos]        # S_{ij} at those coords
    # #         L_neg = L_obs[neg_mask_pos]           # L_{ij} (negative values)
    # #         # sum of (L^2 - 2 S L) over the intersection (observed & L<0)
    # #         errZ_num_block = mse_full + (L_neg*L_neg - 2.0*S_obs*L_neg).sum()
    # # torch.cuda.nvtx.range_pop()
    # # return mse_full, sumSr_block, jacc_num_block, errZ_num_block

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
        with torch.cuda.nvtx.range("block_loss_errz"):
            params = relu_bat_tuned.resolve_params(A_block, B)
            
            mse_full, sumSr_block, jacc_num_block, errZ_num_block = block_loss_errz(A_block, B, local_r, cols_all, vals_all)
    else:
        with torch.cuda.nvtx.range("block_loss_no_errz"):
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

            torch.cuda.nvtx.range_push("block_loss_and_pred")
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
            torch.cuda.nvtx.range_pop()

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
