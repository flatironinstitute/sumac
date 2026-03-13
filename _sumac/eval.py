import torch

from _sumac.dataset import block_span


@torch.compile(fullgraph=True, mode="max-autotune")
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

@torch.compile(fullgraph=True, mode="max-autotune")
def block_loss_errz(
    A_block: torch.Tensor,
    B: torch.Tensor,
    local_r: torch.Tensor,
    cols_all: torch.Tensor,
    vals_all: torch.Tensor,
):
    Sr = torch.relu(A_block @ B.T)
    sum_sr = Sr.sum()
    sum_sr2 = (Sr * Sr).sum()

    A_obs = A_block[local_r]      
    B_obs = B[cols_all]           
    L_obs = (A_obs * B_obs).sum(dim=1)

    sr_obs = torch.relu(L_obs)
    dot_obs = (vals_all * sr_obs).sum()
    jacc_num = torch.minimum(vals_all, sr_obs).sum()

    ssq_vals = (vals_all * vals_all).sum()
    mse_full = sum_sr2 - 2.0 * dot_obs + ssq_vals

    neg = torch.relu(-L_obs)
    corr = (neg * neg + 2.0 * vals_all * neg).sum()
    errZ_num = mse_full + corr

    return mse_full, sum_sr, jacc_num, errZ_num



def compute_local_rows(
    A: torch.Tensor,
    rows_all: torch.Tensor,
    row_indices: torch.Tensor = None,
    start: int = None,
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
    n: int,
    S_index: torch.LongTensor,        
    S_value: torch.Tensor,            
    edge_idx: torch.Tensor,           
    row_indices: torch.Tensor = None, 
    errZ_obj: bool = False,           
):
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

    if errZ_obj:
        mse_full, sumSr_block, jacc_num_block, errZ_num_block = block_loss_errz(A_block, B, local_r, cols_all, vals_all)
    else:
        mse_full, sumSr_block, jacc_num_block, errZ_num_block = block_loss_no_errz(A_block, B, local_r, cols_all, vals_all)

    return mse_full, sumSr_block, jacc_num_block, errZ_num_block


@torch.no_grad()
def eval(
    A,
    B,
    S_index,
    S_value,
    m,
    n,
    num_blocks,
    full_block_loader,   
    device=None,
    errZ_obj: bool = False,
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
                A, B, block_id, num_blocks, m, n,
                S_index, S_value, edge_idx,
                row_indices=row_indices,
                errZ_obj=errZ_obj,
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