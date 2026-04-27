import torch

from _sumac.dataset import block_span
from relu_bat_reduce_jit.api import relu_bat_reduce_fused
from _sumac.tuning import *

@torch.compile
def relu_bat_reduce_fallback(A: torch.Tensor,
                        B: torch.Tensor):
    A64 = A.to(torch.float64)
    B64 = B.to(torch.float64)
    Sr = torch.relu(A64 @ B64.T)
    
    sum_sr = Sr.sum()
    sum_sr2 = (Sr * Sr).sum()
    return sum_sr, sum_sr2


def relu_bat_reduce_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BK: int,
    num_ms: int,
) -> bool:
    if A.shape[1] >= 32 and num_ms > 4:
        return False

    if A.shape[1] >= 64 and num_ms > 2:  
        return False  
    
    props = torch.cuda.get_device_properties(torch.cuda.current_device)

    if props.shared_memory_per_block < 4 * BK * A.shape[1]:
        return False

    return True

def relu_bat_reduce_launcher():
    tune_config = {
        "BM": [32, 64, 128, 256],
        "BK": [16, 32, 64, 128],
        "num_ms": [1, 2, 4, 6],
    }

    @autotune_cuda_kernel(
        configs=tune_config,
        fallback_fn=relu_bat_reduce_fallback,
        constraint_fn=relu_bat_reduce_constraints,
        key_fn=relu_bat_reduce_key,
        cache_path="relu_bat_reduce_jit_autotune.json",
        n_trials=1000,
        warmup=1,
        rep=5,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_bat_reduce(
        A: torch.Tensor,
        B: torch.Tensor,
        BM: int,
        BK: int,
        num_ms: int,
    ) -> tuple[torch.Tensor,torch.Tensor]:
        return relu_bat_reduce_fused(A, B, BM, BK, num_ms)
    
    return relu_bat_reduce

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
    sum_sr, sum_sr2 = relu_bat_tuned(A_block, B)
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

    # 1. are there duplicate edge indices themselves?
    u_edge, c_edge = torch.unique(edge_idx, return_counts=True)
    if (c_edge > 1).any():
        print("duplicate edge_idx entries:",
            (c_edge > 1).sum().item(),
            "max multiplicity:", c_edge.max().item())

    # 2. are there duplicate global coordinates in THIS block?
    coords_g = torch.stack([rows_all, cols_all], dim=1)
    u_g, c_g = torch.unique(coords_g, dim=0, return_counts=True)
    if (c_g > 1).any():
        print("duplicate GLOBAL coords in block:",
            (c_g > 1).sum().item(),
            "max multiplicity:", c_g.max().item())

    # 3. are there duplicate local coordinates?
    coords_l = torch.stack([local_r, cols_all], dim=1)
    u_l, c_l = torch.unique(coords_l, dim=0, return_counts=True)
    if (c_l > 1).any():
        print("duplicate LOCAL coords in block:",
            (c_l > 1).sum().item(),
            "max multiplicity:", c_l.max().item())

    # 4. row_indices uniqueness
    u_rows, c_rows = torch.unique(row_indices, return_counts=True)
    if (c_rows > 1).any():
        print("duplicate row_indices:",
            (c_rows > 1).sum().item(),
            "max multiplicity:", c_rows.max().item())
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