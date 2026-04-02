import torch
from relu_batc_jit.api import relu_bat_c_fused
from _sumac.tuning import *


def relu_bat_c_cuda_launcher():
    tune_config = {
        "BM": [32, 64, 96, 128, 256],
        "BK": [16, 32, 64],
        "num_ms": [1, 2, 4, 6],
    }

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_c_key,
        cache_path="relu_bat_c_jit_autotune.json",
        n_trials=1000,
        warmup=5,
        rep=50,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_batc(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BK: int,
        num_ms: int,
    ) -> torch.Tensor:
        return relu_bat_c_fused(A, B, C, BK=BK, MS=num_ms, BM=BM)

    return relu_batc


relu_bat_c_tuned = relu_bat_c_cuda_launcher()

def lsq_update_single_gpu(
    Ar_dev: torch.Tensor,
    B_blk_dev: torch.Tensor,
    pinvAt_dev: torch.Tensor,
    dB_blk_dev: torch.Tensor,
    blk_idx: torch.Tensor,   
    blk_vals: torch.Tensor,  
    momentum: torch.Tensor,
    unbias: torch.Tensor,
    BM: int,
    BK: int,
    MS: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    stepM_blk = relu_bat_c_fused(Ar_dev, B_blk_dev, pinvAt_dev, BM=BM, BK=BK, MS=MS)

    edge_i = blk_idx[0]
    edge_j = blk_idx[1]

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
    B_blk_new = B_blk_dev + dB_blk_new / unbias
    return B_blk_new, dB_blk_new


@torch.compile(mode='default',dynamic=True)
def batch_update_single_gpu(
    Sr_idx_raw: torch.Tensor,
    Sr_vals: torch.Tensor,
    Factor_fixed: torch.Tensor,
    row_indices_cpu: torch.Tensor,
    B: torch.Tensor,
    dB: torch.Tensor,
    momentum,
    unbias,
    m_fixed: int,
    BM,
    BK,
    MS,
):

    dev = B.device
    dtype = B.dtype

    m_batch = row_indices_cpu.shape[0]

    Ar_dev = Factor_fixed[row_indices_cpu, :]
    GramA = Ar_dev.T @ Ar_dev
    pinvAt_dev = torch.linalg.solve(GramA, Ar_dev.T).T

    blk_idx = Sr_idx_raw.to(dev).clone()
    blk_vals = Sr_vals.to(dev).clone()

    row_idx_dev = row_indices_cpu.to(dev)

    local_map = torch.zeros(m_fixed, dtype=torch.long, device=dev)

    local_map[row_idx_dev] = torch.arange(m_batch, device=dev)
    blk_idx[0] = local_map[blk_idx[0]]
    


    B_new, dB_new = lsq_update_single_gpu(
        Ar_dev=Ar_dev,
        B_blk_dev=B,
        pinvAt_dev=pinvAt_dev,
        dB_blk_dev=dB,
        blk_idx=blk_idx,
        blk_vals=blk_vals,
        momentum=momentum,
        unbias=unbias,
        BM=BM,
        BK=BK,
        MS=MS
    )

    return B_new, dB_new


def update_factor_salsa(
    S_idx_full,
    S_val_full,
    dataset,
    block_id,
    Factor_fixed,
    Factor_update,
    dFactor,
    momentum,
    unbias
):
    m_fixed = Factor_fixed.shape[0]
    _, edge_idx, row_indices = dataset[block_id]
    idx_raw = S_idx_full[:, edge_idx].clone()
    val_raw = S_val_full[edge_idx]

    params = relu_bat_c_tuned.resolve_params(Factor_fixed[row_indices, :], Factor_update, Factor_fixed[row_indices, :])
    BM = params["BM"]
    BK = params["BK"]
    MS = params["num_ms"]

    nextF, dF = batch_update_single_gpu(
        Sr_idx_raw=idx_raw,
        Sr_vals=val_raw,
        Factor_fixed=Factor_fixed,
        row_indices_cpu=row_indices,
        B=Factor_update,
        dB=dFactor,
        momentum=momentum,
        unbias=unbias,
        m_fixed=m_fixed,
        BM=BM,
        BK=BK,
        MS=MS,
    )
    
    return nextF, dF