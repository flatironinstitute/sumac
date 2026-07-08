import torch
from torch import Tensor
from ..datasets import StochasticRowBlockDataset
from ..kernels.cuda_utils import cuda_is_available
from ..kernels.relu_bat_c import relu_bat_c_fallback_launcher


relu_bat_c_tuned = relu_bat_c_fallback_launcher()
relu_bat_c_kernel_mode = "fallback"
relu_bat_c_kernel_d = None


def configure_kernel_prec(
    *,
    allow_tf32: bool,
    device,
    D: int,
) -> None:
    global relu_bat_c_tuned
    global relu_bat_c_kernel_mode
    global relu_bat_c_kernel_d

    device = torch.device(device)
    if device.type != "cuda" or not cuda_is_available():
        if relu_bat_c_kernel_mode != "fallback":
            relu_bat_c_tuned = relu_bat_c_fallback_launcher()
            relu_bat_c_kernel_mode = "fallback"
            relu_bat_c_kernel_d = D
        return

    from ..kernels.relu_bat_c import (
        relu_bat_c_cuda_launcher,
        relu_bat_c_tf32_sync_launcher,
        relu_bat_c_tf32_wgmma_launcher,
        select_relu_bat_c_kernel_mode,
    )

    mode = select_relu_bat_c_kernel_mode(allow_tf32, device)
    if mode == relu_bat_c_kernel_mode and D == relu_bat_c_kernel_d:
        return

    if mode == "tf32_wgmma":
        relu_bat_c_tuned = relu_bat_c_tf32_wgmma_launcher(D)
    elif mode == "tf32_mma_sync":
        relu_bat_c_tuned = relu_bat_c_tf32_sync_launcher(D)
    elif mode == "fallback":
        relu_bat_c_tuned = relu_bat_c_fallback_launcher()
    else:
        relu_bat_c_tuned = relu_bat_c_cuda_launcher()

    relu_bat_c_kernel_mode = mode
    relu_bat_c_kernel_d = D



def lsq_update_single_gpu(
    Ar_dev: torch.Tensor,
    B_blk_dev: torch.Tensor,
    pinvAt_dev: torch.Tensor,
    dB_blk_dev: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    blk_vals: torch.Tensor,  
    momentum: torch.Tensor,
    unbias: torch.Tensor,
    lrate: Tensor | float,
) -> tuple[Tensor, Tensor]:

    stepM_blk = relu_bat_c_tuned(Ar_dev, B_blk_dev, pinvAt_dev)

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
    S_idx_full: torch.Tensor,
    S_val_full: torch.Tensor,
    edge_idx: torch.Tensor,
    Factor_fixed: torch.Tensor,
    row_indices: torch.Tensor,
    B: torch.Tensor,
    dB: torch.Tensor,
    momentum,
    unbias,
    lrate,
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
    relu_bat_c_tuned.resolve_params(
        Factor_fixed[row_indices, :],
        Factor_update,
        Factor_fixed[row_indices, :],
    )

    nextF, dF = batch_update_single_gpu(
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
