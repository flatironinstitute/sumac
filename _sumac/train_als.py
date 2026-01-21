import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import math


def least_squares_update_fast(
    S_idx: torch.LongTensor, #shape (2, nmz)
    S_vals: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    num_blocks: int,
    cols_per_block: int, 
    multi_gpu: bool = True,
) -> tuple[torch.Tensor, float, float]:
    """
    PyTorch version of the least_squares_update: faster version
        B = pinvA (-Sr + (S - ABt + Sr)[S>0])
        - within the block loop, compute the first term Sr
        - outside the block, compute the second term (S - ABt + Sr)[S>0]
    Args:
      S_index   : index of edge pairs
      S_values  : nonnegative entries
      A         : dense tensor, shape (m, r)
      B         : dense tensor, shape (n, r)
      cache_MB  : how many megabytes to dedicate per block
      multi_gpu : if True, will round-robin blocks across all available GPUs
    Returns:
      nextB : updated B (n, r)
      rmse  : float
      jacc  : float (1 – Jaccard similarity)
    """

    # ——————————————
    # 1) block setup (matches MATLAB)
    m = A.shape[0]
    n = B.shape[0]

    # ——————————————
    # 3) keep CPU copies of A, B, and compute pseudoinverse on CPU
    device_cpu = torch.device("cpu")
    A_cpu = A.to(device_cpu, copy=False)
    B_cpu = B.to(device_cpu, copy=False)
    pinvA_trans_cpu = torch.linalg.solve(A_cpu.T @ A_cpu, A_cpu.T).T   # shape (m,r)

    # ——————————————
    # 4) prepare for multi‐GPU and preload factors to device, add non_blocking
    if multi_gpu and torch.cuda.device_count() > 1:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        dev = torch.device("cuda:0") if torch.cuda.is_available() else device_cpu
        devices = [dev]
    A_devs      = [A_cpu.to(dev, non_blocking=True) for dev in devices]
    pinvA_trans_devs  = [pinvA_trans_cpu.to(dev, non_blocking=True) for dev in devices]
    B_devs      = [B_cpu.to(dev, non_blocking=True) for dev in devices] ##OLD
    dB   = torch.zeros_like(B_cpu) ##OLD
    # ——————————————
    # 5) accumulators
    # on-device metric accumulators (avoid .item() in the loop)
    sumSr_devs = [torch.zeros((), device=d, dtype=S_vals.dtype) for d in devices]
    ssqSr_devs = [torch.zeros((), device=d, dtype=S_vals.dtype) for d in devices]

    # 6) loop over blocks
    for b in range(num_blocks):
        start = b * cols_per_block
        end = min((b+1) * cols_per_block, n)
        dev_idx = b % len(devices)
        A_dev = A_devs[dev_idx]
        pinvA_trans_dev = pinvA_trans_devs[dev_idx]
        B_blk_dev = B_devs[dev_idx][start:end] #(block_size, r) ##OLD

        # 1) reconstruct full block and immediately sparsify
        dense_Sr = torch.clamp(A_dev @ B_blk_dev.T, min=0.0)      # (m, block_size)
        Sr = dense_Sr.to_sparse_coo().coalesce()          # keep only >0 entries
        # 2) compute the first term -Sr
        sumSr_devs[dev_idx] += Sr.sum()
        ssqSr_devs[dev_idx] += (Sr * Sr).sum()
        dB_block = torch.sparse.mm(-Sr.transpose(0,1), pinvA_trans_dev) # (block_size, r)
        dB[start:end] = dB_block.to(dB.device) ##OLD

    #compute the second term, correction (S - ABt + Sr)[S>0]
    pred_vals = torch.sum(
        A_cpu[S_idx[0], :] * B_cpu[S_idx[1], :], 
        dim=1
    )
    Sr_vals = torch.clamp(pred_vals, min=0.0)
    correction_vals = S_vals - pred_vals + Sr_vals
    dZ_pos = torch.sparse_coo_tensor(
        S_idx,
        correction_vals,
        (m, n),
        device=device_cpu,
        dtype=S_vals.dtype
    ).coalesce()

    # final update
    dB += torch.sparse.mm(dZ_pos.transpose(0,1), pinvA_trans_cpu)
    nextB = B_cpu + dB
    # ——————————————
    # 8) metrics
    # reduce device accumulators once, avoid .item()
    sumSr = sum(t.detach().to(device_cpu) for t in sumSr_devs)
    ssqSr = sum(t.detach().to(device_cpu) for t in ssqSr_devs)

    num_jacc = torch.sum(torch.minimum(S_vals, Sr_vals)) #the second term is 0 as both S_vals, Sr_vals nonnegative
    denom_jacc = torch.sum(S_vals) + sumSr - num_jacc
    jacc = 1.0 - num_jacc / denom_jacc
    # ‖S‖ on its nonzero entries:
    S_norm = torch.norm(S_vals).item()
    ssqS = S_norm**2
    ssqe = ssqS + ssqSr - 2*torch.sum(S_vals * Sr_vals)
    rmse  = math.sqrt(ssqe) / (S_norm + 1e-16)
    return nextB, rmse, jacc


def refactor(A: torch.Tensor, B: torch.Tensor):
    '''
    refactor Ar Br.T = A B.T where Ar, Br share the same singular values
    TODO: call it "canonicalize"
    '''
    # QR decompositions (reduced mode)
    Qa, Ra = torch.linalg.qr(A, mode='reduced')
    Qb, Rb = torch.linalg.qr(B, mode='reduced')

    # SVD of Ra * Rb^T
    U, S, Vh = torch.linalg.svd(Ra @ Rb.T, full_matrices=False)

    sqrtS = torch.diag(torch.sqrt(S))

    Ar = Qa @ U @ sqrtS
    Br = Qb @ Vh.T @ sqrtS

    return Ar, Br


###
#DEPRECIATED: OLD REFERENCE CODE
###

def least_squares_update(
    S_idx: torch.LongTensor, #shape (2, nmz)
    S_vals: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    num_blocks: int,
    cols_per_block: int, 
    multi_gpu: bool = True,
) -> tuple[torch.Tensor, float, float]:
    """
    [DEPRECIATED - Default to least_squares_fast]
    PyTorch version of the least_squares_update.
    Args:
      S_index   : index of edge pairs
      S_values  : nonnegative entries
      A         : dense tensor, shape (m, r)
      B         : dense tensor, shape (n, r)
      cache_MB  : how many megabytes to dedicate per block
      multi_gpu : if True, will round-robin blocks across all available GPUs
    Returns:
      nextB : updated B (n, r)
      rmse  : float
      jacc  : float (1 – Jaccard similarity)
    """

    # ——————————————
    # 1) block setup (matches MATLAB)
    m = A.shape[0]
    n = B.shape[0]

    # ——————————————
    # 2) pull out sparse‐COO indices & values -- TODO: clean up m code? instantiate S1 per block?
    # S1 = torch.sparse_coo_tensor(
    #             S_idx, 
    #             torch.ones_like(S_vals), 
    #             (m,n)
    #         ).coalesce()

    # ——————————————
    # 3) keep CPU copies of A, B, and compute pseudoinverse on CPU
    device_cpu = A.device
    A_cpu = A.to(device_cpu)
    B_cpu = B.to(device_cpu)
    pinvA_trans_cpu = torch.linalg.solve(A_cpu.T @ A_cpu, A_cpu.T).T   # shape (m,r)

    # ——————————————
    # 4) prepare for multi‐GPU and preload factors to device
    if multi_gpu and torch.cuda.device_count() > 1:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        dev = torch.device("cuda:0") if torch.cuda.is_available() else device_cpu
        devices = [dev]
    A_devs      = [A_cpu.to(dev) for dev in devices]
    pinvA_trans_devs  = [pinvA_trans_cpu.to(dev) for dev in devices]
    B_devs      = [B_cpu.to(dev) for dev in devices]

    # ——————————————
    # 5) accumulators
    ssqe = 0.0
    numJ = 0.0
    denJ = 0.0
    dB   = torch.zeros_like(B_cpu)
    # 6) loop over blocks
    for b in range(num_blocks):
        start = b * cols_per_block
        end = min((b+1) * cols_per_block, n)
        # device = devices[b % len(devices)]
        # A_dev = A_cpu.to(device)
        # pinvA_trans_dev = pinvA_trans_cpu.to(device)
        # B_blk_dev = B_cpu[start:end].to(device)   # (block_size, r)
        dev_idx = b % len(devices)
        device = f"cuda:{dev_idx}"
        A_dev = A_devs[dev_idx]
        pinvA_trans_dev = pinvA_trans_devs[dev_idx]
        B_blk_dev = B_devs[dev_idx][start:end] #(block_size, r)

        # 1) reconstruct full block and immediately sparsify
        dense_Sr = torch.clamp(A_dev @ B_blk_dev.T, min=0.0)      # (m, block_size)
        Sr = dense_Sr.to_sparse_coo().coalesce()          # keep only >0 entries

        # 2) slice out the matching column‐block of S from (S_idx, S_vals)
        mask = (S_idx[1] >= start) & (S_idx[1] < end)
        block_idx = S_idx[:, mask].clone()       # shape (2, nnz_block)
        block_idx[1] -= start                     # shift columns into [0 … block_size)
        block_vals  = S_vals[mask].to(device)     # the observed values in this block
        Sb = torch.sparse_coo_tensor(
                    block_idx,
                    block_vals,
                    (m, end - start),                      # shape = (m, block_size)
                    device=device
                ).coalesce()

        # 3) metrics on the sparse difference (note: cannot do torch.min/max on CUDA sparse tensors -> moved to cpu)
        diff = Sr - Sb                                         
        ssqe += torch.norm(diff.coalesce().values()).pow(2).item()
        sp_elem_min, sp_elem_max = sparse_min_max(dense_Sr.to(device), block_vals, block_idx)
        numJ += sp_elem_min.item() #torch.minimum(Sr.cpu(), Sb.cpu()).values().sum().item()
        denJ += sp_elem_max.item() #torch.maximum(Sr.cpu(), Sb.cpu()).values().sum().item()
        del dense_Sr
        torch.cuda.empty_cache()

        # 4) one‐liner sparse residual: Sr·S1 - Sr 
        S1b = torch.sparse_coo_tensor(
                    block_idx,
                    torch.ones_like(block_vals),
                    (m, end - start),                      # shape = (m, block_size)
                    device=device
                ).coalesce()    
        # contribution from dZ where S = 0 
        # we can use Sr = ReLU(AB) instead of recomputing AB, because
        # (i) S=0, AB > 0, then Sr = AB; (ii) S=0, AB < 0, then Sr = 0 which is desired (no update)    
        dZ_neg = -Sr + Sr * S1b                           # sparse COO
        # 5) update block of dB
        # pinvA_trans: (m, r) dense, dZ_neg: (m, block_size) sparse
        dB_block = torch.sparse.mm(dZ_neg.transpose(0,1), pinvA_trans_dev) # (block_size, r)
        dB[start:end] = dB_block.to(dB.device)

    # ——————————————
    # 7) contribution from dZ where S > 0
    #    dZ_pos(i,j) = S(i,j) − <A[i,:], B[j,:]>
    pred_vals = torch.sum(
        A_cpu[S_idx[0], :] * B_cpu[S_idx[1], :], 
        dim=1
    )
    dZ_pos_vals = S_vals - pred_vals
    dZ_pos = torch.sparse_coo_tensor(
        S_idx,
        dZ_pos_vals,
        (m, n),
        device=device_cpu,
        dtype=dZ_pos_vals.dtype
    ).coalesce()

    # final update on cpu
    dB += torch.sparse.mm(dZ_pos.transpose(0,1), pinvA_trans_cpu)

    # ——————————————
    # 8) metrics
    jacc = 1.0 - numJ / denJ
    # ‖S‖ on its nonzero entries:
    S_norm = torch.norm(S_vals).item()
    rmse  = math.sqrt(ssqe) / (S_norm + 1e-16)
    nextB = B_cpu + dB
    return nextB, rmse, jacc

def sparse_min_max(S_pred, S_value, S_index):
    '''
    [DEPRECIATED - used only in least_squares_update()]
    S_pred: dense matrix (reconstructed matrix)
    '''
    idx1 = S_index[0]
    idx2 = S_index[1]
    ##compute S>0 errors
    num_nonzero = torch.minimum(S_pred[idx1, idx2], S_value).sum()
    denom_nonzero = torch.maximum(S_pred[idx1, idx2], S_value).sum()
    ##compute S=0 errors
    S_pred_zeroed = S_pred.clone()  # Avoid modifying input
    S_pred_zeroed[idx1, idx2] = 0
    denom_zero = S_pred_zeroed.sum()
    return num_nonzero, denom_nonzero+denom_zero
