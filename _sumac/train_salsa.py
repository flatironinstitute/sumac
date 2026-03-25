import torch
from contextlib import nullcontext
from _sumac.dataset import block_span
import relu_bat_c_fused_cuda as kernel_ext
from _sumac.tuning import *



def relu_bat_c_cuda_launcher():
    tuning_config = {
        "BM": [32, 64, 128, 256],
        "BK": [16, 32, 64],
        "num_ms": [2,4,6]
    }
    @autotune_cuda_kernel(
        configs=tuning_config,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_constraints,
        validate_fn=relu_bat_c_validate,
        cache_path="relu_bat_c_autotune.json",
        n_trials=500,
        warmup=5,
        rep=50,
        sampler=optuna.samplers.GridSampler(search_space=tuning_config)
    )
    def relu_bat_c_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BK: int,
        num_ms: int,
    ) -> torch.Tensor:
        return kernel_ext.relu_bat_c_fused_cuda(A, B, C, BM, BK, num_ms)
    return relu_bat_c_cuda


relu_bat_c_tuned = relu_bat_c_cuda_launcher()

@torch.compile(mode='max-autotune-no-cudagraphs', dynamic=True)
def lsq_update_nomatmul(Ar_dev, B_blk_dev, pinvAt_dev, stepM_blk, dB_blk_dev, blk_idx, blk_vals, momentum, unbias):
    stepM_blk = -stepM_blk
   
    Lij_blk = torch.sum(Ar_dev[blk_idx[0], :] * B_blk_dev[blk_idx[1], :], dim=1)
    Mij_blk = torch.relu(Lij_blk)

    Ct_vals = blk_vals - Lij_blk + Mij_blk

    bs = B_blk_dev.shape[0]
    r = B_blk_dev.shape[1]

    stepC_blk = torch.zeros((bs, r), device=B_blk_dev.device, dtype=B_blk_dev.dtype)

    stepC_blk.index_add_(
        0,
        blk_idx[1],
        Ct_vals[:, None] * pinvAt_dev[blk_idx[0], :]
    )
    lsqB_blk = B_blk_dev + stepM_blk + stepC_blk

    dB_blk_new = (lsqB_blk - B_blk_dev) * (1 - momentum) + dB_blk_dev * momentum
    return B_blk_dev + dB_blk_new / unbias, dB_blk_new

@torch.compile(mode='max-autotune-no-cudagraphs', dynamic=True)
def matmul_relu_fused(Ar_dev, B_blk_dev, pinvAt_dev, dB_blk_dev, momentum, unbias):
    Mt_blk = torch.relu(B_blk_dev @ Ar_dev.T)
    stepM_blk = -Mt_blk @ pinvAt_dev

    lsqB_blk = B_blk_dev + stepM_blk

    dB_blk_new = (lsqB_blk - B_blk_dev) * (1 - momentum) + dB_blk_dev * momentum
    return B_blk_dev + dB_blk_new / unbias, dB_blk_new


def update_factor_salsa(S_idx_full, S_val_full, dataset, block_id, Factor_fixed, Factor_update, dFactor, opts, stepnum, multi_gpu=True, streams=None, map_buffers=None):
    """
    Directly aligns with Matlab: [B,dB] = batch_update(Sr,A(rowsA,:),B,dB,opts,stepnum);
    """
    # 1) Unpack indices
    torch.cuda.nvtx.range_push("Unpack indices")
    m_fixed = Factor_fixed.shape[0]
    _, edge_idx, row_indices = dataset[block_id]
    torch.cuda.nvtx.range_pop()

    # 2) Slicing S (mimics sparse_slice in Matlab)
    torch.cuda.nvtx.range_push("Slicing S")
    idx_raw = S_idx_full[:, edge_idx].clone()
    val_raw = S_val_full[edge_idx]
    torch.cuda.nvtx.range_pop() 

    nextF, dF = batch_update_multi_gpu(idx_raw, val_raw, Factor_fixed, row_indices, Factor_update, dFactor, opts, stepnum, m_fixed, streams=streams, map_buffers=map_buffers)
    
    return nextF, dF

def batch_update_multi_gpu(
    Sr_idx_raw, Sr_vals, Factor_fixed, row_indices_cpu, B, dB,
    opts, stepnum, m_fixed, streams=None, map_buffers=None
):
    """
    Asynchronous Multi-GPU version of batch_update_torch.
    Now performs re-mapping in parallel on each GPU.
    CPU-only fallback: runs the same logic on CPU with a single "device".
    Update logic (Reference code): lsq_update_torch()
    """
    m_batch = row_indices_cpu.shape[0]
    n = B.shape[0]
    device_cpu = torch.device("cpu")
    dtype = Factor_fixed.dtype

    use_cuda = torch.cuda.is_available() and (torch.cuda.device_count() > 0)
    num_gpus = torch.cuda.device_count() if use_cuda else 1
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)] if use_cuda else [device_cpu]

    # Use provided persistent streams or create temporary ones (fallback)
    if use_cuda:
        active_streams = streams if streams is not None else [torch.cuda.Stream(device=d) for d in devices]
    else:
        active_streams = [None]  # placeholder

    # Pre-calculate pseudoinverse. We slice Factor_fixed first.
    torch.cuda.nvtx.range_push("pseudo-inverse")
    Ar = Factor_fixed[row_indices_cpu, :]

    GramA = (Ar.T @ Ar)
    pinvAt_cpu = torch.linalg.solve(GramA, Ar.T).T  # (m_batch, d) = A (A.T A)^(-1)
    torch.cuda.nvtx.range_pop()

    momentum = torch.tensor(opts.get('exaggerate', 0.7), device=Factor_fixed.device)
    unbias = 1 - (momentum ** stepnum)
    next_B_blks = [None] * num_gpus
    next_dB_blks = [None] * num_gpus
    block_ranges = []

    for dev_idx in range(num_gpus):
        dev = devices[dev_idx]
        start, end = block_span(dev_idx, n, num_gpus)
        block_ranges.append((start, end))

        ctx = torch.cuda.stream(active_streams[dev_idx]) if use_cuda else nullcontext()
        with ctx:
            # 1. Transfers (async on GPU, normal copies on CPU)
            torch.cuda.nvtx.range_push("H2D")
            Ar_dev = Ar.to(dev, non_blocking=use_cuda)
            row_idx_dev = row_indices_cpu.to(dev, non_blocking=use_cuda)
            pinvAt_dev = pinvAt_cpu.to(dev, non_blocking=use_cuda)
            B_blk_dev = B[start:end, :].to(dev, non_blocking=use_cuda)
            dB_blk_dev = dB[start:end, :].to(dev, non_blocking=use_cuda)
            torch.cuda.nvtx.range_pop()

            
            torch.cuda.nvtx.range_push("set blk_idx, blk_vals")
            blk_idx = Sr_idx_raw.to(dev, non_blocking=use_cuda)
            blk_vals = Sr_vals.to(dev, non_blocking=use_cuda)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("set local_map")
            # RE-MAP locally on this device using persistent buffer if available
            local_map = (map_buffers[dev_idx] if map_buffers is not None else
                            torch.zeros(m_fixed, dtype=torch.long, device=dev))
            local_map.zero_()
            local_map[row_idx_dev] = torch.arange(m_batch, device=dev)
            blk_idx[0] = local_map[blk_idx[0]]
            blk_idx[1] -= start
            torch.cuda.nvtx.range_pop()
            # 3. LSQ Update

            torch.cuda.nvtx.range_push("lsq_update")
            tmp = relu_bat_c_tuned(Ar_dev, B_blk_dev, pinvAt_dev)
            B_blk_new, dB_blk_new = lsq_update_nomatmul(Ar_dev, B_blk_dev, pinvAt_dev, tmp, dB_blk_dev, blk_idx, blk_vals, momentum, unbias)
            torch.cuda.nvtx.range_pop()
            

            # 4. Momentum and Back
            next_B_blks[dev_idx] = B_blk_new 
            next_dB_blks[dev_idx] = dB_blk_new

    # Synchronize only if using CUDA
    if use_cuda:
        for s in active_streams:
            s.synchronize()

    torch.cuda.nvtx.range_push("nextB setup")
    nextB = torch.empty_like(B)  
    next_dB = torch.empty_like(dB)
    for i, (start, end) in enumerate(block_ranges):
        nextB[start:end] = next_B_blks[i]  
        next_dB[start:end] = next_dB_blks[i]
    torch.cuda.nvtx.range_pop()
    return nextB, next_dB


def lsq_update_torch(S, A, B):
    """
    Computes the least squares update for B given the (sliced) sparse matrix S and (sliced) factor A.
    """
    device = A.device
    dtype = A.dtype
    m_batch, n = S.shape

    # CONTRIBUTION FROM ELEMENTS WITH S=0
    GramA = A.T @ A
    pseudoInverseAt = torch.linalg.solve(GramA, A.T).T # (m_batch, d)
    Mt = torch.clamp(B @ A.T, min=0.0)
    stepM = -Mt @ pseudoInverseAt

    # CONTRIBUTION AND CORRECTION FROM ELEMENTS WITH S>0
    indices = S.indices()
    i = indices[0] # row indices in [0, m_batch)
    j = indices[1] # column indices in [0, n)
    Sij = S.values()

    Lij = torch.sum(A[i, :] * B[j, :], dim=1)
    Mij = torch.clamp(Lij, min=0.0)
    
    Ct_vals = Sij - Lij + Mij
    Ct = torch.sparse_coo_tensor(
        torch.stack([j, i]),
        Ct_vals,
        (n, m_batch),
        device=device,
        dtype=dtype
    ).coalesce()
    
    stepC = torch.sparse.mm(Ct, pseudoInverseAt)
    
    return B + stepM + stepC
