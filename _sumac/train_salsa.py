import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from _sumac.dataset import block_span, RowBlockDataset

def update_factor_salsa(S_idx_full, S_val_full, dataset, block_id, Factor_fixed, Factor_update, dFactor, opts, stepnum, multi_gpu=True, streams=None, map_buffers=None):
    """
    Update the full factor (e.g. A) using a subset of the other factor (e.g., B[rows]) and edges (e.g., S_idx_full[:, edge_idx], S_val_full[edge_idx])
    """
    # Unpack indices
    m_fixed = Factor_fixed.shape[0]
    n_update = Factor_update.shape[0]
    _, edge_idx, row_indices = dataset[block_id]
    
    # Slicing S (mimics sparse_slice in Matlab) and make contiguous block (for GPU compute)
    idx_raw = S_idx_full[:, edge_idx].clone()
    Ar_cpu = Factor_fixed[row_indices, :]
    
    if multi_gpu and torch.cuda.device_count() > 1:
        # Parallel Multi-GPU Update (Re-mapping from global index to local ones happens inside)
        nextF, dF = batch_update_multi_gpu(idx_raw, S_val_full[edge_idx], Ar_cpu, row_indices, Factor_update, dFactor, opts, stepnum, m_fixed, streams=streams, map_buffers=map_buffers)
    else:
        # Local Single Device Update
        dev = S_val_full.device
        Sr = torch.sparse_coo_tensor(idx_raw, S_val_full[edge_idx], (m_fixed, n_update), device=dev)
        
        # Optimization: use pre-allocated buffer/stream if available
        buf = map_buffers[0] if map_buffers is not None else None
        strm = streams[0] if streams is not None else None
        
        nextF_dev, dF_dev = batch_update_torch(Sr, Ar_cpu.to(dev), row_indices.to(dev), Factor_update.to(dev), dFactor.to(dev), opts, stepnum, m_fixed, stream=strm, map_buffer=buf)
        nextF, dF = nextF_dev.cpu(), dF_dev.cpu()
    
    return nextF, dF


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


def batch_update_torch(Sr, Ar, row_indices, B, dB, opts, stepnum, m_fixed, stream=None, map_buffer=None):
    """
    Stochastic Alternating Least Squares Algorithm Minibatch Update (CPU / Single-GPU)
    """
    # Local index mapping: map global rows to [0, len(row_indices))
    m_batch = len(row_indices)
    indices = Sr.indices().clone()
    
    # Use pre-allocated buffer if available
    local_row_map = map_buffer if map_buffer is not None else torch.zeros(m_fixed, dtype=torch.long, device=Sr.device)
    # Use context manager to keep the code applicable for both CPU-only and GPU
    ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with ctx:
        if map_buffer is not None:
            local_row_map.zero_()
            
        local_row_map[row_indices] = torch.arange(m_batch, device=Sr.device)
        indices[0] = local_row_map[indices[0]]
        Sr_local = torch.sparse_coo_tensor(indices, Sr.values(), (m_batch, B.shape[0]), device=Sr.device).coalesce()
        #Least squares update
        lsqB = lsq_update_torch(Sr_local, Ar, B)
    momentum = opts.get('exaggerate', 0.7)
    
    dB_new = (lsqB - B) * (1 - momentum) + dB * momentum
    unbias = 1 - (momentum ** stepnum)
    B_new = B + dB_new / unbias
    
    return B_new, dB_new


def batch_update_multi_gpu(Sr_idx_raw, Sr_vals, Ar_cpu, row_indices_cpu, B, dB, opts, stepnum, m_fixed, streams=None, map_buffers=None):
    """
    Asynchronous Multi-GPU version of batch_update_torch. Perform re-mapping (global->local indexes) in parallel on each GPU.
    """
    m_batch, n = Ar_cpu.shape[0], B.shape[0]
    device_cpu = torch.device("cpu")
    dtype = Ar_cpu.dtype
    
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    
    # Use provided persistent streams or create temporary ones (fallback)
    active_streams = streams if streams is not None else [torch.cuda.Stream(device=d) for d in devices]
    
    # Pre-calculate pseudoinverse on CPU once
    GramA = Ar_cpu.T @ Ar_cpu
    pinvAt_cpu = torch.linalg.solve(GramA, Ar_cpu.T).T # (m_batch, d)

    momentum = opts.get('exaggerate', 0.7)
    unbias = 1 - (momentum ** stepnum)
    
    next_B_blks = [None] * num_gpus
    next_dB_blks = [None] * num_gpus
    block_ranges = []

    for dev_idx in range(num_gpus):
        dev = devices[dev_idx]
        start, end = block_span(dev_idx, n, num_gpus)
        block_ranges.append((start, end))
        
        with torch.cuda.stream(active_streams[dev_idx]):
            # 1. Async transfers
            Ar_gpu = Ar_cpu.to(dev, non_blocking=True)
            row_idx_gpu = row_indices_cpu.to(dev, non_blocking=True)
            pinvAt_gpu = pinvAt_cpu.to(dev, non_blocking=True)
            B_blk_gpu = B[start:end, :].to(dev, non_blocking=True)
            dB_blk_gpu = dB[start:end, :].to(dev, non_blocking=True)
            
            # 2. Parallel Masking and Re-mapping
            mask = (Sr_idx_raw[1] >= start) & (Sr_idx_raw[1] < end)
            
            if mask.any():
                blk_idx = Sr_idx_raw[:, mask].to(dev, non_blocking=True)
                blk_vals = Sr_vals[mask].to(dev, non_blocking=True)
                
                # RE-MAP locally on this GPU using persistent buffer if available
                local_map = map_buffers[dev_idx] if map_buffers is not None else torch.zeros(m_fixed, dtype=torch.long, device=dev)
                local_map.zero_()
                local_map[row_idx_gpu] = torch.arange(m_batch, device=dev)
                blk_idx[0] = local_map[blk_idx[0]]
                blk_idx[1] -= start
                
                # 3. LSQ Update
                Mt_blk = torch.clamp(B_blk_gpu @ Ar_gpu.T, min=0.0)
                stepM_blk = -Mt_blk @ pinvAt_gpu
                
                Lij_blk = torch.sum(Ar_gpu[blk_idx[0], :] * B_blk_gpu[blk_idx[1], :], dim=1)
                Mij_blk = torch.clamp(Lij_blk, min=0.0)
                
                Ct_vals = blk_vals - Lij_blk + Mij_blk
                Ct = torch.sparse_coo_tensor(
                    torch.stack([blk_idx[1], blk_idx[0]]),
                    Ct_vals,
                    (end - start, m_batch),
                    device=dev,
                    dtype=dtype
                ).coalesce()
                
                stepC_blk = torch.sparse.mm(Ct, pinvAt_gpu)
            else:
                Mt_blk = torch.clamp(B_blk_gpu @ Ar_gpu.T, min=0.0)
                stepM_blk = -Mt_blk @ pinvAt_gpu
                stepC_blk = 0.0
                
            lsqB_blk = B_blk_gpu + stepM_blk + stepC_blk
            
            # 4. Momentum and Async Back
            dB_blk_new = (lsqB_blk - B_blk_gpu) * (1 - momentum) + dB_blk_gpu * momentum
            B_blk_new = B_blk_gpu + dB_blk_new / unbias
            
            next_B_blks[dev_idx] = B_blk_new.to(device_cpu, non_blocking=True)
            next_dB_blks[dev_idx] = dB_blk_new.to(device_cpu, non_blocking=True)

    for s in streams:
        s.synchronize()
        
    nextB = torch.empty_like(B)
    next_dB = torch.empty_like(dB)
    for i, (start, end) in enumerate(block_ranges):
        nextB[start:end] = next_B_blks[i]
        next_dB[start:end] = next_dB_blks[i]
        
    return nextB, next_dB
