import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from contextlib import nullcontext
from _sumac.dataset import block_span, RowBlockDataset
import triton
import triton.language as tl

# @triton.autotune(
#     configs=[
#         triton.Config({"BM": 16, "BN": 64}, num_warps=2, num_stages=1),
#         triton.Config({"BM": 16, "BN": 64}, num_warps=2, num_stages=2),
#         triton.Config({"BM": 16, "BN": 64}, num_warps=2, num_stages=4),
#         triton.Config({"BM": 16, "BN": 128}, num_warps=1, num_stages=1),
#         triton.Config({"BM": 16, "BN": 128}, num_warps=1, num_stages=2),
#         triton.Config({"BM": 16, "BN": 128}, num_warps=1, num_stages=4),
#         triton.Config({"BM": 16, "BN": 128}, num_warps=2, num_stages=1),
#         triton.Config({"BM": 16, "BN": 128}, num_warps=2, num_stages=2),
#         triton.Config({"BM": 16, "BN": 128}, num_warps=2, num_stages=4),
#         triton.Config({"BM": 16, "BN": 256}, num_warps=4, num_stages=1),
#         triton.Config({"BM": 16, "BN": 256}, num_warps=4, num_stages=2),
#         triton.Config({"BM": 16, "BN": 256}, num_warps=4, num_stages=4),
#         triton.Config({"BM": 32, "BN": 128}, num_warps=4),
#         triton.Config({"BM": 32, "BN": 256}, num_warps=8)
#     ],
#     key=["M", "N", "D"],
# ) Best config from autotuning runs BM: 16, BN: 128, num_warps: 1, num_ctas: 1, num_stages: 2, maxnreg: None

# Fused Triton kernel for: Y = relu(B @ A.T) @ A
@triton.jit
def relu_bat_a_fused_kernel(
    A_ptr, B_ptr, Y_ptr,
    N, M, D: tl.constexpr,
    stride_an: tl.constexpr, stride_ad: tl.constexpr,
    stride_bm: tl.constexpr, stride_bd: tl.constexpr,
    stride_ym: tl.constexpr, stride_yd: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m * BM + tl.arange(0, BM)
    d = tl.arange(0, D)

    b_ptrs = B_ptr + m[:, None] * stride_bm + d[None, :] * stride_bd
    b = tl.load(b_ptrs, mask=(m[:, None] < M), other=0.0).to(tl.float32)

    y = tl.zeros((BM, D), dtype=tl.float32)

    for n0 in tl.range(0, N, BN):
        n = n0 + tl.arange(0, BN)

        a_ptrs = A_ptr + n[:, None] * stride_an + d[None, :] * stride_ad
        a = tl.load(a_ptrs, mask=(n[:, None] < N), other=0.0).to(tl.float32)

        # Force IEEE fp32 precision for dot
        s = tl.dot(b, tl.trans(a), input_precision="ieee")
        s = tl.maximum(s, 0.0)

        y += tl.dot(s, a, input_precision="ieee")

    y_ptrs = Y_ptr + m[:, None] * stride_ym + d[None, :] * stride_yd
    tl.store(y_ptrs, y, mask=(m[:, None] < M))


def relu_bat_a_fused(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not (A.is_cuda and B.is_cuda):
        raise ValueError("A and B must be CUDA tensors.")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D.")
    if A.shape[1] != B.shape[1]:
        raise ValueError("Feature dims must match.")
    A = A.contiguous()
    B = B.contiguous()

    N, D = A.shape
    M, _ = B.shape
    Y = torch.empty((M, D), device=A.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META["BM"]),)
    relu_bat_a_fused_kernel[grid](
        A, B, Y,
        N=N, M=M, D=D,
        stride_an=A.stride(0), stride_ad=A.stride(1),
        stride_bm=B.stride(0), stride_bd=B.stride(1),
        stride_ym=Y.stride(0), stride_yd=Y.stride(1),
        BM=16, BN=128,      # Hard coding autotuning results here
        num_warps=1,
        num_stages=2,
    )
    return Y

@torch.compile(mode='max-autotune-no-cudagraphs')
def lsq_update_nomatmul(Ar_dev, B_blk_dev, pinvAt_dev, AtAinv, stepM_blk, dB_blk_dev, blk_idx, blk_vals, momentum, unbias):
    #Mt_blk = torch.relu(B_blk_dev @ Ar_dev.T)
    #stepM_blk = -Mt_blk @ pinvAt_dev
    stepM_blk = -stepM_blk @ AtAinv

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

@torch.compile(mode='max-autotune-no-cudagraphs')
def lsq_update(Ar_dev, B_blk_dev, pinvAt_dev, dB_blk_dev, blk_idx, blk_vals, momentum, unbias):
    Mt_blk = torch.relu(B_blk_dev @ Ar_dev.T)
    stepM_blk = -Mt_blk @ pinvAt_dev

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

@torch.compile(mode='max-autotune-no-cudagraphs')
def matmul_relu_fused(Ar_dev, B_blk_dev, pinvAt_dev, dB_blk_dev, momentum, unbias):
    Mt_blk = torch.relu(B_blk_dev @ Ar_dev.T)
    stepM_blk = -Mt_blk @ pinvAt_dev

    lsqB_blk = B_blk_dev + stepM_blk

    dB_blk_new = (lsqB_blk - B_blk_dev) * (1 - momentum) + dB_blk_dev * momentum
    return B_blk_dev + dB_blk_new / unbias, dB_blk_new

# @torch.compile(mode='max-autotune-no-cudagraphs')
# def prepare_invs(Ar_cpu):
#     GramA = Ar_cpu.T @ Ar_cpu
#     AtAinv = torch.linalg.pinv(GramA, hermitian=True)
#     pinvAt_cpu = Ar_cpu @ AtAinv
#     return AtAinv, pinvAt_cpu

def update_factor_salsa(S_idx_full, S_val_full, dataset, block_id, Factor_fixed, Factor_update, dFactor, opts, stepnum, multi_gpu=True, streams=None, map_buffers=None):
    """
    Directly aligns with Matlab: [B,dB] = batch_update(Sr,A(rowsA,:),B,dB,opts,stepnum);
    """
    # 1) Unpack indices
    torch.cuda.nvtx.range_push("Unpack indices")
    m_fixed = Factor_fixed.shape[0]
    n_update = Factor_update.shape[0]
    _, edge_idx, row_indices = dataset[block_id]
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("Slicing S")
    # 2) Slicing S (mimics sparse_slice in Matlab)
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
    # print(f"Factor_fixed is on {Factor_fixed.device}") Var names are confusing, this is already sitting on the GPU
    Ar_cpu = Factor_fixed[row_indices_cpu, :].to(device_cpu, non_blocking=True)
    GramA = Ar_cpu.T @ Ar_cpu #dxd - should probably not even solve this on the GPU
    #pinvAt_cpu = torch.linalg.solve(GramA, Ar_cpu.T).T  # (m_batch, d) = A (A.T A)^(-1)
    AtAinv = torch.linalg.pinv(GramA, hermitian=True)
    pinvAt_cpu = Ar_cpu @ AtAinv
    #AtAinv, pinvAt_cpu = prepare_invs(Ar_cpu) - not worth it
    torch.cuda.nvtx.range_pop()

    momentum = opts.get('exaggerate', 0.7)
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
            AtAinv_dev = AtAinv.to(dev, non_blocking=use_cuda)
            Ar_dev = Ar_cpu.to(dev, non_blocking=use_cuda)
            row_idx_dev = row_indices_cpu.to(dev, non_blocking=use_cuda)
            pinvAt_dev = pinvAt_cpu.to(dev, non_blocking=use_cuda)
            B_blk_dev = B[start:end, :].to(dev, non_blocking=use_cuda)
            dB_blk_dev = dB[start:end, :].to(dev, non_blocking=use_cuda)
            torch.cuda.nvtx.range_pop()
            # 2. Parallel Masking and Re-mapping
            mask = (Sr_idx_raw[1] >= start) & (Sr_idx_raw[1] < end)

            if mask.any():
                torch.cuda.nvtx.range_push("set blk_idx, blk_vals")
                blk_idx = Sr_idx_raw[:, mask].to(dev, non_blocking=use_cuda)
                blk_vals = Sr_vals[mask].to(dev, non_blocking=use_cuda)
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
                tmp = relu_bat_a_fused(Ar_dev, B_blk_dev)
                B_blk_new, dB_blk_new = lsq_update_nomatmul(Ar_dev, B_blk_dev, pinvAt_dev, AtAinv_dev, tmp, dB_blk_dev, blk_idx, blk_vals, momentum, unbias)
                #B_blk_new, dB_blk_new = lsq_update(Ar_dev, B_blk_dev, pinvAt_dev, dB_blk_dev, blk_idx, blk_vals, momentum, unbias)
                torch.cuda.nvtx.range_pop()
            else:
                
                torch.cuda.nvtx.range_push("matmul_relu_fused")
                B_blk_new, dB_blk_new = matmul_relu_fused(Ar_dev, B_blk_dev, pinvAt_dev, dB_blk_dev, momentum, unbias)
                torch.cuda.nvtx.range_pop()

            # 4. Momentum and Back
            next_B_blks[dev_idx] = B_blk_new #.to(devices[dev_idx], non_blocking=use_cuda)
            next_dB_blks[dev_idx] = dB_blk_new #.to(devices[dev_idx], non_blocking=use_cuda) this was causing a circular D2H - H2D copy

    # Synchronize only if using CUDA
    if use_cuda:
        for s in active_streams:
            s.synchronize()

    torch.cuda.nvtx.range_push("nextB setup")
    nextB = torch.empty_like(B)  #B and dB are on the device
    next_dB = torch.empty_like(dB)
    for i, (start, end) in enumerate(block_ranges):
        nextB[start:end] = next_B_blks[i]  #this would trigger H2D and form a D2H H2D loop with the "4. Momentum and Back" block
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
