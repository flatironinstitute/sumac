import torch
from relu_batc_jit.api import relu_bat_c_fused
from _sumac.tuning import *
import triton
import triton.language as tl

@triton.jit
def round_f32_to_tf32(x):
    # Round-to-nearest-even to TF32 precision by rounding FP32 mantissa
    xb = tl.cast(x, tl.uint32, bitcast=True)
    lsb = (xb >> 13) & 1       #least significant bit that won't be discarded
    bias = 0x00001000 + lsb    #rounding
    xb = (xb + bias) & 0xFFFFE000       #xb + rounding bit, zero lower 13 bits  
    return tl.cast(xb, tl.float32, bitcast=True) #bitcast back to fp32

@triton.autotune(
configs=[
        triton.Config({"BM": m, "BN": n}, num_warps=w, num_stages=s)
        for m in [32, 64, 128]
        for n in [32, 64, 128]
        for w in [1, 2, 4, 8]
        for s in [1, 2]
    ],
    key=["M", "N", "D"],
    cache_results=True
)
@triton.jit
def relu_bat_c_fused_kernel(
    A_ptr, B_ptr, C_ptr, Y_ptr,
    N, M, D: tl.constexpr,
    stride_an: tl.constexpr, stride_ad: tl.constexpr,
    stride_bm: tl.constexpr, stride_bd: tl.constexpr,
    stride_ym: tl.constexpr, stride_yd: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
    ROUND_TF32: tl.constexpr, DOT_PREC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m * BM + tl.arange(0, BM)
    d = tl.arange(0, D)

    b_ptrs = B_ptr + m[:, None] * stride_bm + d[None, :] * stride_bd
    b = tl.load(b_ptrs, mask=(m[:, None] < M), other=0.0).to(tl.float32)
    if ROUND_TF32:
        b = round_f32_to_tf32(b)

    y = tl.zeros((BM, D), dtype=tl.float32)

    for n0 in tl.range(0, N, BN):
        n = n0 + tl.arange(0, BN)

        a_ptrs = A_ptr + n[:, None] * stride_an + d[None, :] * stride_ad
        a = tl.load(a_ptrs, mask=(n[:, None] < N), other=0.0).to(tl.float32)
        c_ptrs = C_ptr + n[:, None] * stride_an + d[None, :] * stride_ad
        c  = tl.load(c_ptrs, mask=(n[:,None] < N), other=0.0).to(tl.float32)
        if ROUND_TF32:
            a = round_f32_to_tf32(a)

        s = tl.dot(b, tl.trans(a),input_precision=DOT_PREC)
        s = tl.maximum(s, 0.0)

        y += tl.dot(s, c, input_precision=DOT_PREC)

    y_ptrs = Y_ptr + m[:, None] * stride_ym + d[None, :] * stride_yd
    tl.store(y_ptrs, y, mask=(m[:, None] < M))


def relu_bat_c_fused_triton(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    if not (A.is_cuda and B.is_cuda and C.is_cuda):
        raise ValueError("A and B and C must be CUDA tensors.")
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        raise ValueError("A and B and C must be 2D.")
    if A.shape[1] != B.shape[1]:
        raise ValueError("Feature dims must match.")
    if A.shape[0] != C.shape[0]:
        raise ValueError("A and C must have same dims")
    if A.shape[1] != C.shape[1]:
        raise ValueError("A and C must have same dims")
    

    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()

    N, D = A.shape
    M, _ = B.shape
    Y = torch.empty((M, D), device=A.device, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(M, META["BM"]),)
    relu_bat_c_fused_kernel[grid](
        A, B, C, Y,
        N=N, M=M, D=D,
        stride_an=A.stride(0), stride_ad=A.stride(1),
        stride_bm=B.stride(0), stride_bd=B.stride(1),
        stride_ym=Y.stride(0), stride_yd=Y.stride(1),
        ROUND_TF32=True, DOT_PREC="tf32"
    )
    return Y

def relu_bat_c_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BK: int,
    num_ms: int,
) -> bool:
    if A.shape[1] >= 32 and num_ms > 2:
        return False

    if A.shape[1] >= 64 and num_ms > 1:  
        return False  
    
    props = torch.cuda.get_device_properties(torch.cuda.current_device)

    if props.shared_memory_per_block < 8 * BK * A.shape[1]:
        return False

    return True

@torch.compile(mode='max-autotune-no-cudagraphs')
def relu_bat_c_fallback(A: torch.Tensor,
                        B: torch.Tensor,
                        C: torch.Tensor):
    return torch.relu(B @ A.T) @ C

def relu_bat_c_cuda_launcher():
    tune_config = {
        "BM": [32, 64, 128],
        "BK": [16, 32, 64],
        "num_ms": [1, 2, 4, 6],
    }

    @autotune_cuda_kernel(
        configs=tune_config,
        fallback_fn=relu_bat_c_fallback,
        constraint_fn=relu_bat_c_constraints,
        key_fn=relu_bat_c_key,
        cache_path="relu_bat_c_jit_autotune.json",
        n_trials=1000,
        warmup=1,
        rep=5,
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
    unbias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    stepM_blk = relu_bat_c_tuned(Ar_dev, B_blk_dev, pinvAt_dev)
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
    row_indices: torch.Tensor,
    B: torch.Tensor,
    dB: torch.Tensor,
    momentum,
    unbias,
    m_fixed: int,
):
    
    dev = B.device

    m_batch = row_indices.shape[0]

    Ar_dev = Factor_fixed[row_indices, :]
    GramA = Ar_dev.T @ Ar_dev
    pinvAt_dev = torch.linalg.solve(GramA, Ar_dev.T).T

    blk_idx = Sr_idx_raw
    blk_vals = Sr_vals

    row_idx_dev = row_indices
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
    idx_raw = S_idx_full[:, edge_idx]
    val_raw = S_val_full[edge_idx]

    params = relu_bat_c_tuned.resolve_params(Factor_fixed[row_indices, :], Factor_update, Factor_fixed[row_indices, :])
    #need to resolve params outside of the compiled region

    nextF, dF = batch_update_single_gpu(
        Sr_idx_raw=idx_raw,
        Sr_vals=val_raw,
        Factor_fixed=Factor_fixed,
        row_indices=row_indices,
        B=Factor_update,
        dB=dFactor,
        momentum=momentum,
        unbias=unbias,
        m_fixed=m_fixed,
    )
    
    return nextF, dF