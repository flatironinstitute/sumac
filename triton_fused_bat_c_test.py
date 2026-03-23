import torch
import triton
import triton.language as tl
import os
import argparse
os.environ["MPLBACKEND"] = "Agg"

from _sumac.tuning import *
import relu_bat_c_fused_cuda as kernel_ext 

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
        triton.Config({"BM": 16*m, "BN": 16*n, "BK": 16*k}, num_warps=w, num_stages=s)
        for m in [1, 2, 4]
        for n in [1, 2, 4]
        for k in [1]
        for w in [1, 2, 4]
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
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
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

        # a_ptrs = A_ptr + d[:, None] * stride_ad + n[None, :] * stride_an
        # a = tl.load(a_ptrs, mask=(n[None, :] < N), other=0.0).to(tl.float32)
        # if ROUND_TF32:
        #     a = round_f32_to_tf32(a)

        # c_ptrs = C_ptr + n[:, None] * stride_an + d[None, :] * stride_ad
        # c = tl.load(c_ptrs, mask=(n[:, None] < N), other=0.0).to(tl.float32) 
        # if ROUND_TF32:
        #     c = round_f32_to_tf32(c)

        for k0 in tl.static_range(0, BN, BK):
            nk = n0 + k0 + tl.arange(0, BK)  # (BK,)
            a_ptrs = A_ptr + d[:, None] * stride_ad + nk[None, :] * stride_an
            a = tl.load(a_ptrs,mask=(nk[None,:] < N), other=0.0).to(tl.float32)
            c_ptrs = C_ptr + nk[:, None] * stride_an + d[None, :] * stride_ad
            c  = tl.load(c_ptrs, mask=(nk[:,None] < N), other=0.0).to(tl.float32)
            s  = tl.dot(b, a, input_precision=DOT_PREC)       # (BM, BK)
            s  = tl.maximum(s,0.0)
            y += tl.dot(s, c, input_precision=DOT_PREC)        # (BM, D)


        # s = tl.dot(b, a,input_precision=DOT_PREC)
        # s = tl.maximum(s, 0.0)

        # y += tl.dot(s, c, input_precision=DOT_PREC)

    y_ptrs = Y_ptr + m[:, None] * stride_ym + d[None, :] * stride_yd
    tl.store(y_ptrs, y, mask=(m[:, None] < M))


def relu_bat_c_fused(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
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
        ROUND_TF32=False, DOT_PREC="ieee"
    )
    return Y


@torch.compile(mode='max-autotune-no-cudagraphs')
def torch_impl(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return torch.relu(B @ A.T) @ C

def ms_to_tflops(M, N, D, ms):
        flops = M * N * (D + D - 1) + M * D * (N + N - 1)
    
        s = ms * 1e-3
        return flops / s / 1e12

def flop_per_byte_fused(M, N, D, wordsize):
    flops = M * N * (D + D - 1) + M * D * (N + N - 1)

    elems_read = M * D + N * D + N * D
    elems_written = M * D
    
    return flops/((elems_read + elems_written) * wordsize) 

def flop_per_byte_2kernels(M, N, D, wordsize):
    flops = M * N * (D + D - 1) + M * D * (N + N - 1)
    
    elems_read_kernel1 = M * D + N * D
    elems_written_kernel1 = M * N
    
    elems_read_kernel2 = M * N + N * D
    elems_written_kernel2 = M * D

    return flops/((elems_read_kernel1 + elems_read_kernel2 + elems_written_kernel1 + elems_written_kernel2) * wordsize)

def perf_roofline(FLOP_per_Byte, BW_GBs, peak_TFLOP):
        return min(peak_TFLOP,(BW_GBs * FLOP_per_Byte)/1e3) #/1e3 to get to TFLOP/s



def relu_bat_c_cuda_launcher():
    @autotune_cuda_kernel(
        configs={
            "BM": [32, 64, 128, 256],
            "BK": [16, 32, 64],
            "num_ms": [2, 4, 6],
        },
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_constraints,
        validate_fn=relu_bat_c_validate,
        cache_path="relu_bat_c_autotune.json",
        n_trials=1000,
        warmup=5,
        rep=50,
        sampler=optuna.samplers.GridSampler(search_space={"BM": [32, 64, 128, 256],
            "BK": [16, 32, 64],
            "num_ms": [2, 4, 6]})
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


@torch.no_grad()
def bench_one(M: int, N: int, D: int, dtype=torch.float32, device="cuda", wm_iters=25, iters=200):
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)
    C = torch.randn((N, D), device=device, dtype=dtype)
    
    ref = torch_impl(A, B, C)
    relu_bat_c_tuned = relu_bat_c_cuda_launcher()
    # out_triton = relu_bat_c_fused(A, B, C)

    # err_triton = (out_triton - ref).abs().max().item()

    out_cuda = relu_bat_c_tuned(A, B, C)

    err_cuda = (out_cuda - ref).abs().max().item()


    print(f"[correctness] M={M} N={N} D={D} dtype={dtype}")
    # print(f"  triton fused (1D)           max_abs_err vs torch: {err_triton:.6e}")
    print(f"  cuda fused (1D)             max_abs_err vs torch: {err_cuda:.6e}")

    def torch_run():
        return torch_impl(A, B, C)

    # def triton_run():
    #     return relu_bat_c_fused(A, B, C)
    
    def cuda_run():
        return relu_bat_c_tuned(A, B, C)

    torch.cuda.synchronize()

    t_torch   = triton.testing.do_bench(torch_run,          warmup=wm_iters, rep=iters)
    # t_triton  = triton.testing.do_bench(triton_run,         warmup=wm_iters, rep=iters)
    t_cuda    = triton.testing.do_bench(cuda_run,           warmup=wm_iters, rep=iters)

    return {
        "M": M, "N": N, "D": D, "dtype": str(dtype).replace("torch.", ""),

        # "max_abs_err_triton":  float(err_triton),
        "max_abs_err_cuda":    float(err_cuda),
        "torch_ms":   float(t_torch),
        # "triton_ms":  float(t_triton),
        "cuda_ms":    float(t_cuda),
        "torch_TFLOPs":   float(ms_to_tflops(M, N, D, t_torch)),
        # "triton_TFLOPs":  float(ms_to_tflops(M, N, D, t_triton)),
        "cuda_TFLOPs":    float(ms_to_tflops(M, N, D, t_cuda)),
        # "speedup_triton":  float(t_torch / t_triton),
        "speedup_cuda": float(t_torch/t_cuda),
    }



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Triton Kernel benchmark")
    parser.add_argument("--warmup-iters", type=int, default=25,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=200,
                        help="Number of benchmark iterations")
    args = parser.parse_args()

    torch.manual_seed(0)
    assert torch.cuda.is_available()

    r = bench_one(M=145408, N=1408, D=4, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
    print(r)
    r = bench_one(M=145408, N=1408, D=8, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
    print(r)

    r = bench_one(M=145408, N=1408, D=13, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
    print(r)
    r = bench_one(M=145408, N=1408, D=16, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
    print(r)
    r = bench_one(M=145408, N=1408, D=17, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
    print(r)
    r = bench_one(M=145408, N=1408, D=18, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
    print(r)
    r = bench_one(M=145408, N=1408, D=19, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
    #r = bench_one(M=145408, N=14540, D=16, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
    print(r)

    


