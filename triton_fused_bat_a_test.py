import torch
import triton
import triton.language as tl
import os

from relu_bat_a_fused_cuda import *

import argparse
os.environ["MPLBACKEND"] = "Agg"
import cupy as cp

import numpy as np
from numba import cuda, float32
import pandas as pd
pd.set_option('display.max_columns', None)

D = 16

@triton.jit
def round_f32_to_tf32_rn(x):
    # Round-to-nearest-even to TF32 precision by rounding FP32 mantissa
    xb = tl.cast(x, tl.uint32, bitcast=True)
    lsb = (xb >> 13) & 1       #least significant bit that won't be discarded
    bias = 0x00001000 + lsb    #rounding
    xb = (xb + bias) & 0xFFFFE000       #xb + rounding bit, zero lower 13 bits  
    return tl.cast(xb, tl.float32, bitcast=True) #bitcast back to fp32

@triton.autotune(
configs=[
        triton.Config({"BM": 16*m, "BN": 16*n}, num_warps=w, num_stages=s)
        for m in [2, 4, 8]
        for n in [2, 4, 8]
        for w in [1, 2, 4]
        for s in [1, 2, 3]
    ],
    key=["M", "N", "D"],
    cache_results=True
)
@triton.jit
def relu_bat_a_fused_kernel(
    A_ptr, B_ptr, Y_ptr,
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
        b = round_f32_to_tf32_rn(b)

    y = tl.zeros((BM, D), dtype=tl.float32)

    for n0 in tl.range(0, N, BN):
        n = n0 + tl.arange(0, BN)

        a_ptrs = A_ptr + n[:, None] * stride_an + d[None, :] * stride_ad
        a = tl.load(a_ptrs, mask=(n[:, None] < N), other=0.0).to(tl.float32)
        if ROUND_TF32:
            a = round_f32_to_tf32_rn(a)

        # Force IEEE fp32 precision for dot or it will use tf32
        s = tl.dot(b, tl.trans(a),input_precision=DOT_PREC)
        s = tl.maximum(s, 0.0)

        y += tl.dot(s, a, input_precision=DOT_PREC)

    y_ptrs = Y_ptr + m[:, None] * stride_ym + d[None, :] * stride_yd
    tl.store(y_ptrs, y, mask=(m[:, None] < M))


def relu_bat_a_fused_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
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
        ROUND_TF32=False, DOT_PREC="ieee"
    )
    return Y


@torch.compile(mode='max-autotune-no-cudagraphs')
def torch_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.relu(B @ A.T) @ A

def ms_to_tflops(M, N, D, ms):
        flops = M * N * (D + D - 1) + M * D * (N + N - 1)
    
        s = ms * 1e-3
        return flops / s / 1e12

def flop_per_byte_fused(M, N, D, wordsize):
    flops = M * N * (D + D - 1) + M * D * (N + N - 1)

    elems_read = M * D + N * D
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

@torch.no_grad()
def bench_one(M: int, N: int, D: int, BM: int, BK: int, num_stages: int, dtype=torch.float32, device="cuda", wm_iters=25, iters=200):
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)

    ref = torch_impl(A, B)

    out_triton = relu_bat_a_fused_triton(A, B)
    out_cuda = relu_bat_a_fused_cuda(A,B, BM, BK, num_stages)
    err_triton = (out_triton - ref).abs().max().item()
    err_cuda = (out_cuda - ref).abs().max().item()
    print(f"[correctness] M={M} N={N} D={D} dtype={dtype}")
    print(f"  triton fused (1D)           max_abs_err vs torch: {err_triton:.6e}")
    print(f"  cuda fused (1D)             max_abs_err vs torch: {err_cuda:.6e}")    
    def torch_run():
        return torch_impl(A, B)

    def triton_run():
        return relu_bat_a_fused_triton(A, B)
    
    def cuda_run():
        return relu_bat_a_fused_cuda(A, B, BM, BK, num_stages)

    torch.cuda.synchronize()

    t_torch   = triton.testing.do_bench(torch_run,          warmup=wm_iters, rep=iters)
    t_triton  = triton.testing.do_bench(triton_run,         warmup=wm_iters, rep=iters)
    t_cuda = triton.testing.do_bench(cuda_run,  warmup=wm_iters, rep=iters)

    return {
        "BM": BM, "BK": BK, "num_stages": num_stages,
        #"max_abs_err_triton":  float(err_triton),

        "torch_ms":   float(t_torch),
        "triton_ms":  float(t_triton),
        "cuda_ms": float(t_cuda),
        "torch_TFLOPs":   float(ms_to_tflops(M, N, D, t_torch)),
        "triton_TFLOPs":  float(ms_to_tflops(M, N, D, t_triton)),
        "cuda_TFLOPs":   float(ms_to_tflops(M,N,D, t_cuda)),
        "speedup_triton":  float(t_torch / t_triton),
        "speedup_cuda":   float(t_torch/t_cuda),
    }


@torch.no_grad()
def bench_minimal(M: int, N: int, D: int, BM: int, BK: int, num_stages: int, dtype=torch.float32, device="cuda", wm_iters=25, iters=200):
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)
    
    def cuda_run():
        return relu_bat_a_fused_cuda(A, B, BM, BK, num_stages)

    torch.cuda.synchronize()

    t_cuda = triton.testing.do_bench(cuda_run,  warmup=wm_iters, rep=iters)

    return {
        "BM": BM, "BK": BK, "D": D,"num_stages": num_stages,
        "ms": float(t_cuda),
        "TFLOPs":   float(ms_to_tflops(M,N,D, t_cuda)),
    }

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  
        x_vals=[4096, 8192, 16384, 32768, 65536, 131072],
        line_arg="provider",
        line_vals=["torch", "triton", "cuda", "roofline"],
        line_names=["Torch TFLOP/s", "Triton fused TFLOP/s", "CUDA fused TFLOP/s", "Roofline model TFLOP/s"],
        styles=[("tab:blue", "-"), ("tab:orange", "-"), ("tab:green", "-"), ("tab:red","--")],
        ylabel="TFLOP/s",
        plot_name="relu_bat_a_fused",
        args={"N": 1392, "D": 16, "dtype": torch.float32},
    )
)
def benchmark_sweep(M, N, D, dtype, provider, wm_iters=25, iters=200):
   
    device = "cuda"
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)

    if provider == "torch":
        fn = lambda: torch_impl(A, B)
    elif provider == "triton":
        fn = lambda: relu_bat_a_fused_triton(A, B)
    elif provider == "cuda":
        fn = lambda: relu_bat_a_fused_cuda(A, B, 256, 64, 2)
    else:
        return perf_roofline(FLOP_per_Byte=flop_per_byte_fused(M, N, D, 4), BW_GBs=960, peak_TFLOP=91.1)
    
        
    ms = triton.testing.do_bench(fn, warmup=wm_iters, rep=iters)
    return ms_to_tflops(M, N, D, ms)

Ms = [65536*20]
Ds = [4, 6, 8, 15, 16]
x_vals = [(M, D) for M in Ms for D in Ds]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "D"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["torch", "cuda"],
        line_names=["Torch TFLOP/s", "CUDA fused TFLOP/s"],
        styles=[("tab:blue", "-"), ("tab:green", "-")],
        ylabel="TFLOP/s",
        plot_name="relu_bat_a_fused_sweep_MD",
        args={"N": 1392, "dtype": torch.float32},
    )
)
def benchmark_sweep_D(M, D, N, dtype, provider, wm_iters=25, iters=200):
    device = "cuda"
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)

    if provider == "torch":
        fn = lambda: torch_impl(A, B)
        ms = triton.testing.do_bench(fn, warmup=wm_iters, rep=iters)
        tflops = ms_to_tflops(M, N, D, ms)

    elif provider == "cuda":
        if D < 15:
            fn = lambda: relu_bat_a_fused_cuda(A, B, 512, 128, 2)
        else:
            fn = lambda: relu_bat_a_fused_cuda(A, B, 256, 64, 2)
        ms = triton.testing.do_bench(fn, warmup=wm_iters, rep=iters)
        tflops = ms_to_tflops(M, N, D, ms)

    else:  
        tflops = perf_roofline(
            FLOP_per_Byte=flop_per_byte_fused(M, N, D, 4),
            BW_GBs=960,
            peak_TFLOP=91.1,
        )

    return tflops

def correctness_sweep_D(N, dtype):
    device="cuda"
    for M in Ms:
        for D in Ds:
            A = torch.randn((N,D), device=device, dtype=dtype)
            B = torch.randn((M,D), device=device, dtype=dtype)

            ref = torch_impl(A, B)
            out_cuda = relu_bat_a_fused_cuda(A, B, 256, 64, 2)
        
            err_cuda = (out_cuda - ref).abs().max().item()
            print(f" D={D} cuda fused max_abs_err: {err_cuda:.6e}")  


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Triton Kernel benchmark")
    parser.add_argument("--warmup-iters", type=int, default=25,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=200,
                        help="Number of benchmark iterations")
    parser.add_argument("--minimal", action='store_true', default=False, help="Benchmark cuda kernel implementation")
    parser.add_argument("--relative", action='store_true', default=False, help="Benchmark pytorch, triton and cuda kernel implementations")
    parser.add_argument("--sweep_M", action='store_true', default=False, help="Benchmark pytorch, triton and cuda kernel implementations across a range of M")
    parser.add_argument("--sweep_D", action='store_true', default=False, help="Benchmark pytorch, triton and cuda kernel implementations across a range of D")


    args = parser.parse_args()

    torch.manual_seed(0)
    assert torch.cuda.is_available()

    if not args.minimal and not args.relative and not args.sweep_M and not args.sweep_D:
        print("Please pick at least one benchmarking mode. See --help for available options.")

    if args.relative:
        r = bench_one(M=139255, N=1392, D=16, BM=256, BK=64, num_stages=2, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
        print(r)

    if args.minimal:
        for D in [4, 5, 6, 8, 12, 13, 15, 16]:
            for BM in [128, 256, 512]:
                for BK in [64, 128]:
                    for num_stages in [2, 3]:
                        r = bench_minimal(M=139255, N=1392, D=D, BM=BM, BK=BK, num_stages=num_stages, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
                        print(r)

    if args.sweep_M:
        df = benchmark_sweep.run(print_data=False, show_plots=True, save_path="triton_bench", wm_iters=args.warmup_iters, iters=args.iters, return_df=True)

        FPB_fused = lambda M: flop_per_byte_fused(M=M, N=1392, D=16,wordsize=4) 
        FPB_2kernel = lambda M: flop_per_byte_2kernels(M=M, N=1392, D=16, wordsize=4)

        kernel_roofline = lambda Flop_per_Byte: perf_roofline(FLOP_per_Byte=Flop_per_Byte, BW_GBs=960, peak_TFLOP=91.1)

        df["Torch [FLOP/Byte]"] = df["M"].apply(FPB_2kernel)
        df["Triton fused [FLOP/Byte]"] = df["M"].apply(FPB_fused)
        df["Torch DRAM roofline TFLOP/s"] = df["Torch [FLOP/Byte]"].apply(kernel_roofline)

        df["Triton fused DRAM roofline TFLOP/s"] = df["Triton fused [FLOP/Byte]"].apply(kernel_roofline)

        df["Torch % peak achieved"] = df["Torch TFLOP/s"]/df["Torch DRAM roofline TFLOP/s"]*100

        df["Triton fused % peak achieved"] = df["Triton fused TFLOP/s"]/df["Triton fused DRAM roofline TFLOP/s"]*100

        df["CUDA % peak achieved"] = df["CUDA fused TFLOP/s"]/df["Triton fused DRAM roofline TFLOP/s"]*100

        print(df)

    if args.sweep_D:
        df = benchmark_sweep_D.run(print_data=False, show_plots=False, save_path="triton_bench", wm_iters=args.warmup_iters, iters=args.iters, return_df=True)

        print(df)
        #correctness_sweep_D(N=1392, dtype=torch.float32)


# Fused kernel traffic:
# Read A, Read B, write C
# A in R^{m x d}, B in R^{n x d}, C in R^{n x d} = (m*d+n*d+n*d)*sizeof(word) min DRAM traffic

# Two matmul kernels traffic:
# Read A, Read B, Write BA.T, Read BA.T, Read A, Write C
# (m*d + n*d + m*n)*sizeof(word) + (m*n+m*d+n*d)*sizeof(word) min DRAM traffic 

