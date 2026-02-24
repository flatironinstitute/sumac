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

#This cannot issue vectorized loads for some reason... LSU pipeline pressure kills perf
@cuda.jit
def relu_bat_a_fused_fp32_numba(A, B, Y, N):
    
    BM = cuda.blockDim.x
    BK = 32  

    
    As = cuda.shared.array(shape=(32, D), dtype=float32)  # BK=32
    m0 = cuda.blockIdx.x * BM
    m = cuda.blockIdx.x * BM + cuda.threadIdx.x
    M = B.shape[0]
    Ys = cuda.shared.array((8, D, 32), float32)
    
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = b7 = float32(0.0)
    b8 = b9 = b10 = b11 = b12 = b13 = b14 = b15 = float32(0.0)
    if m < M:
        b0  = B[m, 0];  b1  = B[m, 1];  b2  = B[m, 2];  b3  = B[m, 3]
        b4  = B[m, 4];  b5  = B[m, 5];  b6  = B[m, 6];  b7  = B[m, 7]
        b8  = B[m, 8];  b9  = B[m, 9];  b10 = B[m,10];  b11 = B[m,11]
        b12 = B[m,12];  b13 = B[m,13];  b14 = B[m,14];  b15 = B[m,15]

    
    y0 = y1 = y2 = y3 = y4 = y5 = y6 = y7 = float32(0.0)
    y8 = y9 = y10 = y11 = y12 = y13 = y14 = y15 = float32(0.0)

    tid = cuda.threadIdx.x
    warp = tid >> 5                         
    lane = tid & 31   

    n0 = 0
    while n0 < N:
        idx = tid
        while idx < BK * D:
            k = idx // D
            d = idx - k * D
            n = n0 + k
            As[k, d] = A[n, d] if n < N else 0.0
            idx += BM

        cuda.syncthreads()

        k = 0
        while k < BK:
            acc = float32(0.0)
            acc = cuda.fma(b0,  As[k, 0],  acc); acc = cuda.fma(b1,  As[k, 1],  acc)
            acc = cuda.fma(b2,  As[k, 2],  acc); acc = cuda.fma(b3,  As[k, 3],  acc)
            acc = cuda.fma(b4,  As[k, 4],  acc); acc = cuda.fma(b5,  As[k, 5],  acc)
            acc = cuda.fma(b6,  As[k, 6],  acc); acc = cuda.fma(b7,  As[k, 7],  acc)
            acc = cuda.fma(b8,  As[k, 8],  acc); acc = cuda.fma(b9,  As[k, 9],  acc)
            acc = cuda.fma(b10, As[k,10],  acc); acc = cuda.fma(b11, As[k,11],  acc)
            acc = cuda.fma(b12, As[k,12],  acc); acc = cuda.fma(b13, As[k,13],  acc)
            acc = cuda.fma(b14, As[k,14],  acc); acc = cuda.fma(b15, As[k,15],  acc)

            # ReLU
            if acc < float32(0.0):
                acc = float32(0.0)

            y0  = cuda.fma(acc, As[k, 0],  y0);   y1  = cuda.fma(acc, As[k, 1],  y1)
            y2  = cuda.fma(acc, As[k, 2],  y2);   y3  = cuda.fma(acc, As[k, 3],  y3)
            y4  = cuda.fma(acc, As[k, 4],  y4);   y5  = cuda.fma(acc, As[k, 5],  y5)
            y6  = cuda.fma(acc, As[k, 6],  y6);   y7  = cuda.fma(acc, As[k, 7],  y7)
            y8  = cuda.fma(acc, As[k, 8],  y8);   y9  = cuda.fma(acc, As[k, 9],  y9)
            y10 = cuda.fma(acc, As[k,10],  y10);  y11 = cuda.fma(acc, As[k,11],  y11)
            y12 = cuda.fma(acc, As[k,12],  y12);  y13 = cuda.fma(acc, As[k,13],  y13)
            y14 = cuda.fma(acc, As[k,14],  y14);  y15 = cuda.fma(acc, As[k,15],  y15)

            k += 1

        cuda.syncthreads()
        n0 += BK

        Ys[warp, 0, lane]  = y0;  Ys[warp, 1, lane]  = y1;  Ys[warp, 2, lane]  = y2;  Ys[warp,  3, lane]  = y3
        Ys[warp, 4, lane]  = y4;  Ys[warp, 5, lane]  = y5;  Ys[warp, 6, lane]  = y6;  Ys[warp,  7, lane]  = y7
        Ys[warp, 8, lane]  = y8;  Ys[warp, 9, lane]  = y9;  Ys[warp,10, lane]  = y10; Ys[warp, 11, lane]  = y11
        Ys[warp,12, lane]  = y12; Ys[warp,13, lane]  = y13; Ys[warp,14, lane]  = y14; Ys[warp, 15, lane]  = y15

        cuda.syncthreads()
    
    base_row = m0 + warp * 32
    if lane < 16:
        d = lane
        for r in range(16):
            mr = base_row + r
            if mr < M:
                Y[mr, d] = Ys[warp, d, r]
    else:
        d = lane - 16
        for r in range(16, 32):
            mr = base_row + r
            if mr < M:
                Y[mr, d] = Ys[warp, d, r]

def relu_bat_a_numba(A_torch, B_torch, BM=128):
    assert A_torch.is_cuda and B_torch.is_cuda
    assert A_torch.dtype == torch.float32 and B_torch.dtype == torch.float32
    assert A_torch.is_contiguous() and B_torch.is_contiguous()
    assert A_torch.shape[1] == 16 and B_torch.shape[1] == 16

    A = cp.from_dlpack(torch.utils.dlpack.to_dlpack(A_torch))
    B = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B_torch))
    Y = cp.empty((B.shape[0], 16), dtype=cp.float32)

    grid = ((B.shape[0] + BM - 1) // BM,)
    block = (BM,)
    relu_bat_a_fused_fp32_numba[grid, block](A, B, Y, np.int32(A.shape[0]))
    return torch.utils.dlpack.from_dlpack(Y.toDlpack())

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
        for m in [2, 4]
        for n in [2, 4]
        for w in [1]
        for s in [1]
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
def bench_one(M: int, N: int, D: int, dtype=torch.float32, device="cuda", wm_iters=25, iters=200):
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)

    ref = torch_impl(A, B)

    out_triton = relu_bat_a_fused_triton(A, B)
    out_cuda = relu_bat_a_fused_cuda(A,B)
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
        return relu_bat_a_fused_cuda(A, B)

    torch.cuda.synchronize()

    t_torch   = triton.testing.do_bench(torch_run,          warmup=wm_iters, rep=iters)
    t_triton  = triton.testing.do_bench(triton_run,         warmup=wm_iters, rep=iters)
    t_cuda = triton.testing.do_bench(cuda_run,  warmup=wm_iters, rep=iters)

    return {
        "M": M, "N": N, "D": D, "dtype": str(dtype).replace("torch.", ""),

        "max_abs_err_triton":  float(err_triton),

        "torch_ms":   float(t_torch),
        "triton_ms":  float(t_triton),
        "cuda_ms": float(t_cuda),
        "torch_TFLOPs":   float(ms_to_tflops(M, N, D, t_torch)),
        "triton_TFLOPs":  float(ms_to_tflops(M, N, D, t_triton)),
        "cuda_TFLOPs":   float(ms_to_tflops(M,N,D, t_cuda)),
        "speedup_triton":  float(t_torch / t_triton),
        "speedup_cuda":   float(t_torch/t_cuda),
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
def benchmark(M, N, D, dtype, provider, wm_iters=25, iters=200):
   
    device = "cuda"
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)

    if provider == "torch":
        fn = lambda: torch_impl(A, B)
    elif provider == "triton":
        fn = lambda: relu_bat_a_fused_triton(A, B)
    elif provider == "cuda":
        fn = lambda: relu_bat_a_fused_cuda(A, B)
    else:
        return perf_roofline(FLOP_per_Byte=flop_per_byte_fused(M, N, D, 4), BW_GBs=960, peak_TFLOP=91.1)
    
        
    ms = triton.testing.do_bench(fn, warmup=wm_iters, rep=iters)
    return ms_to_tflops(M, N, D, ms)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Triton Kernel benchmark")
    parser.add_argument("--warmup-iters", type=int, default=25,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=200,
                        help="Number of benchmark iterations")
    parser.add_argument("--sweep", action='store_true', default=False)

    args = parser.parse_args()

    torch.manual_seed(0)
    assert torch.cuda.is_available()

    r = bench_one(M=139255, N=1392, D=16, dtype=torch.float32, wm_iters=args.warmup_iters, iters=args.iters)
    print(r)

    if(args.sweep):
        df = benchmark.run(print_data=False, show_plots=True, save_path="triton_bench", wm_iters=args.warmup_iters, iters=args.iters, return_df=True)

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


# Fused kernel traffic:
# Read A, Read B, write C
# A in R^{m x d}, B in R^{n x d}, C in R^{n x d} = (m*d+n*d+n*d)*sizeof(word) min DRAM traffic

# Two matmul kernels traffic:
# Read A, Read B, Write BA.T, Read BA.T, Read A, Write C
# (m*d + n*d + m*n)*sizeof(word) + (m*n+m*d+n*d)*sizeof(word) min DRAM traffic 

