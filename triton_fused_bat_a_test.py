import torch
import triton
import triton.language as tl
import os
os.environ["MPLBACKEND"] = "Agg"

@triton.autotune(
    configs=[
        triton.Config({"BM": 16, "BN": 128}, num_warps=2),
        triton.Config({"BM": 16, "BN": 256}, num_warps=4),
        triton.Config({"BM": 32, "BN": 128}, num_warps=4),
        triton.Config({"BM": 32, "BN": 256}, num_warps=8)
    ],
    key=["M", "N"],
)

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
    )
    return Y


@torch.compile(mode='max-autotune-no-cudagraphs')
def torch_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.relu(B @ A.T) @ A

@torch.no_grad()
def bench_one(M: int, N: int, D: int, dtype=torch.float32, device="cuda", iters=200):
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)

    # Warm up + correctness check 
    ref = torch_impl(A, B)
    out = relu_bat_a_fused(A, B)  
    max_err = (out - ref).abs().max().item()

    def torch_run():
        y = torch_impl(A, B)
        return y

    def triton_run():
        y = relu_bat_a_fused(A, B)
        return y

    torch.cuda.synchronize()

    t_torch = triton.testing.do_bench(torch_run, warmup=25, rep=iters)
    t_triton = triton.testing.do_bench(triton_run, warmup=25, rep=iters)

    # Approx FLOPs: 2 matmuls, ignore ReLU
    flops = 4.0 * M * N * D

    t_torch_s = t_torch * 1e-3
    t_triton_s = t_triton * 1e-3
    torch_tflops = flops / t_torch_s / 1e12
    triton_tflops = flops / t_triton_s / 1e12

    return {
        "M": M, "N": N, "D": D, "dtype": str(dtype).replace("torch.", ""),
        "max_abs_err": max_err,
        "torch_ms": float(t_torch),
        "triton_ms": float(t_triton),
        "torch_TFLOPs": float(torch_tflops),
        "triton_TFLOPs": float(triton_tflops),
        "speedup": float(t_torch / t_triton),
    }


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  
        x_vals=[4096, 8192, 16384, 32768, 65536, 131072],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["Torch", "Triton fused"],
        styles=[("r", "-"), ("b", "-")],
        ylabel="ms",
        plot_name="relu_bat_a_fused",
        args={"N": 4096, "D": 16, "dtype": torch.float32},
    )
)
def benchmark(M, N, D, dtype, provider):
    device = "cuda"
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)

    if provider == "torch":
        fn = lambda: torch_impl(A, B)
    else:
        fn = lambda: relu_bat_a_fused(A, B)

    ms = triton.testing.do_bench(fn, warmup=25, rep=200)
    return ms


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.cuda.is_available()

    r = bench_one(M=65536, N=4096, D=16, dtype=torch.float16)
    print(r)

    benchmark.run(print_data=True, show_plots=True, save_path="triton_bench")




