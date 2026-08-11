import argparse
from dataclasses import asdict
import torch
from torch import Tensor
import triton

from sumac.kernels.tuning import (
    grid_size,
    make_choices,
    kernel_autotune_options,
    AutotuneReluBatReduce,
    relu_bat_reduce_tune_config
)

from sumac.config import AutotuneMode
from sumac.kernels.relu_bat_reduce_jit.api import relu_bat_reduce_fused


def ms_to_tflops(M: int, N: int, D: int, ms: float) -> float:
    num_outputs = M * N
    dot_flops = num_outputs * (D + D - 1)
    sum_flops = (num_outputs - 1) + num_outputs + (num_outputs - 1)
    flops = dot_flops + sum_flops
    return flops / (ms * 1e-3) / 1e12


def _abs_errs(
    got: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
) -> tuple[float, float]:
    err0 = (got[0].squeeze().double() - ref[0].squeeze().double()).abs()
    err1 = (got[1].squeeze().double() - ref[1].squeeze().double()).abs()
    return float(err0.item()), float(err1.item())


def _format_abs_errs(errs: tuple[float, float]) -> str:
    return f"sum_abs_err: {errs[0]:.6e}, sum_sq_abs_err: {errs[1]:.6e}"


def print_perf_summary(result: dict) -> None:
    rows = [
        (
            "torch",
            result["torch_ms"],
            result["torch_TFLOPs"],
            1.0,
        ),
        (
            "cuda fused FP32",
            result["fp32_cuda_ms"],
            result["fp32_cuda_TFLOPs"],
            result["speedup_fp32_cuda"],
        ),
    ]

    print("[performance]")
    print(f"  {'implementation':<28} {'time [ms]':>10} {'TFLOP/s':>10} {'speedup':>9}")
    for name, ms, tflops, speedup in rows:
        print(f"  {name:<28} {ms:10.4f} {tflops:10.2f} {speedup:8.2f}x")
    print()


@torch.compile
def relu_bat_reduce_fallback(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    S = torch.relu(B @ A.T)
    return S.sum(), (S * S).sum()
    

@torch.no_grad()
def bench_one(
    M: int,
    N: int,
    D: int,
    dtype=torch.float32,
    device="cuda",
    warmup_ms=0,
    rep_ms=1,
    tune_trials=200,
    tune_warmup_ms=1,
    tune_rep_ms=5,
):
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)

    ref = relu_bat_reduce_fallback(A, B)
    fp32_tuned = AutotuneReluBatReduce(
        configs = relu_bat_reduce_tune_config,
        n_trials = max(tune_trials, grid_size(make_choices(relu_bat_reduce_tune_config))),
        cache_path = "relu_bat_reduce_bench_fp32_kahan_sum2_autotune.json",
        warmup = tune_warmup_ms,
        rep = tune_rep_ms
    )

    fp32_tuned.resolve_decision((B, A))
    _chosen_params = fp32_tuned.decision_config

    if _chosen_params is None:
        out_fp32 = (torch.tensor([0.]), torch.tensor([0.]))
        fp32_params = {}
    else:
        fp32_params = asdict(_chosen_params)
        fp32_params['MS'] = _chosen_params.num_ms
        fp32_params.pop('num_ms', None)
        out_fp32 = relu_bat_reduce_fused(B, A, **fp32_params)
    err_fp32 = _abs_errs(out_fp32, ref)

    print(f"[correctness] M={M} N={N} D={D} dtype={dtype}")
    print(
        "  cuda fused FP32                 "
        f"{_format_abs_errs(err_fp32)}"
    )
    print(f"  cuda fused FP32                 tuned params: {fp32_params}")

    def torch_run(): return relu_bat_reduce_fallback(A, B)

    def fp32_run(): return fp32_tuned((B, A))

    torch.cuda.synchronize()

    t_torch = triton.testing.do_bench(torch_run, warmup=warmup_ms, rep=rep_ms)
    t_fp32 = triton.testing.do_bench(fp32_run, warmup=warmup_ms, rep=rep_ms)
    assert isinstance(t_torch, float)
    assert isinstance(t_fp32, float)

    return {
        "M": M,
        "N": N,
        "D": D,
        "dtype": str(dtype).replace("torch.", ""),
        "torch_ms": t_torch,
        "fp32_cuda_ms": t_fp32,
        "fp32_cuda_params": dict(fp32_params),
        "fp32_cuda_tune_ms": fp32_tuned.decision_runtime_ms,
        "torch_TFLOPs": ms_to_tflops(M, N, D, t_torch),
        "fp32_cuda_TFLOPs": ms_to_tflops(M, N, D, t_fp32),
        "speedup_fp32_cuda": float(t_torch / t_fp32),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="sum(ReLU(B A.T)) reduce kernel benchmark"
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=5,
        help="Benchmark warmup duration in milliseconds",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=20,
        help="Benchmark measurement duration in milliseconds",
    )
    parser.add_argument(
        "--autotune",
        type=str,
        default="cache",
        choices=['cache', 'force', 'disable', 'fallback'],
        help="CUDA kernel autotuning mode for the kernel benchmark",
    )
    parser.add_argument(
        "--autotune-cache-dir",
        "--autotune_cache_dir",
        type=str,
        default=None,
        help="Directory for SUMAC kernel autotune cache files",
    )
    parser.add_argument(
        "--autotune-verbose",
        "--autotune_verbose",
        action="store_true",
        help="Print CUDA kernel autotuning decisions and pruned trials",
    )
    args = parser.parse_args()
    autotune_mode = AutotuneMode(str.lower(args.autotune))
    torch.manual_seed(0)
    assert torch.cuda.is_available()

    with kernel_autotune_options(
        mode=autotune_mode,
        cache_dir=args.autotune_cache_dir,
        verbose=args.autotune_verbose,
    ):
        for D in (16,32,64,128,256):
            result = bench_one(
                M=288768,
                N=1408,
                D=D,
                dtype=torch.float32,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            print_perf_summary(result)
