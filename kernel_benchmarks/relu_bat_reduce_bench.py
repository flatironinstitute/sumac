import argparse
from functools import lru_cache

import optuna
import torch
import triton

from sumac.kernels.relu_bat_reduce import (
    grid_size as _grid_size,
    relu_bat_reduce_constraints,
    relu_bat_reduce_tune_config,
)
from sumac.kernels.tuning import autotune_cuda_kernel, relu_bat_reduce_key
from sumac.kernels.relu_bat_reduce_jit.custom_op import relu_bat_reduce_fused_op


def torch_impl(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    S = torch.relu(B @ A.T)
    return S.sum(), (S * S).sum()


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


def _require_valid_params(
    name: str,
    constraint_fn,
    args: tuple,
    params: dict,
) -> None:
    if constraint_fn(*args, **params):
        return
    raise RuntimeError(
        f"{name} resolved invalid autotune params {params}. "
        "Remove the corresponding autotune cache or set KERNEL_AUTOTUNE_FORCE=1."
    )


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


@lru_cache(maxsize=None)
def relu_bat_reduce_fp32_launcher(
    n_trials: int,
    warmup: int,
    rep: int,
):
    tune_config = relu_bat_reduce_tune_config()
    n_trials = max(n_trials, _grid_size(tune_config))

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_reduce_key,
        constraint_fn=relu_bat_reduce_constraints,
        cache_path="relu_bat_reduce_bench_fp32_kahan_sum2_autotune.json",
        n_trials=n_trials,
        warmup=warmup,
        rep=rep,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_bat_reduce_fp32_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        BM: int,
        BK: int,
        num_ms: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return relu_bat_reduce_fused_op(B, A, BM, BK, num_ms)

    return relu_bat_reduce_fp32_cuda


@torch.no_grad()
def bench_one(
    M: int,
    N: int,
    D: int,
    dtype=torch.float32,
    device="cuda",
    wm_iters=0,
    iters=1,
    tune_trials=200,
    tune_warmup_iters=1,
    tune_iters=5,
):
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)

    ref = torch_impl(A, B)

    fp32_tuned = relu_bat_reduce_fp32_launcher(
        tune_trials,
        tune_warmup_iters,
        tune_iters,
    )

    fp32_decision = fp32_tuned.resolve_decision(A, B)
    fp32_params = fp32_decision["params"]
    _require_valid_params(
        "cuda fused FP32",
        relu_bat_reduce_constraints,
        (A, B),
        fp32_params,
    )
    out_fp32 = relu_bat_reduce_fused_op(
        B,
        A,
        fp32_params["BM"],
        fp32_params["BK"],
        fp32_params["num_ms"],
    )
    err_fp32 = _abs_errs(out_fp32, ref)

    print(f"[correctness] M={M} N={N} D={D} dtype={dtype}")
    print(
        "  cuda fused FP32                 "
        f"{_format_abs_errs(err_fp32)}"
    )
    print(f"  cuda fused FP32                 tuned params: {fp32_params}")

    def torch_run():
        return torch_impl(A, B)

    def fp32_run():
        return relu_bat_reduce_fused_op(
            B,
            A,
            fp32_params["BM"],
            fp32_params["BK"],
            fp32_params["num_ms"],
        )

    torch.cuda.synchronize()

    t_torch = triton.testing.do_bench(torch_run, warmup=wm_iters, rep=iters)
    t_fp32 = triton.testing.do_bench(fp32_run, warmup=wm_iters, rep=iters)

    return {
        "M": M,
        "N": N,
        "D": D,
        "dtype": str(dtype).replace("torch.", ""),
        "torch_ms": float(t_torch),
        "fp32_cuda_ms": float(t_fp32),
        "fp32_cuda_params": dict(fp32_params),
        "fp32_cuda_tune_ms": float(fp32_decision["runtime_ms"]),
        "torch_TFLOPs": float(ms_to_tflops(M, N, D, t_torch)),
        "fp32_cuda_TFLOPs": float(ms_to_tflops(M, N, D, t_fp32)),
        "speedup_fp32_cuda": float(t_torch / t_fp32),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="sum(ReLU(B A.T)) reduce kernel benchmark"
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of benchmark warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=200,
        help="Number of autotuning trials",
    )
    parser.add_argument(
        "--tune-warmup-iters",
        type=int,
        default=1,
        help="Number of autotuning warmup iterations",
    )
    parser.add_argument(
        "--tune-iters",
        type=int,
        default=5,
        help="Number of autotuning benchmark iterations",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    assert torch.cuda.is_available()

    for D in (16,32,64,128,256):
        result = bench_one(
            M=288768,
            N=1408,
            D=D,
            dtype=torch.float32,
            wm_iters=args.warmup_iters,
            iters=args.iters,
            tune_trials=args.tune_trials,
            tune_warmup_iters=args.tune_warmup_iters,
            tune_iters=args.tune_iters,
        )
        print_perf_summary(result)
