import argparse
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import Tensor
import triton

from sumac.kernels.tuning import (
    grid_size,
    make_choices,
    kernel_autotune_options,
    AutotuneReluBatReduce,
    AutotuneReluBatReduceMfmaAMD,
    relu_bat_reduce_mfma_tune_config,
    relu_bat_reduce_tune_config,
)

from sumac.config import AutotuneMode
from sumac.kernels.relu_bat_reduce_jit.api import relu_bat_reduce_fused
from sumac.kernels.relu_bat_reduce_jit_amd.api import (
    relu_bat_reduce_fp32_mfma,
    relu_bat_reduce_fp32_mfma_available,
)


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
            result["fp32_label"],
            result["fp32_ms"],
            result["fp32_TFLOPs"],
            result["speedup_fp32"],
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


class _CustomKernelOnlyBenchmarkTuner:
    """Keep fallback out of custom-kernel candidate selection."""

    def _bench_fallback(self, params) -> float:
        return float("inf")


class _BenchmarkReluBatReduce(
    _CustomKernelOnlyBenchmarkTuner,
    AutotuneReluBatReduce,
):
    pass


class _BenchmarkReluBatReduceMfmaAMD(
    _CustomKernelOnlyBenchmarkTuner,
    AutotuneReluBatReduceMfmaAMD,
):
    pass


@dataclass(frozen=True)
class _BenchmarkFunctions:
    backend: str
    fp32_label: str
    fp32_tuned: AutotuneReluBatReduce | AutotuneReluBatReduceMfmaAMD
    fp32_op: Callable
    rename_num_ms_to_MS: bool


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _init_functions(
    A: Tensor,
    D: int,
    tune_trials: int,
    tune_warmup_ms: int,
    tune_rep_ms: int,
) -> _BenchmarkFunctions:
    if _is_rocm():
        if not relu_bat_reduce_fp32_mfma_available(A.device, D):
            raise RuntimeError(
                "The HIP FP32 reduction benchmark requires gfx942 and a D "
                "supported by the MFMA register/LDS bounds"
            )
        configs = relu_bat_reduce_mfma_tune_config
        tuned = _BenchmarkReluBatReduceMfmaAMD(
            configs=configs,
            n_trials=max(tune_trials, grid_size(make_choices(configs))),
            cache_path="relu_bat_reduce_bench_hip_mfma_autotune.json",
            warmup=tune_warmup_ms,
            rep=tune_rep_ms,
        )
        return _BenchmarkFunctions(
            backend="rocm",
            fp32_label="HIP fused FP32 MFMA",
            fp32_tuned=tuned,
            fp32_op=relu_bat_reduce_fp32_mfma,
            rename_num_ms_to_MS=False,
        )

    configs = relu_bat_reduce_tune_config
    tuned = _BenchmarkReluBatReduce(
        configs=configs,
        n_trials=max(tune_trials, grid_size(make_choices(configs))),
        cache_path="relu_bat_reduce_bench_fp32_kahan_sum2_autotune.json",
        warmup=tune_warmup_ms,
        rep=tune_rep_ms,
    )
    return _BenchmarkFunctions(
        backend="cuda",
        fp32_label="CUDA fused FP32",
        fp32_tuned=tuned,
        fp32_op=relu_bat_reduce_fused,
        rename_num_ms_to_MS=True,
    )


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
    functions = _init_functions(
        B,
        D,
        tune_trials,
        tune_warmup_ms,
        tune_rep_ms,
    )
    fp32_tuned = functions.fp32_tuned

    fp32_tuned.resolve_decision((B, A))
    _chosen_params = fp32_tuned.decision_config

    if _chosen_params is None:
        raise RuntimeError(
            "fused FP32 benchmark did not resolve to a custom-kernel configuration"
        )
    fp32_params = asdict(_chosen_params)
    if functions.rename_num_ms_to_MS:
        fp32_params["MS"] = _chosen_params.num_ms
        fp32_params.pop("num_ms", None)
    out_fp32 = functions.fp32_op(B, A, **fp32_params)
    err_fp32 = _abs_errs(out_fp32, ref)

    print(f"[correctness] M={M} N={N} D={D} dtype={dtype}")
    print(
        f"  {functions.fp32_label:<32}"
        f"{_format_abs_errs(err_fp32)}"
    )
    print(
        f"  {functions.fp32_label:<32}"
        f"tuned params: {fp32_params}"
    )

    def torch_run(): return relu_bat_reduce_fallback(A, B)

    def fp32_run(): return fp32_tuned((B, A))

    torch.cuda.synchronize()

    t_torch = triton.testing.do_bench(torch_run, warmup=warmup_ms, rep=rep_ms)
    t_fp32 = triton.testing.do_bench(fp32_run, warmup=warmup_ms, rep=rep_ms)
    assert isinstance(t_torch, float)
    assert isinstance(t_fp32, float)

    fp32_tflops = ms_to_tflops(M, N, D, t_fp32)
    speedup_fp32 = float(t_torch / t_fp32)
    is_rocm = functions.backend == "rocm"

    return {
        "M": M,
        "N": N,
        "D": D,
        "dtype": str(dtype).replace("torch.", ""),
        "backend": functions.backend,
        "fp32_label": functions.fp32_label,
        "torch_ms": t_torch,
        "fp32_ms": t_fp32,
        "fp32_params": dict(fp32_params),
        "fp32_tune_ms": fp32_tuned.decision_runtime_ms,
        "torch_TFLOPs": ms_to_tflops(M, N, D, t_torch),
        "fp32_TFLOPs": fp32_tflops,
        "speedup_fp32": speedup_fp32,
        "fp32_mfma_ms": t_fp32 if is_rocm else None,
        "fp32_mfma_params": dict(fp32_params) if is_rocm else None,
        "fp32_mfma_tune_ms": (
            fp32_tuned.decision_runtime_ms if is_rocm else None
        ),
        "fp32_mfma_TFLOPs": fp32_tflops if is_rocm else None,
        "speedup_fp32_mfma": speedup_fp32 if is_rocm else None,
        # Preserve the existing NVIDIA result keys for downstream consumers.
        "fp32_cuda_ms": None if is_rocm else t_fp32,
        "fp32_cuda_params": None if is_rocm else dict(fp32_params),
        "fp32_cuda_tune_ms": (
            None if is_rocm else fp32_tuned.decision_runtime_ms
        ),
        "fp32_cuda_TFLOPs": None if is_rocm else fp32_tflops,
        "speedup_fp32_cuda": None if is_rocm else speedup_fp32,
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
        choices=['cache', 'force', 'disable'],
        help="Custom GPU kernel autotuning mode for the kernel benchmark",
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
        help="Print custom GPU kernel autotuning decisions and pruned trials",
    )
    args = parser.parse_args()
    autotune_mode = AutotuneMode(str.lower(args.autotune))
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
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
