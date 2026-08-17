import argparse
from dataclasses import asdict
import torch
from torch import Tensor
import triton
from typing import Callable


from sumac.kernels.tuning import (
    grid_size,
    make_choices,
    kernel_autotune_options,
    relu_bat_c_fallback,
    AutotuneCudaKernel,
    AutotuneReluBatCFP32,
    AutotuneReluBatCTf32Sync,
    AutotuneReluBatCTf32Wgmma,
    relu_bat_c_fp32_tune_config,
    relu_bat_c_tf32_sync_tune_config,
    relu_bat_c_tf32_wgmma_tune_config,
    relu_bat_c_tf32_wgmma_available,
)

from sumac.config import AutotuneMode

from sumac.kernels.relu_batc_jit.api import relu_bat_c_fused_op
from sumac.kernels.relu_batc_tf32_jit.api import (
    relu_bat_c_tf32_mma_sync,
    relu_bat_c_tf32_wgmma,
)


class _CudaOnlyBenchmarkTuner:
    """Keep fallback out of CUDA candidate selection for this benchmark."""

    def _bench_fallback(self, params) -> float:
        return float("inf")


class _BenchmarkReluBatCFP32(
    _CudaOnlyBenchmarkTuner,
    AutotuneReluBatCFP32,
):
    pass


class _BenchmarkReluBatCTf32Sync(
    _CudaOnlyBenchmarkTuner,
    AutotuneReluBatCTf32Sync,
):
    pass


class _BenchmarkReluBatCTf32Wgmma(
    _CudaOnlyBenchmarkTuner,
    AutotuneReluBatCTf32Wgmma,
):
    pass


def ms_to_tflops(M, N, D, ms):
        flops = M * N * (D + D - 1) + M * D * (N + N - 1)
    
        s = ms * 1e-3
        return flops / s / 1e12


def _print_perf_summary(result: dict) -> None:
    rows = [
        (
            "torch eager",
            result["torch_eager_ms"],
            result["torch_eager_TFLOPs"],
            1.0,
        ),
        (
            "torch.compile",
            result["torch_ms"],
            result["torch_TFLOPs"],
            result["speedup_torch_compile"],
        ),
        (
            "cuda fused FP32 (1D)",
            result["fp32_cuda_ms"],
            result["fp32_cuda_TFLOPs"],
            result["speedup_fp32_cuda"],
        ),
        (
            "cuda fused TF32 MMA sync (1D)",
            result["tf32_mma_sync_ms"],
            result["tf32_mma_sync_TFLOPs"],
            result["speedup_tf32_mma_sync"],
        ),
    ]
    if result["wgmma_ms"] is not None:
        rows.append(
            (
                "cuda fused TF32 WGMMA (1D)",
                result["wgmma_ms"],
                result["wgmma_TFLOPs"],
                result["speedup_wgmma"],
            )
        )

    print("[performance]")
    print(
        f"  {'implementation':<32} {'time [ms]':>10} "
        f"{'TFLOP/s':>10} {'vs eager':>11}"
    )
    for name, ms, tflops, speedup in rows:
        print(f"  {name:<32} {ms:10.4f} {tflops:10.2f} {speedup:8.2f}x")
    print()


def _print_correctness(
    M: int,
    N: int,
    D: int,
    dtype: torch.dtype,
    err_fp32_cuda: float,
    err_tf32_mma_sync: float,
    fp32_params_dict: dict,
    sync_params_dict: dict,
    use_wgmma: bool,
    err_wgmma: float = -1.,
    wgmma_params_dict: dict = {},
):
    print(f"[correctness] M={M} N={N} D={D} dtype={dtype}")
    print(
        "  cuda fused FP32 (1D)              "
        f"max_abs_err vs torch: {err_fp32_cuda:.6e}"
    )
    print(f"  cuda fused FP32 (1D)              tuned params: {fp32_params_dict}")
    print(
        "  cuda fused TF32 MMA sync (1D)     "
        f"max_abs_err vs torch: {err_tf32_mma_sync:.6e}"
    )
    print(
        "  cuda fused TF32 MMA sync (1D)     "
        f"tuned params: {sync_params_dict}"
    )
    if use_wgmma:
        print(
            "  cuda fused TF32 WGMMA (1D)  "
            f"max_abs_err vs torch: {err_wgmma:.6e}"
        )
        print(f"  cuda fused TF32 WGMMA (1D)  tuned params: {wgmma_params_dict}")
    else:
        print("  cuda fused TF32 WGMMA (1D)  skipped: requires SM90/Hopper")


def _init_functions(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: int,
    tune_trials: int,
    tune_warmup_ms: int,
    tune_rep_ms: int
):
    n_trials_fp32 = max(tune_trials, grid_size(make_choices(relu_bat_c_fp32_tune_config)))
    relu_bat_c_fp32_tuned = _BenchmarkReluBatCFP32(
        configs = relu_bat_c_fp32_tune_config,
        cache_path = "relu_bat_c_jit_autotune.json",
        n_trials = n_trials_fp32,
        warmup = tune_warmup_ms,
        rep = tune_rep_ms,
    )
    sync_config = relu_bat_c_tf32_sync_tune_config(D)
    relu_bat_c_tf32_mma_sync_tuned = _BenchmarkReluBatCTf32Sync(
        configs = sync_config,
        cache_path = "relu_bat_c_tf32_mma_autotune.json",
        n_trials = tune_trials,
        warmup = tune_warmup_ms,
        rep = tune_rep_ms,
    )
    relu_bat_c_wgmma_tuned = None
    if relu_bat_c_tf32_wgmma_available(A):
        tune_config = relu_bat_c_tf32_wgmma_tune_config(D)
        relu_bat_c_wgmma_tuned = _BenchmarkReluBatCTf32Wgmma(
            configs = tune_config,
            cache_path = "relu_bat_c_tf32_wgmma_mode_autotune.json",
            n_trials = tune_trials,
            warmup = tune_warmup_ms,
            rep = tune_rep_ms,
        )
    return (relu_bat_c_fp32_tuned, relu_bat_c_tf32_mma_sync_tuned, relu_bat_c_wgmma_tuned)


def _tune_and_test_correctness(
    ref: Tensor,
    fn_tuner: AutotuneCudaKernel | None,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    label: str,
    base_fn: Callable,
    is_fp32: bool = False
):
    if fn_tuner is None:
        return (0, {})
    fn_tuner.resolve_decision((A, B, C))
    _chosen_params = fn_tuner.decision_config
    if _chosen_params is None:
        raise RuntimeError(
            f"{label} benchmark did not resolve to a CUDA configuration"
        )
    _params_dict = asdict(_chosen_params)
    if is_fp32:
        # Note this is purely a convenience--the name of this parameter
        # in the class differs from the parameter in the custom kernel
        # fn, so we have to correct it if we want to get away with
        # lazily unpacking the dictionary as keyword arguments.
        # But otherwise we'd have to spell out the arguments for each
        # of the underlying cases.
        # possible TODO: sync up the parameter names between fns and config
        _params_dict['MS'] = _chosen_params.num_ms
        _params_dict.pop('num_ms', None)
    out_fp32_cuda = base_fn(A, B, C, **_params_dict)
    _err = (out_fp32_cuda - ref).abs().max().item()
    return (_err, _params_dict)


def _bench_fn(fn: Callable, warmup_ms: int, rep_ms: int):
    time_ms = float( 
        triton.testing.do_bench( #type: ignore
            fn,
            warmup=warmup_ms,
            rep=rep_ms,
        )
    )
    return time_ms


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
    C = torch.randn((N, D), device=device, dtype=dtype)

    ref = relu_bat_c_fallback(A, B, C)
    (fp32_tuned, tf32_mma_sync_tuned, tf32_wgmma_tuned) = \
        _init_functions(A, B, C, D, tune_trials, tune_warmup_ms, tune_rep_ms)

    (fp32_err, fp32_params) = _tune_and_test_correctness(ref, fp32_tuned, A, B, C, "fp32", relu_bat_c_fused_op, True)
    (sync_err, sync_params) = _tune_and_test_correctness(ref, tf32_mma_sync_tuned, A, B, C, "tf32_mma_sync", relu_bat_c_tf32_mma_sync)
    (wgmma_err, wgmma_params) = _tune_and_test_correctness(ref, tf32_wgmma_tuned, A, B, C, "tf32_wgmma", relu_bat_c_tf32_wgmma)

    use_wgmma = tf32_wgmma_tuned is not None
    _print_correctness(M, N, D, dtype, fp32_err, sync_err, fp32_params, sync_params, use_wgmma, wgmma_err, wgmma_params)

    # Set up benchmarkable functions
    def torch_run(): return relu_bat_c_fallback(A, B, C)
    def fp32_cuda_run(): return fp32_tuned((A, B, C))
    def tf32_mma_sync_run(): return tf32_mma_sync_tuned((A, B, C))

    def wgmma_run():
        if tf32_wgmma_tuned is None:
            return relu_bat_c_fallback(A, B, C)
        return tf32_wgmma_tuned((A, B, C))

    torch.cuda.synchronize()

    t_torch = _bench_fn(torch_run, warmup_ms, rep_ms)
    t_fp32_cuda = _bench_fn(fp32_cuda_run, warmup_ms, rep_ms)
    t_tf32_mma_sync = _bench_fn(tf32_mma_sync_run, warmup_ms, rep_ms)
    t_wgmma = None if not use_wgmma else _bench_fn(wgmma_run, warmup_ms, rep_ms)

    return {
        "M": M, "N": N, "D": D, "dtype": str(dtype).replace("torch.", ""),

        "torch_ms":   t_torch,
        "fp32_cuda_ms": t_fp32_cuda,
        "fp32_cuda_params": fp32_params,
        "fp32_cuda_tune_ms": fp32_tuned.decision_runtime_ms,
        "tf32_mma_sync_ms": t_tf32_mma_sync,
        "tf32_mma_sync_params": sync_params,
        "tf32_mma_sync_tune_ms": tf32_mma_sync_tuned.decision_runtime_ms,
        "wgmma_ms": t_wgmma,
        "wgmma_params": None if not use_wgmma else wgmma_params,
        "wgmma_tune_ms": (
            None if tf32_wgmma_tuned is None
            else tf32_wgmma_tuned.decision_runtime_ms
        ),
        "torch_TFLOPs":   ms_to_tflops(M, N, D, t_torch),
        "fp32_cuda_TFLOPs": ms_to_tflops(M, N, D, t_fp32_cuda),
        "tf32_mma_sync_TFLOPs": ms_to_tflops(M, N, D, t_tf32_mma_sync),
        "wgmma_TFLOPs": (
            None if t_wgmma is None
            else ms_to_tflops(M, N, D, t_wgmma)
        ),
        "speedup_fp32_cuda": t_torch / t_fp32_cuda,
        "speedup_tf32_mma_sync": t_torch / t_tf32_mma_sync,
        "speedup_wgmma": None if t_wgmma is None else t_torch / t_wgmma,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ReLU(B A.T)C Kernel benchmark")
    parser.add_argument("--warmup-ms", type=int, default=5,
                        help="Benchmark warmup duration in milliseconds")
    parser.add_argument("--rep-ms", type=int, default=20,
                        help="Benchmark measurement duration in milliseconds")
    parser.add_argument(
        "--autotune",
        type=str,
        default="cache",
        choices=['cache', 'force', 'disable'],
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
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()

    for d in [16, 32, 64, 128, 256]:
        with kernel_autotune_options(
            mode=autotune_mode,
            cache_dir=args.autotune_cache_dir,
            verbose=args.autotune_verbose,
        ):
            r = bench_one(
                M=250000,
                N=2500,
                D=d,
                dtype=torch.float32,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            _print_perf_summary(r)
