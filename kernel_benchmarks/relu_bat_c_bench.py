import argparse
from dataclasses import asdict, dataclass
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
    AutotuneReluBatCFP32MfmaAMD,
    AutotuneReluBatCTf32MfmaAMD,
    AutotuneReluBatCTf32Sync,
    AutotuneReluBatCTf32Wgmma,
    relu_bat_c_fp32_mfma_tune_config,
    relu_bat_c_fp32_tune_config,
    relu_bat_c_tf32_mfma_tune_config,
    relu_bat_c_tf32_sync_tune_config,
    relu_bat_c_tf32_wgmma_tune_config,
    relu_bat_c_tf32_wgmma_available,
)

from sumac.config import AutotuneMode

from sumac.kernels.relu_batc_jit.api import relu_bat_c_fused_op
from sumac.kernels.relu_batc_jit_amd.api import (
    relu_bat_c_fp32_mfma_available,
)
from sumac.kernels.relu_batc_jit_amd.custom_op import (
    relu_bat_c_fp32_mfma_amd_op,
)
from sumac.kernels.relu_batc_tf32_jit_amd.api import (
    relu_bat_c_tf32_mfma_available,
)
from sumac.kernels.relu_batc_tf32_jit_amd.custom_op import (
    relu_bat_c_tf32_mfma_amd_op,
)
from sumac.kernels.relu_batc_tf32_jit.api import (
    relu_bat_c_tf32_mma_sync,
    relu_bat_c_tf32_wgmma,
)


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


class _CustomKernelOnlyBenchmarkTuner:
    """Keep fallback out of custom-kernel candidate selection."""

    def _bench_fallback(self, params) -> float:
        return float("inf")


class _BenchmarkReluBatCFP32(
    _CustomKernelOnlyBenchmarkTuner,
    AutotuneReluBatCFP32,
):
    pass


class _BenchmarkReluBatCFP32MfmaAMD(
    _CustomKernelOnlyBenchmarkTuner,
    AutotuneReluBatCFP32MfmaAMD,
):
    pass


class _BenchmarkReluBatCTf32MfmaAMD(
    _CustomKernelOnlyBenchmarkTuner,
    AutotuneReluBatCTf32MfmaAMD,
):
    pass


class _BenchmarkReluBatCTf32Sync(
    _CustomKernelOnlyBenchmarkTuner,
    AutotuneReluBatCTf32Sync,
):
    pass


class _BenchmarkReluBatCTf32Wgmma(
    _CustomKernelOnlyBenchmarkTuner,
    AutotuneReluBatCTf32Wgmma,
):
    pass


@dataclass(frozen=True)
class _BenchmarkFunctions:
    backend: str
    backend_label: str
    fp32_label: str
    fp32_tuned: AutotuneCudaKernel
    fp32_op: Callable
    fp32_rename_num_ms_to_MS: bool
    tf32_mfma_tuned: AutotuneCudaKernel | None
    tf32_mfma_op: Callable | None
    tf32_mma_sync_tuned: AutotuneCudaKernel | None
    tf32_wgmma_tuned: AutotuneCudaKernel | None


def ms_to_tflops(M, N, D, ms):
    flops = M * N * (D + D - 1) + M * D * (N + N - 1)

    s = ms * 1e-3
    return flops / s / 1e12


def _print_perf_summary(result: dict) -> None:
    rows = [
        (
            "torch.compile",
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
    if result["tf32_mfma_ms"] is not None:
        rows.append(
            (
                "HIP fused TF32/XF32 MFMA",
                result["tf32_mfma_ms"],
                result["tf32_mfma_TFLOPs"],
                result["speedup_tf32_mfma"],
            )
        )
    if result["tf32_mma_sync_ms"] is not None:
        rows.append(
            (
                "CUDA fused TF32 MMA sync (1D)",
                result["tf32_mma_sync_ms"],
                result["tf32_mma_sync_TFLOPs"],
                result["speedup_tf32_mma_sync"],
            )
        )
    if result["wgmma_ms"] is not None:
        rows.append(
            (
                "CUDA fused TF32 WGMMA (1D)",
                result["wgmma_ms"],
                result["wgmma_TFLOPs"],
                result["speedup_wgmma"],
            )
        )

    print("[performance]")
    print(
        f"  {'implementation':<32} {'time [ms]':>10} "
        f"{'TFLOP/s':>10} {'speedup':>9}"
    )
    for name, ms, tflops, speedup in rows:
        print(f"  {name:<32} {ms:10.4f} {tflops:10.2f} {speedup:8.2f}x")
    print()


def _print_correctness(
    M: int,
    N: int,
    D: int,
    dtype: torch.dtype,
    backend_label: str,
    fp32_label: str,
    err_fp32: float,
    err_tf32_mfma: float | None,
    err_tf32_mma_sync: float | None,
    fp32_params_dict: dict,
    tf32_mfma_params_dict: dict,
    sync_params_dict: dict,
    use_wgmma: bool,
    err_wgmma: float | None = None,
    wgmma_params_dict: dict | None = None,
):
    print(f"[correctness] M={M} N={N} D={D} dtype={dtype}")
    print(
        f"  {fp32_label:<34}"
        f"max_abs_err vs torch: {err_fp32:.6e}"
    )
    print(
        f"  {fp32_label:<34}"
        f"tuned params: {fp32_params_dict}"
    )
    if err_tf32_mfma is not None:
        print(
            "  HIP fused TF32/XF32 MFMA        "
            f"max_abs_err vs torch: {err_tf32_mfma:.6e}"
        )
        print(
            "  HIP fused TF32/XF32 MFMA        "
            f"tuned params: {tf32_mfma_params_dict}"
        )
    if err_tf32_mma_sync is not None:
        print(
            "  CUDA fused TF32 MMA sync (1D)     "
            f"max_abs_err vs torch: {err_tf32_mma_sync:.6e}"
        )
        print(
            "  CUDA fused TF32 MMA sync (1D)     "
            f"tuned params: {sync_params_dict}"
        )
    if use_wgmma and err_wgmma is not None:
        print(
            "  CUDA fused TF32 WGMMA (1D)  "
            f"max_abs_err vs torch: {err_wgmma:.6e}"
        )
        print(f"  CUDA fused TF32 WGMMA (1D)  tuned params: {wgmma_params_dict}")
    elif backend_label == "CUDA":
        print("  CUDA fused TF32 WGMMA (1D)  skipped: requires SM90/Hopper")


def _init_functions(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: int,
    tune_trials: int,
    tune_warmup_ms: int,
    tune_rep_ms: int
) -> _BenchmarkFunctions:
    if _is_rocm():
        if not relu_bat_c_fp32_mfma_available(A.device, D):
            raise RuntimeError(
                "The HIP FP32 benchmark requires gfx942 and a D whose "
                "minimum MFMA panel fits the device LDS budget"
            )
        mfma_config = relu_bat_c_fp32_mfma_tune_config(D)
        relu_bat_c_fp32_tuned = _BenchmarkReluBatCFP32MfmaAMD(
            configs=mfma_config,
            cache_path="relu_bat_c_hip_fp32_mfma_autotune.json",
            n_trials=max(tune_trials, grid_size(make_choices(mfma_config))),
            warmup=tune_warmup_ms,
            rep=tune_rep_ms,
        )
        relu_bat_c_tf32_mfma_tuned = None
        relu_bat_c_tf32_mfma_op = None
        if relu_bat_c_tf32_mfma_available(A.device, D):
            tf32_mfma_config = relu_bat_c_tf32_mfma_tune_config(D)
            relu_bat_c_tf32_mfma_tuned = _BenchmarkReluBatCTf32MfmaAMD(
                configs=tf32_mfma_config,
                cache_path="relu_bat_c_hip_tf32_mfma_autotune.json",
                n_trials=max(
                    tune_trials,
                    grid_size(make_choices(tf32_mfma_config)),
                ),
                warmup=tune_warmup_ms,
                rep=tune_rep_ms,
            )
            relu_bat_c_tf32_mfma_op = relu_bat_c_tf32_mfma_amd_op
        return _BenchmarkFunctions(
            backend="rocm",
            backend_label="HIP",
            fp32_label="HIP fused FP32 MFMA",
            fp32_tuned=relu_bat_c_fp32_tuned,
            fp32_op=relu_bat_c_fp32_mfma_amd_op,
            fp32_rename_num_ms_to_MS=False,
            tf32_mfma_tuned=relu_bat_c_tf32_mfma_tuned,
            tf32_mfma_op=relu_bat_c_tf32_mfma_op,
            tf32_mma_sync_tuned=None,
            tf32_wgmma_tuned=None,
        )

    n_trials_fp32 = max(
        tune_trials,
        grid_size(make_choices(relu_bat_c_fp32_tune_config)),
    )
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
    return _BenchmarkFunctions(
        backend="cuda",
        backend_label="CUDA",
        fp32_label="CUDA fused FP32 (1D)",
        fp32_tuned=relu_bat_c_fp32_tuned,
        fp32_op=relu_bat_c_fused_op,
        fp32_rename_num_ms_to_MS=True,
        tf32_mfma_tuned=None,
        tf32_mfma_op=None,
        tf32_mma_sync_tuned=relu_bat_c_tf32_mma_sync_tuned,
        tf32_wgmma_tuned=relu_bat_c_wgmma_tuned,
    )


def _tune_and_test_correctness(
    ref: Tensor,
    fn_tuner: AutotuneCudaKernel | None,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    label: str,
    base_fn: Callable | None,
    rename_num_ms_to_MS: bool = False,
):
    if fn_tuner is None or base_fn is None:
        return (None, {})
    fn_tuner.resolve_decision((A, B, C))
    _chosen_params = fn_tuner.decision_config
    if _chosen_params is None:
        raise RuntimeError(
            f"{label} benchmark did not resolve to a custom-kernel configuration"
        )
    _params_dict = asdict(_chosen_params)
    if rename_num_ms_to_MS:
        # Note this is purely a convenience--the name of this parameter
        # in the class differs from the parameter in the custom kernel
        # fn, so we have to correct it if we want to get away with
        # lazily unpacking the dictionary as keyword arguments.
        # But otherwise we'd have to spell out the arguments for each
        # of the underlying cases.
        # possible TODO: sync up the parameter names between fns and config
        _params_dict['MS'] = _chosen_params.num_ms
        _params_dict.pop('num_ms', None)
    out_custom = base_fn(A, B, C, **_params_dict)
    _err = (out_custom - ref).abs().max().item()
    return (_err, _params_dict)


def _bench_fn(fn: Callable, warmup_ms: int, rep_ms: int):
    time_ms = float(
        triton.testing.do_bench(  # type: ignore
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
    functions = _init_functions(
        A, B, C, D, tune_trials, tune_warmup_ms, tune_rep_ms
    )
    fp32_tuned = functions.fp32_tuned
    tf32_mfma_tuned = functions.tf32_mfma_tuned
    tf32_mma_sync_tuned = functions.tf32_mma_sync_tuned
    tf32_wgmma_tuned = functions.tf32_wgmma_tuned

    (fp32_err, fp32_params) = _tune_and_test_correctness(
        ref,
        fp32_tuned,
        A,
        B,
        C,
        "fp32",
        functions.fp32_op,
        rename_num_ms_to_MS=functions.fp32_rename_num_ms_to_MS,
    )
    assert fp32_err is not None
    (tf32_mfma_err, tf32_mfma_params) = _tune_and_test_correctness(
        ref,
        tf32_mfma_tuned,
        A,
        B,
        C,
        "tf32_mfma",
        functions.tf32_mfma_op,
    )
    (sync_err, sync_params) = _tune_and_test_correctness(
        ref,
        tf32_mma_sync_tuned,
        A,
        B,
        C,
        "tf32_mma_sync",
        relu_bat_c_tf32_mma_sync if tf32_mma_sync_tuned is not None else None,
    )
    (wgmma_err, wgmma_params) = _tune_and_test_correctness(
        ref,
        tf32_wgmma_tuned,
        A,
        B,
        C,
        "tf32_wgmma",
        relu_bat_c_tf32_wgmma if tf32_wgmma_tuned is not None else None,
    )

    use_wgmma = tf32_wgmma_tuned is not None
    _print_correctness(
        M,
        N,
        D,
        dtype,
        functions.backend_label,
        functions.fp32_label,
        fp32_err,
        tf32_mfma_err,
        sync_err,
        fp32_params,
        tf32_mfma_params,
        sync_params,
        use_wgmma,
        wgmma_err,
        wgmma_params,
    )

    # Set up benchmarkable functions
    def torch_run(): return relu_bat_c_fallback(A, B, C)
    def fp32_run(): return fp32_tuned((A, B, C))

    torch.cuda.synchronize()

    t_torch = _bench_fn(torch_run, warmup_ms, rep_ms)
    t_fp32 = _bench_fn(fp32_run, warmup_ms, rep_ms)
    t_tf32_mfma = None
    if tf32_mfma_tuned is not None:
        t_tf32_mfma = _bench_fn(
            lambda: tf32_mfma_tuned((A, B, C)), warmup_ms, rep_ms
        )
    t_tf32_mma_sync = None
    if tf32_mma_sync_tuned is not None:
        t_tf32_mma_sync = _bench_fn(
            lambda: tf32_mma_sync_tuned((A, B, C)), warmup_ms, rep_ms
        )
    t_wgmma = None
    if tf32_wgmma_tuned is not None:
        t_wgmma = _bench_fn(
            lambda: tf32_wgmma_tuned((A, B, C)), warmup_ms, rep_ms
        )

    fp32_tflops = ms_to_tflops(M, N, D, t_fp32)
    speedup_fp32 = t_torch / t_fp32
    is_rocm = functions.backend == "rocm"

    return {
        "M": M, "N": N, "D": D, "dtype": str(dtype).replace("torch.", ""),
        "backend": functions.backend,
        "backend_label": functions.backend_label,
        "fp32_label": functions.fp32_label,

        "torch_ms":   t_torch,
        "fp32_ms": t_fp32,
        "fp32_params": fp32_params,
        "fp32_tune_ms": fp32_tuned.decision_runtime_ms,
        "fp32_mfma_ms": t_fp32 if is_rocm else None,
        "fp32_mfma_params": fp32_params if is_rocm else None,
        "fp32_mfma_tune_ms": (
            fp32_tuned.decision_runtime_ms if is_rocm else None
        ),
        "tf32_mfma_ms": t_tf32_mfma,
        "tf32_mfma_params": (
            None if tf32_mfma_tuned is None else tf32_mfma_params
        ),
        "tf32_mfma_tune_ms": (
            None if tf32_mfma_tuned is None
            else tf32_mfma_tuned.decision_runtime_ms
        ),
        "tf32_mma_sync_ms": t_tf32_mma_sync,
        "tf32_mma_sync_params": sync_params,
        "tf32_mma_sync_tune_ms": (
            None if tf32_mma_sync_tuned is None
            else tf32_mma_sync_tuned.decision_runtime_ms
        ),
        "wgmma_ms": t_wgmma,
        "wgmma_params": None if not use_wgmma else wgmma_params,
        "wgmma_tune_ms": (
            None if tf32_wgmma_tuned is None
            else tf32_wgmma_tuned.decision_runtime_ms
        ),
        "torch_TFLOPs":   ms_to_tflops(M, N, D, t_torch),
        "fp32_TFLOPs": fp32_tflops,
        "fp32_mfma_TFLOPs": fp32_tflops if is_rocm else None,
        "tf32_mfma_TFLOPs": (
            None if t_tf32_mfma is None
            else ms_to_tflops(M, N, D, t_tf32_mfma)
        ),
        "tf32_mma_sync_TFLOPs": (
            None if t_tf32_mma_sync is None
            else ms_to_tflops(M, N, D, t_tf32_mma_sync)
        ),
        "wgmma_TFLOPs": (
            None if t_wgmma is None
            else ms_to_tflops(M, N, D, t_wgmma)
        ),
        "speedup_fp32": speedup_fp32,
        "speedup_fp32_mfma": speedup_fp32 if is_rocm else None,
        "speedup_tf32_mfma": (
            None if t_tf32_mfma is None else t_torch / t_tf32_mfma
        ),
        "speedup_tf32_mma_sync": (
            None if t_tf32_mma_sync is None
            else t_torch / t_tf32_mma_sync
        ),
        "speedup_wgmma": None if t_wgmma is None else t_torch / t_wgmma,

        # Preserve the original result keys for downstream NVIDIA consumers.
        "fp32_cuda_ms": None if is_rocm else t_fp32,
        "fp32_cuda_params": None if is_rocm else fp32_params,
        "fp32_cuda_tune_ms": (
            None if is_rocm else fp32_tuned.decision_runtime_ms
        ),
        "fp32_cuda_TFLOPs": None if is_rocm else fp32_tflops,
        "speedup_fp32_cuda": None if is_rocm else speedup_fp32,
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
    parser.add_argument(
        "--d-values",
        "--d",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="Inner dimensions to benchmark (default: 16 32 64 128 256)",
    )
    args = parser.parse_args()

    if any(d <= 0 for d in args.d_values):
        parser.error("--d-values entries must all be positive")

    autotune_mode = AutotuneMode(str.lower(args.autotune))

    torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()

    for d in args.d_values:
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
