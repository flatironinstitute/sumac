import torch
import triton
import argparse
import optuna

from functools import lru_cache

from sumac.kernels.relu_bat_c import (
    grid_size as _grid_size,
    relu_bat_c_fp32_constraints,
    relu_bat_c_fp32_tune_config,
    relu_bat_c_tf32_sync_constraints,
    relu_bat_c_tf32_sync_tune_config,
    relu_bat_c_tf32_wgmma_available,
    relu_bat_c_tf32_wgmma_constraints,
    relu_bat_c_tf32_wgmma_tune_config,
)
from sumac.kernels.tuning import (
    AUTOTUNE_MODES,
    autotune_cuda_kernel,
    kernel_autotune_options,
    relu_bat_c_key,
)
from sumac.kernels.relu_batc_jit.api import relu_bat_c_fused_op
from sumac.kernels.relu_batc_tf32_jit.api import (
    relu_bat_c_tf32_mma_sync,
    relu_bat_c_tf32_wgmma,
)

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


def print_perf_summary(result: dict) -> None:
    rows = [
        (
            "torch",
            result["torch_ms"],
            result["torch_TFLOPs"],
            1.0,
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
    print(f"  {'implementation':<32} {'time [ms]':>10} {'TFLOP/s':>10} {'speedup':>9}")
    for name, ms, tflops, speedup in rows:
        print(f"  {name:<32} {ms:10.4f} {tflops:10.2f} {speedup:8.2f}x")
    print()


@lru_cache(maxsize=None)
def relu_bat_c_fp32_cuda_launcher(
    n_trials: int,
    warmup_ms: int,
    rep_ms: int,
):
    tune_config = relu_bat_c_fp32_tune_config()
    n_trials = max(n_trials, _grid_size(tune_config))

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_fp32_constraints,
        cache_path="relu_bat_c_jit_autotune.json",
        n_trials=n_trials,
        warmup=warmup_ms,
        rep=rep_ms,
        sampler=optuna.samplers.GridSampler(search_space=tune_config)
    )
    def relu_bat_c_fp32_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BK: int,
        num_ms: int,
    ) -> torch.Tensor:
        return relu_bat_c_fused_op(A, B, C, BM, BK, num_ms)

    return relu_bat_c_fp32_cuda


@lru_cache(maxsize=None)
def relu_bat_c_tf32_sync_launcher(
    D: int,
    n_trials: int,
    warmup_ms: int,
    rep_ms: int,
):
    tune_config = relu_bat_c_tf32_sync_tune_config(D)

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_tf32_sync_constraints,
        cache_path="relu_bat_c_tf32_mma_autotune.json",
        n_trials=n_trials,
        warmup=warmup_ms,
        rep=rep_ms,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_bat_c_tf32_sync_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BN: int,
        M_TILES: int,
        num_stages: int,
    ) -> torch.Tensor:
        return relu_bat_c_tf32_mma_sync(
            A,
            B,
            C,
            BM=BM,
            BN=BN,
            M_TILES=M_TILES,
            num_stages=num_stages,
        )

    return relu_bat_c_tf32_sync_cuda


@lru_cache(maxsize=None)
def relu_bat_c_tf32_wgmma_launcher(
    D: int,
    n_trials: int,
    warmup_ms: int,
    rep_ms: int,
):
    tune_config = relu_bat_c_tf32_wgmma_tune_config(D)
    n_trials = max(n_trials, _grid_size(tune_config))

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_tf32_wgmma_constraints,
        cache_path="relu_bat_c_tf32_wgmma_mode_autotune.json",
        n_trials=n_trials,
        warmup=warmup_ms,
        rep=rep_ms,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_bat_c_tf32_wgmma_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BN: int,
        WGMMA_S_N: int,
        WGMMA_Y_N: int,
        num_stages: int,
        wgmma_mode: str,
    ) -> torch.Tensor:
        return relu_bat_c_tf32_wgmma(
            A,
            B,
            C,
            BM=BM,
            BN=BN,
            WGMMA_S_N=WGMMA_S_N,
            WGMMA_Y_N=WGMMA_Y_N,
            num_stages=num_stages,
            wgmma_mode=wgmma_mode,
        )

    return relu_bat_c_tf32_wgmma_cuda


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
    
    ref = torch_impl(A, B, C)
    relu_bat_c_fp32_tuned = relu_bat_c_fp32_cuda_launcher(
        tune_trials,
        tune_warmup_ms,
        tune_rep_ms,
    )
    relu_bat_c_tf32_mma_sync_tuned = relu_bat_c_tf32_sync_launcher(
        D,
        tune_trials,
        tune_warmup_ms,
        tune_rep_ms,
    )
    relu_bat_c_wgmma_tuned = None
    if relu_bat_c_tf32_wgmma_available(A):
        relu_bat_c_wgmma_tuned = relu_bat_c_tf32_wgmma_launcher(
            D,
            tune_trials,
            tune_warmup_ms,
            tune_rep_ms,
        )

    fp32_cuda_decision = relu_bat_c_fp32_tuned.resolve_decision(A, B, C)
    fp32_cuda_params = fp32_cuda_decision["params"]
    out_fp32_cuda = relu_bat_c_fused_op(
        A,
        B,
        C,
        fp32_cuda_params["BM"],
        fp32_cuda_params["BK"],
        fp32_cuda_params["num_ms"],
    )
    err_fp32_cuda = (out_fp32_cuda - ref).abs().max().item()

    tf32_mma_sync_decision = relu_bat_c_tf32_mma_sync_tuned.resolve_decision(
        A,
        B,
        C,
    )
    tf32_mma_sync_params = tf32_mma_sync_decision["params"]
    out_tf32_mma_sync = relu_bat_c_tf32_mma_sync(
        A,
        B,
        C,
        **tf32_mma_sync_params,
    )
    err_tf32_mma_sync = (out_tf32_mma_sync - ref).abs().max().item()

    wgmma_decision = None
    wgmma_params = None
    err_wgmma = None
    if relu_bat_c_wgmma_tuned is not None:
        wgmma_decision = relu_bat_c_wgmma_tuned.resolve_decision(A, B, C)
        wgmma_params = wgmma_decision["params"]
        out_wgmma = relu_bat_c_tf32_wgmma(A, B, C, **wgmma_params)
        err_wgmma = (out_wgmma - ref).abs().max().item()


    print(f"[correctness] M={M} N={N} D={D} dtype={dtype}")
    print(
        "  cuda fused FP32 (1D)              "
        f"max_abs_err vs torch: {err_fp32_cuda:.6e}"
    )
    print(f"  cuda fused FP32 (1D)              tuned params: {fp32_cuda_params}")
    print(
        "  cuda fused TF32 MMA sync (1D)     "
        f"max_abs_err vs torch: {err_tf32_mma_sync:.6e}"
    )
    print(
        "  cuda fused TF32 MMA sync (1D)     "
        f"tuned params: {tf32_mma_sync_params}"
    )
    if wgmma_params is not None:
        print(
            "  cuda fused TF32 WGMMA (1D)  "
            f"max_abs_err vs torch: {err_wgmma:.6e}"
        )
        print(f"  cuda fused TF32 WGMMA (1D)  tuned params: {wgmma_params}")
    else:
        print("  cuda fused TF32 WGMMA (1D)  skipped: requires SM90/Hopper")

    def torch_run():
        return torch_impl(A, B, C)

    def fp32_cuda_run():
        return relu_bat_c_fused_op(
            A,
            B,
            C,
            fp32_cuda_params["BM"],
            fp32_cuda_params["BK"],
            fp32_cuda_params["num_ms"],
        )

    def tf32_mma_sync_run():
        return relu_bat_c_tf32_mma_sync(A, B, C, **tf32_mma_sync_params)

    def wgmma_run():
        return relu_bat_c_tf32_wgmma(A, B, C, **wgmma_params)

    torch.cuda.synchronize()

    t_torch   = triton.testing.do_bench(torch_run, warmup=warmup_ms, rep=rep_ms)
    t_fp32_cuda = triton.testing.do_bench(
        fp32_cuda_run,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    t_tf32_mma_sync = triton.testing.do_bench(
        tf32_mma_sync_run,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    t_wgmma = None
    if wgmma_params is not None:
        t_wgmma = triton.testing.do_bench(
            wgmma_run,
            warmup=warmup_ms,
            rep=rep_ms,
        )

    return {
        "M": M, "N": N, "D": D, "dtype": str(dtype).replace("torch.", ""),

        "torch_ms":   float(t_torch),
        "fp32_cuda_ms": float(t_fp32_cuda),
        "fp32_cuda_params": dict(fp32_cuda_params),
        "fp32_cuda_tune_ms": float(fp32_cuda_decision["runtime_ms"]),
        "tf32_mma_sync_ms": float(t_tf32_mma_sync),
        "tf32_mma_sync_params": dict(tf32_mma_sync_params),
        "tf32_mma_sync_tune_ms": float(
            tf32_mma_sync_decision["runtime_ms"]
        ),
        "wgmma_ms": None if t_wgmma is None else float(t_wgmma),
        "wgmma_params": None if wgmma_params is None else dict(wgmma_params),
        "wgmma_tune_ms": (
            None if wgmma_decision is None
            else float(wgmma_decision["runtime_ms"])
        ),
        "torch_TFLOPs":   float(ms_to_tflops(M, N, D, t_torch)),
        "fp32_cuda_TFLOPs": float(ms_to_tflops(M, N, D, t_fp32_cuda)),
        "tf32_mma_sync_TFLOPs": float(
            ms_to_tflops(M, N, D, t_tf32_mma_sync)
        ),
        "wgmma_TFLOPs": (
            None if t_wgmma is None
            else float(ms_to_tflops(M, N, D, t_wgmma))
        ),
        "speedup_fp32_cuda": float(t_torch / t_fp32_cuda),
        "speedup_tf32_mma_sync": float(t_torch / t_tf32_mma_sync),
        "speedup_wgmma": None if t_wgmma is None else float(t_torch / t_wgmma),
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
        choices=tuple(mode for mode in AUTOTUNE_MODES if mode != "fallback"),
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

    torch.manual_seed(0)
    torch.set_float32_matmul_precision('high')
    assert torch.cuda.is_available()

    with kernel_autotune_options(
        mode=args.autotune,
        cache_dir=args.autotune_cache_dir,
        verbose=args.autotune_verbose,
    ):
        r = bench_one(
            M=250000,
            N=2500,
            D=16,
            dtype=torch.float32,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        print_perf_summary(r)
        r = bench_one(
            M=250000,
            N=2500,
            D=32,
            dtype=torch.float32,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        print_perf_summary(r)
        r = bench_one(
            M=250000,
            N=2500,
            D=64,
            dtype=torch.float32,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        print_perf_summary(r)
        r = bench_one(
            M=250000,
            N=2500,
            D=128,
            dtype=torch.float32,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        print_perf_summary(r)
        r = bench_one(
            M=250000,
            N=2500,
            D=256,
            dtype=torch.float32,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        print_perf_summary(r)
    
    
