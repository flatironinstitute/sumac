import torch
import triton
import argparse

from dataclasses import asdict

from sumac.kernels.tuning import (
    grid_size,
    make_choices,
    kernel_autotune_options,
    relu_bat_c_fallback,
    AutotuneReluBatCFP32,
    AutotuneReluBatCTf32Sync,
    AutotuneReluBatCTf32Wgmma,
    relu_bat_c_fp32_tune_config,
    relu_bat_c_tf32_sync_tune_config,
    relu_bat_c_tf32_wgmma_tune_config,
    relu_bat_c_tf32_wgmma_available
)

from sumac.config import AutotuneMode

from sumac.kernels.relu_batc_jit.api import relu_bat_c_fused_op
from sumac.kernels.relu_batc_tf32_jit.api import (
    relu_bat_c_tf32_mma_sync,
    relu_bat_c_tf32_wgmma,
)


def ms_to_tflops(M, N, D, ms):
        flops = M * N * (D + D - 1) + M * D * (N + N - 1)
    
        s = ms * 1e-3
        return flops / s / 1e12


# def flop_per_byte_fused(M, N, D, wordsize):
#     flops = M * N * (D + D - 1) + M * D * (N + N - 1)

#     elems_read = M * D + N * D + N * D
#     elems_written = M * D
    
#     return flops/((elems_read + elems_written) * wordsize) 


# def flop_per_byte_2kernels(M, N, D, wordsize):
#     flops = M * N * (D + D - 1) + M * D * (N + N - 1)
    
#     elems_read_kernel1 = M * D + N * D
#     elems_written_kernel1 = M * N
    
#     elems_read_kernel2 = M * N + N * D
#     elems_written_kernel2 = M * D

#     return flops/((elems_read_kernel1 + elems_read_kernel2 + elems_written_kernel1 + elems_written_kernel2) * wordsize)


# def perf_roofline(FLOP_per_Byte, BW_GBs, peak_TFLOP):
#         return min(peak_TFLOP,(BW_GBs * FLOP_per_Byte)/1e3) #/1e3 to get to TFLOP/s


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
    n_trials_fp32 = max(tune_trials, grid_size(make_choices(relu_bat_c_fp32_tune_config)))
    relu_bat_c_fp32_tuned = AutotuneReluBatCFP32(
        configs = relu_bat_c_fp32_tune_config,
        cache_path = "relu_bat_c_jit_autotune.json",
        n_trials = n_trials_fp32,
        warmup = tune_warmup_ms,
        rep = tune_rep_ms
    )
    sync_config = relu_bat_c_tf32_sync_tune_config(D)
    relu_bat_c_tf32_mma_sync_tuned = AutotuneReluBatCTf32Sync(
        configs = sync_config,
        cache_path = "relu_bat_c_tf32_mma_autotune.json",
        n_trials = tune_trials,
        warmup = tune_warmup_ms,
        rep = tune_rep_ms
    )
    relu_bat_c_wgmma_tuned = None
    if relu_bat_c_tf32_wgmma_available(A):
        tune_config = relu_bat_c_tf32_wgmma_tune_config(D)
        relu_bat_c_wgmma_tuned = AutotuneReluBatCTf32Wgmma(
            configs = tune_config,
            cache_path = "relu_bat_c_tf32_wgmma_mode_autotune.json",
            n_trials = tune_trials,
            warmup = tune_warmup_ms,
            rep = tune_rep_ms
        )

    relu_bat_c_fp32_tuned.resolve_decision((A, B, C))
    fp32_cuda_params = relu_bat_c_fp32_tuned.decision_config
    assert fp32_cuda_params is not None
    fp32_params_dict = asdict(fp32_cuda_params)
    out_fp32_cuda = relu_bat_c_fused_op(A, B, C, **fp32_params_dict)
    err_fp32_cuda = (out_fp32_cuda - ref).abs().max().item()

    relu_bat_c_tf32_mma_sync_tuned.resolve_decision((A, B, C))
    sync_params = relu_bat_c_tf32_mma_sync_tuned.decision_config
    assert sync_params is not None
    # sync_params_dict = {f.name: getattr(sync_params, f.name)[0] for f in fields(sync_params)}
    sync_params_dict = asdict(sync_params)
    out_tf32_mma_sync = relu_bat_c_tf32_mma_sync(A, B, C, **sync_params_dict)
    err_tf32_mma_sync = (out_tf32_mma_sync - ref).abs().max().item()

    wgmma_params = None
    wgmma_params_dict = {}
    err_wgmma = None
    if relu_bat_c_wgmma_tuned is not None:
        relu_bat_c_wgmma_tuned.resolve_decision((A, B, C))
        wgmma_params = relu_bat_c_wgmma_tuned.decision_config
        assert wgmma_params is not None
        # wgmma_params_dict = {
        #     f.name: getattr(wgmma_params, f.name)[0] for f in fields(wgmma_params)
        # }
        wgmma_params_dict = asdict(wgmma_params)
        out_wgmma = relu_bat_c_tf32_wgmma(A, B, C, **wgmma_params_dict)
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
        f"tuned params: {sync_params_dict}"
    )
    if wgmma_params is not None:
        print(
            "  cuda fused TF32 WGMMA (1D)  "
            f"max_abs_err vs torch: {err_wgmma:.6e}"
        )
        print(f"  cuda fused TF32 WGMMA (1D)  tuned params: {wgmma_params_dict}")
    else:
        print("  cuda fused TF32 WGMMA (1D)  skipped: requires SM90/Hopper")

    def torch_run():
        return relu_bat_c_fallback(A, B, C)

    def fp32_cuda_run():
        return relu_bat_c_fused_op(A, B, C, **fp32_params_dict)

    def tf32_mma_sync_run():
        return relu_bat_c_tf32_mma_sync(A, B, C, **sync_params_dict)

    def wgmma_run():
        return relu_bat_c_tf32_wgmma(A, B, C, **wgmma_params_dict)

    torch.cuda.synchronize()

    t_torch   = float(
        triton.testing.do_bench(torch_run, warmup=warmup_ms, rep=rep_ms)  #type: ignore
    )
    t_fp32_cuda = float( 
        triton.testing.do_bench( #type: ignore
            fp32_cuda_run,
            warmup=warmup_ms,
            rep=rep_ms,
        )
    )
    t_tf32_mma_sync = float (
        triton.testing.do_bench(  #type: ignore
            tf32_mma_sync_run,
            warmup=warmup_ms,
            rep=rep_ms,
        )
    )
    t_wgmma = None
    if wgmma_params is not None:
        t_wgmma = float(
            triton.testing.do_bench(   #type: ignore
                wgmma_run,
                warmup=warmup_ms,
                rep=rep_ms,
            )
        )

    return {
        "M": M, "N": N, "D": D, "dtype": str(dtype).replace("torch.", ""),

        "torch_ms":   t_torch,
        "fp32_cuda_ms": t_fp32_cuda,
        "fp32_cuda_params": fp32_params_dict,
        "fp32_cuda_tune_ms": relu_bat_c_fp32_tuned.decision_runtime_ms,
        "tf32_mma_sync_ms": t_tf32_mma_sync,
        "tf32_mma_sync_params": sync_params_dict,
        "tf32_mma_sync_tune_ms": relu_bat_c_tf32_mma_sync_tuned.decision_runtime_ms,
        "wgmma_ms": None if t_wgmma is None else t_wgmma,
        "wgmma_params": None if wgmma_params is None else wgmma_params_dict,
        "wgmma_tune_ms": (
            None if relu_bat_c_wgmma_tuned is None
            else relu_bat_c_wgmma_tuned.decision_runtime_ms
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
            print_perf_summary(r)
