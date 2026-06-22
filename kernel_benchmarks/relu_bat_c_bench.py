import torch
import triton
import argparse
import optuna

from functools import lru_cache

from _sumac.tuning import *
from relu_batc_jit.api import relu_bat_c_fused_op
from relu_batc_tf32_jit.jit_kernel_tf32_sync import launch_relu_batc_mma_sync_tf32
from relu_batc_tf32_jit.jit_kernel_tf32_wgmma import launch_relu_bat_c_wgmma_tf32_tma

#@torch.compile(mode='max-autotune-no-cudagraphs')
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


def relu_bat_c_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BK: int,
    num_ms: int,
) -> bool:
    if A.shape[1] >= 32 and num_ms > 2:
        return False

    if A.shape[1] >= 64 and num_ms > 1:  
        return False  
    
    props = torch.cuda.get_device_properties(torch.cuda.current_device)

    if props.shared_memory_per_block < 8 * BK * A.shape[1]:
        return False

    return True

@lru_cache(maxsize=None)
def relu_bat_c_fp32_cuda_launcher(
    n_trials: int,
    warmup: int,
    rep: int,
):
    tune_config = {
            "BM": [32, 64, 128, 256],
            "BK": [16, 32, 64],
            "num_ms": [1, 2, 4, 6],
        }
    n_trials = max(n_trials, _grid_size(tune_config))

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_constraints,
        cache_path="relu_bat_c_jit_autotune.json",
        n_trials=n_trials,
        warmup=warmup,
        rep=rep,
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


def _neighbor_values(base, allowed):
    idx = allowed.index(base)
    values = [base]
    if idx > 0:
        values.append(allowed[idx - 1])
    if idx + 1 < len(allowed):
        values.append(allowed[idx + 1])
    return values


def _grid_size(config: dict) -> int:
    size = 1
    for values in config.values():
        size *= len(values)
    return size


def _max_dynamic_smem_bytes(props) -> int:
    candidates = [
        int(getattr(props, "shared_memory_per_block", 0) or 0),
        int(getattr(props, "shared_memory_per_block_optin", 0) or 0),
    ]

    if getattr(props, "major", 0) == 9:
        candidates.append(227 * 1024)

    return max(candidates)


def relu_bat_c_tf32_sync_baseline(D: int) -> dict:
    params = {
        "BM": 256,
        "BN": 32,
        "M_TILES": 4,
        "num_stages": 2,
    }

    if D == 32:
        params.update(
            BM=256,
            BN=32,
            M_TILES=4,
            num_stages=2,
        )
    if D == 64:
        params.update(
            BM=256,
            BN=64,
            M_TILES=2,
            num_stages=2,
        )
    if D == 128:
        params.update(
            BM=128,
            BN=16,
            M_TILES=2,
            num_stages=2,
        )
    if D == 256:
        params.update(
            BM=128,
            BN=8,
            M_TILES=1,
            num_stages=1,
        )

    return params


def relu_bat_c_tf32_sync_tune_config(D: int) -> dict:
    baseline = relu_bat_c_tf32_sync_baseline(D)

    return {
        "BM": _neighbor_values(baseline["BM"], [64, 128, 256]),
        "BN": _neighbor_values(baseline["BN"], [8, 16, 32, 64, 128]),
        "M_TILES": _neighbor_values(baseline["M_TILES"], [1, 2, 4]),
        "num_stages": _neighbor_values(baseline["num_stages"], [1, 2, 3]),
    }


def relu_bat_c_tf32_sync_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
    num_stages: int,
) -> bool:
    _, D = A.shape

    if D % 8 != 0:
        return False
    if BN % 8 != 0:
        return False
    if num_stages < 1:
        return False

    warp_m_rows = M_TILES * 16
    if BM % warp_m_rows != 0:
        return False

    compute_warps = BM // warp_m_rows
    if compute_warps < 1 or compute_warps > 8:
        return False

    props = torch.cuda.get_device_properties(A.device)
    max_smem = getattr(
        props,
        "shared_memory_per_block_optin",
        props.shared_memory_per_block,
    )
    smem_bytes = 2 * num_stages * BN * D * 4
    smem_bytes += 127

    if smem_bytes > max_smem:
        return False

    return True


@lru_cache(maxsize=None)
def relu_bat_c_tf32_sync_launcher(
    D: int,
    n_trials: int,
    warmup: int,
    rep: int,
):
    tune_config = relu_bat_c_tf32_sync_tune_config(D)

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_tf32_sync_constraints,
        cache_path="relu_bat_c_tf32_mma_autotune.json",
        n_trials=n_trials,
        warmup=warmup,
        rep=rep,
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
        return launch_relu_batc_mma_sync_tf32(
            A,
            B,
            C,
            BM=BM,
            BN=BN,
            M_TILES=M_TILES,
            num_stages=num_stages,
        )

    return relu_bat_c_tf32_sync_cuda


def relu_bat_c_tf32_wgmma_baseline(D: int) -> dict:
    params = {
        "BM": 256,
        "BN": 32,
        "WGMMA_S_N": 16,
        "WGMMA_Y_N": 16,
        "num_stages": 2,
        "wgmma_mode": "RS",
    }

    if D == 32:
        params.update(
            BM=256,
            BN=64,
            WGMMA_S_N=32,
            WGMMA_Y_N=32,
        )
    if D == 64:
        params.update(
            BM=256,
            BN=128,
            WGMMA_S_N=128,
            WGMMA_Y_N=64,
        )
    if D == 128:
        params.update(
            BM=128,
            BN=128,
            WGMMA_S_N=128,
            WGMMA_Y_N=128,
        )
    if D == 256:
        params.update(
            BM=128,
            BN=128,
            WGMMA_S_N=128,
            WGMMA_Y_N=128,
        )

    return params


def relu_bat_c_tf32_wgmma_tune_config(D: int) -> dict:
    if D == 16:
        return {
            "BM": [256, 320],
            "BN": [64, 128],
            "WGMMA_S_N": [64],
            "WGMMA_Y_N": [16],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }

    if D == 32:
        return {
            "BM": [192, 256, 320],
            "BN": [64, 128],
            "WGMMA_S_N": [64],
            "WGMMA_Y_N": [32],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }

    if D == 64:
        return {
            "BM": [128, 192, 256],
            "BN": [64, 128],
            "WGMMA_S_N": [64],
            "WGMMA_Y_N": [64],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }

    if D == 128:
        return {
            "BM": [128, 192, 256],
            "BN": [32, 64, 128],
            "WGMMA_S_N": [32, 64],
            "WGMMA_Y_N": [64, 128],
            "num_stages": [2],
            "wgmma_mode": ["SS"],
        }

    if D == 256:
        return {
            "BM": [64, 128, 192],
            "BN": [16, 32, 64],
            "WGMMA_S_N": [16, 32],
            "WGMMA_Y_N": [64, 128],
            "num_stages": [2],
            "wgmma_mode": ["SS"],
        }

    wgmma_n_values = [16, 32, 64, 128]
    y_shapes = [n for n in wgmma_n_values if D % n == 0]
    if not y_shapes:
        y_shapes = [16]

    return {
        "BM": [64, 128, 192, 256, 320],
        "BN": [16, 32, 64, 128, 256],
        "WGMMA_S_N": wgmma_n_values,
        "WGMMA_Y_N": y_shapes,
        "num_stages": [1, 2],
        "wgmma_mode": ["RS", "SS"],
    }


def relu_bat_c_tf32_wgmma_available(A: torch.Tensor) -> bool:
    props = torch.cuda.get_device_properties(A.device)
    return props.major == 9


def relu_bat_c_tf32_wgmma_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    WGMMA_S_N: int,
    WGMMA_Y_N: int,
    num_stages: int,
    wgmma_mode: str,
) -> bool:
    _, D = A.shape

    props = torch.cuda.get_device_properties(A.device)
    if props.major != 9:
        return False

    if WGMMA_S_N not in (16, 32, 64, 128):
        return False
    if WGMMA_Y_N not in (16, 32, 64, 128):
        return False
    if BN % WGMMA_S_N != 0:
        return False
    if D % WGMMA_Y_N != 0:
        return False
    if num_stages not in (1, 2, 3):
        return False
    if wgmma_mode not in ("RS", "SS"):
        return False

    if BM % 64 != 0:
        return False

    compute_warpgroups = BM // 64
    if compute_warpgroups < 1:
        return False

    threads_per_block = (compute_warpgroups + 1) * 128
    max_threads = getattr(props, "max_threads_per_block", 1024)
    if threads_per_block > max_threads:
        return False

    max_smem = _max_dynamic_smem_bytes(props)
    smem_elems = num_stages * 2 * BN * D
    if wgmma_mode == "SS":
        smem_elems += BM * D
    smem_bytes = smem_elems * 4 + 127

    if smem_bytes > max_smem:
        return False

    return True


@lru_cache(maxsize=None)
def relu_bat_c_tf32_wgmma_launcher(
    D: int,
    n_trials: int,
    warmup: int,
    rep: int,
):
    tune_config = relu_bat_c_tf32_wgmma_tune_config(D)
    n_trials = max(n_trials, _grid_size(tune_config))

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_tf32_wgmma_constraints,
        cache_path="relu_bat_c_tf32_wgmma_mode_autotune.json",
        n_trials=n_trials,
        warmup=warmup,
        rep=rep,
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
        return launch_relu_bat_c_wgmma_tf32_tma(
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
    wm_iters=0,
    iters=1,
    tune_trials=200,
    tune_warmup_iters=1,
    tune_iters=5,
):
    A = torch.randn((N, D), device=device, dtype=dtype)
    B = torch.randn((M, D), device=device, dtype=dtype)
    C = torch.randn((N, D), device=device, dtype=dtype)
    
    ref = torch_impl(A, B, C)
    relu_bat_c_fp32_tuned = relu_bat_c_fp32_cuda_launcher(
        tune_trials,
        tune_warmup_iters,
        tune_iters,
    )
    relu_bat_c_tf32_mma_sync_tuned = relu_bat_c_tf32_sync_launcher(
        D,
        tune_trials,
        tune_warmup_iters,
        tune_iters,
    )
    relu_bat_c_wgmma_tuned = None
    if relu_bat_c_tf32_wgmma_available(A):
        relu_bat_c_wgmma_tuned = relu_bat_c_tf32_wgmma_launcher(
            D,
            tune_trials,
            tune_warmup_iters,
            tune_iters,
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
    out_tf32_mma_sync = launch_relu_batc_mma_sync_tf32(
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
        out_wgmma = launch_relu_bat_c_wgmma_tf32_tma(A, B, C, **wgmma_params)
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
        return launch_relu_batc_mma_sync_tf32(A, B, C, **tf32_mma_sync_params)

    def wgmma_run():
        return launch_relu_bat_c_wgmma_tf32_tma(A, B, C, **wgmma_params)

    torch.cuda.synchronize()

    t_torch   = triton.testing.do_bench(torch_run, warmup=wm_iters, rep=iters)
    t_fp32_cuda = triton.testing.do_bench(
        fp32_cuda_run,
        warmup=wm_iters,
        rep=iters,
    )
    t_tf32_mma_sync = triton.testing.do_bench(
        tf32_mma_sync_run,
        warmup=wm_iters,
        rep=iters,
    )
    t_wgmma = None
    if wgmma_params is not None:
        t_wgmma = triton.testing.do_bench(wgmma_run, warmup=wm_iters, rep=iters)

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
    parser.add_argument("--warmup-iters", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of benchmark iterations")
    args = parser.parse_args()

    torch.manual_seed(0)
    assert torch.cuda.is_available()

    r = bench_one(
        M=288768,
        N=1408,
        D=16,
        dtype=torch.float32,
        wm_iters=args.warmup_iters,
        iters=args.iters,
    )
    print_perf_summary(r)
    r = bench_one(
        M=288768,
        N=1408,
        D=32,
        dtype=torch.float32,
        wm_iters=args.warmup_iters,
        iters=args.iters,
    )
    print_perf_summary(r)
    r = bench_one(
        M=288768,
        N=1408,
        D=64,
        dtype=torch.float32,
        wm_iters=args.warmup_iters,
        iters=args.iters,
    )
    print_perf_summary(r)
    r = bench_one(
        M=288768,
        N=1408,
        D=128,
        dtype=torch.float32,
        wm_iters=args.warmup_iters,
        iters=args.iters,
    )
    print_perf_summary(r)
    r = bench_one(
        M=288768, 
        N=1408, 
        D=256, 
        dtype=torch.float32, 
        wm_iters=args.warmup_iters, 
        iters=args.iters,
    )
    print_perf_summary(r)
    
    
