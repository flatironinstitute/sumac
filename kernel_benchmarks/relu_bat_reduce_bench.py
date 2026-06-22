import argparse
from functools import lru_cache

import optuna
import torch
import triton

from _sumac.tuning import autotune_cuda_kernel, relu_bat_reduce_key
from relu_bat_reduce_jit.custom_op import relu_bat_reduce_fused_op
from relu_bat_reduce_tf32_jit.jit_kernel_tf32_sync import (
    launch_relu_bat_reduce_mma_sync_tf32,
)
from relu_bat_reduce_tf32_jit.jit_kernel_tf32_wgmma import (
    launch_relu_bat_reduce_wgmma_tf32_tma,
)


def torch_impl(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    S = torch.relu(B @ A.T)
    return S.sum(), (S * S).sum()


def torch_ref_impl(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    S = torch.relu(B @ A.T)
    return S.sum(dtype=torch.float64), (S * S).sum(dtype=torch.float64)


def ms_to_tflops(M: int, N: int, D: int, ms: float) -> float:
    flops = M * N * (D + D - 1)
    return flops / (ms * 1e-3) / 1e12


def _grid_size(config: dict) -> int:
    size = 1
    for values in config.values():
        size *= len(values)
    return size


def _neighbor_values(base, allowed):
    idx = allowed.index(base)
    values = [base]
    if idx > 0:
        values.append(allowed[idx - 1])
    if idx + 1 < len(allowed):
        values.append(allowed[idx + 1])
    return values


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _max_dynamic_smem_bytes(props) -> int:
    candidates = [
        int(getattr(props, "shared_memory_per_block", 0) or 0),
        int(getattr(props, "shared_memory_per_block_optin", 0) or 0),
    ]
    if getattr(props, "major", 0) == 9:
        candidates.append(227 * 1024)
    return max(candidates)


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
        (
            "cuda fused TF32 MMA sync",
            result["tf32_mma_sync_ms"],
            result["tf32_mma_sync_TFLOPs"],
            result["speedup_tf32_mma_sync"],
        ),
    ]
    if result["wgmma_ms"] is not None:
        rows.append(
            (
                "cuda fused TF32 WGMMA",
                result["wgmma_ms"],
                result["wgmma_TFLOPs"],
                result["speedup_wgmma"],
            )
        )

    print("[performance]")
    print(f"  {'implementation':<28} {'time [ms]':>10} {'TFLOP/s':>10} {'speedup':>9}")
    for name, ms, tflops, speedup in rows:
        print(f"  {name:<28} {ms:10.4f} {tflops:10.2f} {speedup:8.2f}x")
    print()


def relu_bat_reduce_fp32_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BK: int,
    num_ms: int,
) -> bool:
    _, D = A.shape
    props = torch.cuda.get_device_properties(A.device)

    if D >= 32 and num_ms > 4:
        return False
    if D >= 64 and num_ms > 2:
        return False
    if not _is_power_of_two(BM):
        return False
    if BM > getattr(props, "max_threads_per_block", 1024):
        return False

    return 4 * BK * D + 2 * BM * 8 <= props.shared_memory_per_block


@lru_cache(maxsize=None)
def relu_bat_reduce_fp32_launcher(
    n_trials: int,
    warmup: int,
    rep: int,
):
    tune_config = {
        "BM": [32, 64, 128, 256],
        "BK": [16, 32, 64, 128],
        "num_ms": [1, 2, 4, 6],
    }
    n_trials = max(n_trials, _grid_size(tune_config))

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_reduce_key,
        constraint_fn=relu_bat_reduce_fp32_constraints,
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


def relu_bat_reduce_tf32_sync_baseline(D: int) -> dict:
    params = {
        "BM": 256,
        "BN": 32,
        "M_TILES": 4,
        "num_stages": 1,
    }
    if D == 64:
        params.update(BM=256, BN=64, M_TILES=2, num_stages=1)
    if D == 128:
        params.update(BM=128, BN=32, M_TILES=2, num_stages=1)
    if D == 256:
        params.update(BM=128, BN=16, M_TILES=1, num_stages=1)
    return params


def relu_bat_reduce_tf32_sync_tune_config(D: int) -> dict:
    if D <= 32:
        return {
            "BM": [128, 256],
            "BN": [32, 64, 128],
            "M_TILES": [1, 2, 4],
            "num_stages": [1],
        }
    if D == 64:
        return {
            "BM": [128, 256],
            "BN": [32, 64, 128],
            "M_TILES": [1, 2],
            "num_stages": [1],
        }
    if D == 128:
        return {
            "BM": [64, 128, 256],
            "BN": [16, 32, 64],
            "M_TILES": [1, 2],
            "num_stages": [1],
        }
    if D == 256:
        return {
            "BM": [64, 128],
            "BN": [8, 16, 32],
            "M_TILES": [1],
            "num_stages": [1],
        }

    baseline = relu_bat_reduce_tf32_sync_baseline(D)
    return {
        "BM": _neighbor_values(baseline["BM"], [64, 128, 256]),
        "BN": _neighbor_values(baseline["BN"], [8, 16, 32, 64, 128]),
        "M_TILES": _neighbor_values(baseline["M_TILES"], [1, 2, 4]),
        "num_stages": [1],
    }


def relu_bat_reduce_tf32_sync_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
    num_stages: int,
) -> bool:
    _, D = A.shape
    props = torch.cuda.get_device_properties(A.device)

    if props.major < 8:
        return False
    if D % 8 != 0:
        return False
    if BN % 8 != 0:
        return False
    if num_stages < 1:
        return False
    if num_stages != 1:
        return False

    warp_m_rows = M_TILES * 16
    if BM % warp_m_rows != 0:
        return False

    compute_warps = BM // warp_m_rows
    if compute_warps < 1 or compute_warps > 8:
        return False

    threads_per_block = compute_warps * 32
    if not _is_power_of_two(threads_per_block):
        return False

    n_tiles = BN // 8
    k_tiles = D // 8
    s_regs_per_thread = M_TILES * n_tiles * 4
    b_regs_per_thread = M_TILES * k_tiles * 4
    if s_regs_per_thread + b_regs_per_thread > 160:
        return False

    smem_bytes = num_stages * BN * D * 4
    smem_bytes += 2 * threads_per_block * 8
    smem_bytes += 255
    return smem_bytes <= _max_dynamic_smem_bytes(props)


@lru_cache(maxsize=None)
def relu_bat_reduce_tf32_sync_launcher(
    D: int,
    n_trials: int,
    warmup: int,
    rep: int,
):
    tune_config = relu_bat_reduce_tf32_sync_tune_config(D)
    n_trials = max(n_trials, _grid_size(tune_config))

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_reduce_key,
        constraint_fn=relu_bat_reduce_tf32_sync_constraints,
        cache_path="relu_bat_reduce_tf32_mma_kahan_sum2_autotune.json",
        n_trials=n_trials,
        warmup=warmup,
        rep=rep,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_bat_reduce_tf32_sync_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        BM: int,
        BN: int,
        M_TILES: int,
        num_stages: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return launch_relu_bat_reduce_mma_sync_tf32(
            A,
            B,
            BM=BM,
            BN=BN,
            M_TILES=M_TILES,
            num_stages=num_stages,
        )

    return relu_bat_reduce_tf32_sync_cuda


def relu_bat_reduce_tf32_wgmma_tune_config(D: int) -> dict:
    if D == 16:
        return {
            "BM": [256],
            "BN": [64, 128],
            "WGMMA_S_N": [64],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }
    if D == 32:
        return {
            "BM": [256],
            "BN": [64, 128],
            "WGMMA_S_N": [64],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }
    if D == 64:
        return {
            "BM": [128, 256],
            "BN": [64, 128],
            "WGMMA_S_N": [64],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }
    if D == 128:
        return {
            "BM": [128, 256],
            "BN": [32, 64, 128],
            "WGMMA_S_N": [32, 64],
            "num_stages": [2],
            "wgmma_mode": ["RS", "SS"],
        }
    if D == 256:
        return {
            "BM": [64, 128],
            "BN": [16, 32, 64],
            "WGMMA_S_N": [16, 32],
            "num_stages": [2],
            "wgmma_mode": ["SS"],
        }
    return {
        "BM": [64, 128, 256],
        "BN": [16, 32, 64, 128, 256],
        "WGMMA_S_N": [16, 32, 64, 128],
        "num_stages": [1, 2],
        "wgmma_mode": ["RS", "SS"],
    }


def relu_bat_reduce_tf32_wgmma_available(A: torch.Tensor) -> bool:
    props = torch.cuda.get_device_properties(A.device)
    return props.major == 9


def relu_bat_reduce_tf32_wgmma_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    BM: int,
    BN: int,
    WGMMA_S_N: int,
    num_stages: int,
    wgmma_mode: str,
) -> bool:
    _, D = A.shape
    props = torch.cuda.get_device_properties(A.device)

    if props.major != 9:
        return False
    if D % 8 != 0:
        return False
    if WGMMA_S_N not in (16, 32, 64, 128):
        return False
    if BN % WGMMA_S_N != 0:
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

    compute_threads = compute_warpgroups * 128
    if not _is_power_of_two(compute_threads):
        return False

    threads_per_block = (compute_warpgroups + 1) * 128
    if threads_per_block > getattr(props, "max_threads_per_block", 1024):
        return False

    smem_elems = num_stages * BN * D
    if wgmma_mode == "SS":
        smem_elems += BM * D
    smem_bytes = smem_elems * 4
    smem_bytes += 2 * compute_threads * 8
    smem_bytes += 255
    return smem_bytes <= _max_dynamic_smem_bytes(props)


@lru_cache(maxsize=None)
def relu_bat_reduce_tf32_wgmma_launcher(
    D: int,
    n_trials: int,
    warmup: int,
    rep: int,
):
    tune_config = relu_bat_reduce_tf32_wgmma_tune_config(D)
    n_trials = max(n_trials, _grid_size(tune_config))

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_reduce_key,
        constraint_fn=relu_bat_reduce_tf32_wgmma_constraints,
        cache_path="relu_bat_reduce_tf32_wgmma_kahan_sum2_autotune.json",
        n_trials=n_trials,
        warmup=warmup,
        rep=rep,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_bat_reduce_tf32_wgmma_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        BM: int,
        BN: int,
        WGMMA_S_N: int,
        num_stages: int,
        wgmma_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return launch_relu_bat_reduce_wgmma_tf32_tma(
            A,
            B,
            BM=BM,
            BN=BN,
            WGMMA_S_N=WGMMA_S_N,
            num_stages=num_stages,
            wgmma_mode=wgmma_mode,
        )

    return relu_bat_reduce_tf32_wgmma_cuda


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

    ref = torch_ref_impl(A, B)

    fp32_tuned = relu_bat_reduce_fp32_launcher(
        tune_trials,
        tune_warmup_iters,
        tune_iters,
    )
    tf32_mma_sync_tuned = relu_bat_reduce_tf32_sync_launcher(
        D,
        tune_trials,
        tune_warmup_iters,
        tune_iters,
    )
    tf32_wgmma_tuned = None
    if relu_bat_reduce_tf32_wgmma_available(A):
        tf32_wgmma_tuned = relu_bat_reduce_tf32_wgmma_launcher(
            D,
            tune_trials,
            tune_warmup_iters,
            tune_iters,
        )

    fp32_decision = fp32_tuned.resolve_decision(A, B)
    fp32_params = fp32_decision["params"]
    _require_valid_params(
        "cuda fused FP32",
        relu_bat_reduce_fp32_constraints,
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

    tf32_mma_sync_decision = tf32_mma_sync_tuned.resolve_decision(A, B)
    tf32_mma_sync_params = tf32_mma_sync_decision["params"]
    _require_valid_params(
        "cuda fused TF32 MMA sync",
        relu_bat_reduce_tf32_sync_constraints,
        (A, B),
        tf32_mma_sync_params,
    )
    out_tf32_mma_sync = launch_relu_bat_reduce_mma_sync_tf32(
        A,
        B,
        **tf32_mma_sync_params,
    )
    err_tf32_mma_sync = _abs_errs(out_tf32_mma_sync, ref)

    wgmma_decision = None
    wgmma_params = None
    err_wgmma = None
    if tf32_wgmma_tuned is not None:
        wgmma_decision = tf32_wgmma_tuned.resolve_decision(A, B)
        wgmma_params = wgmma_decision["params"]
        _require_valid_params(
            "cuda fused TF32 WGMMA",
            relu_bat_reduce_tf32_wgmma_constraints,
            (A, B),
            wgmma_params,
        )
        out_wgmma = launch_relu_bat_reduce_wgmma_tf32_tma(A, B, **wgmma_params)
        err_wgmma = _abs_errs(out_wgmma, ref)

    print(f"[correctness] M={M} N={N} D={D} dtype={dtype}")
    print(
        "  cuda fused FP32                 "
        f"{_format_abs_errs(err_fp32)}"
    )
    print(f"  cuda fused FP32                 tuned params: {fp32_params}")
    print(
        "  cuda fused TF32 MMA sync        "
        f"{_format_abs_errs(err_tf32_mma_sync)}"
    )
    print(
        "  cuda fused TF32 MMA sync        "
        f"tuned params: {tf32_mma_sync_params}"
    )
    if wgmma_params is not None:
        print(
            "  cuda fused TF32 WGMMA           "
            f"{_format_abs_errs(err_wgmma)}"
        )
        print(f"  cuda fused TF32 WGMMA           tuned params: {wgmma_params}")
    else:
        print("  cuda fused TF32 WGMMA           skipped: requires SM90/Hopper")

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

    def tf32_mma_sync_run():
        return launch_relu_bat_reduce_mma_sync_tf32(
            A,
            B,
            **tf32_mma_sync_params,
        )

    def wgmma_run():
        return launch_relu_bat_reduce_wgmma_tf32_tma(A, B, **wgmma_params)

    torch.cuda.synchronize()

    t_torch = triton.testing.do_bench(torch_run, warmup=wm_iters, rep=iters)
    t_fp32 = triton.testing.do_bench(fp32_run, warmup=wm_iters, rep=iters)
    t_tf32_mma_sync = triton.testing.do_bench(
        tf32_mma_sync_run,
        warmup=wm_iters,
        rep=iters,
    )
    t_wgmma = None
    if wgmma_params is not None:
        t_wgmma = triton.testing.do_bench(wgmma_run, warmup=wm_iters, rep=iters)

    return {
        "M": M,
        "N": N,
        "D": D,
        "dtype": str(dtype).replace("torch.", ""),
        "torch_ms": float(t_torch),
        "fp32_cuda_ms": float(t_fp32),
        "fp32_cuda_params": dict(fp32_params),
        "fp32_cuda_tune_ms": float(fp32_decision["runtime_ms"]),
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
        "torch_TFLOPs": float(ms_to_tflops(M, N, D, t_torch)),
        "fp32_cuda_TFLOPs": float(ms_to_tflops(M, N, D, t_fp32)),
        "tf32_mma_sync_TFLOPs": float(
            ms_to_tflops(M, N, D, t_tf32_mma_sync)
        ),
        "wgmma_TFLOPs": (
            None if t_wgmma is None
            else float(ms_to_tflops(M, N, D, t_wgmma))
        ),
        "speedup_fp32_cuda": float(t_torch / t_fp32),
        "speedup_tf32_mma_sync": float(t_torch / t_tf32_mma_sync),
        "speedup_wgmma": None if t_wgmma is None else float(t_torch / t_wgmma),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="sum(ReLU(B A.T)) TF32 reduce kernel benchmark"
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

    for D in (64,):
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
