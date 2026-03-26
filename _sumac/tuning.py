import functools
import inspect
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Optional

import optuna
import torch
import triton


class JsonConfigStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cache.get(key)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = value
            self._save()


def _normalize_for_json(x: Any) -> Any:
    if isinstance(x, torch.device):
        return str(x)
    if isinstance(x, tuple):
        return [_normalize_for_json(v) for v in x]
    if isinstance(x, list):
        return [_normalize_for_json(v) for v in x]
    if isinstance(x, dict):
        return {k: _normalize_for_json(v) for k, v in x.items()}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _stable_key(d: Dict[str, Any]) -> str:
    return json.dumps(_normalize_for_json(d), sort_keys=True)


def _default_device_key(t: torch.Tensor) -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(t.device)
    return {
        "gpu_name": props.name,
        "cc": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "device_index": int(t.device.index if t.device.index is not None else 0),
    }


def _bench_callable(fn: Callable[[], Any], *, warmup: int, rep: int) -> float:
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


@dataclass(frozen=True)
class TuneResult:
    params: Dict[str, Any]
    runtime_ms: float


def autotune_cuda_kernel(
    *,
    configs: Dict[str, list],
    key_fn: Callable[..., Hashable],
    constraint_fn: Optional[Callable[..., bool]] = None,
    validate_fn: Optional[Callable[..., None]] = None,
    cache_path: str = "kernel_autotune_cache.json",
    n_trials: int = 24,
    warmup: int = 25,
    rep: int = 100,
    sampler=None,
    force_env_var: str = "KERNEL_AUTOTUNE_FORCE",
    disable_env_var: str = "KERNEL_AUTOTUNE_DISABLE",
    validate_env_var: str = "KERNEL_AUTOTUNE_VALIDATE",
    verbose_env_var: str = "KERNEL_AUTOTUNE_VERBOSE",
):
    store = JsonConfigStore(cache_path)

    memo: Dict[Hashable, Dict[str, Any]] = {}
    default_params = {k: v[0] for k, v in configs.items()}

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=0)

    disable_default = os.getenv(disable_env_var, "0") == "1"
    force_default = os.getenv(force_env_var, "0") == "1"
    do_validate_default = os.getenv(validate_env_var, "0") == "1"
    verbose_default = os.getenv(verbose_env_var, "0") == "1"

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this autotuned kernel")

            if disable_default:
                return fn(*args, **kwargs, **default_params)

            mem_key = key_fn(*args, **kwargs)
            params = None if force_default else memo.get(mem_key)
            if params is not None:
                return fn(*args, **kwargs, **params)

            disk_key = json.dumps(_normalize_for_json(mem_key), separators=(",", ":"))

            payload = None if force_default else store.get(disk_key)
            if payload is not None:
                params = payload["params"]
                memo[mem_key] = params
                return fn(*args, **kwargs, **params)

            result = _run_study(
                fn=fn,
                args=args,
                kwargs=kwargs,
                configs=configs,
                constraint_fn=constraint_fn,
                validate_fn=validate_fn if do_validate_default else None,
                n_trials=n_trials,
                warmup=warmup,
                rep=rep,
                sampler=sampler,
            )

            params = result.params
            memo[mem_key] = params

            payload = {
                "function": fn.__name__,
                "params": params,
                "runtime_ms": result.runtime_ms,
            }
            store.put(disk_key, payload)

            if verbose_default:
                print(
                    f"[autotune:{fn.__name__}] tuned key={mem_key} "
                    f"params={params} runtime_ms={result.runtime_ms:.4f}"
                )

            return fn(*args, **kwargs, **params)

        def clear_memo() -> None:
            memo.clear()

        wrapper.clear_memo = clear_memo
        return wrapper

    return decorator


def _run_study(
    *,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
    configs: Dict[str, list],
    constraint_fn: Optional[Callable[..., bool]],
    validate_fn: Optional[Callable[..., None]],
    n_trials: int,
    warmup: int,
    rep: int,
    sampler: optuna.samplers.BaseSampler,
) -> TuneResult:
    static_kwargs = dict(kwargs)

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: trial.suggest_categorical(name, values)
            for name, values in configs.items()
        }
        merged = {**static_kwargs, **params}

        if constraint_fn is not None and not constraint_fn(*args, **merged):
            raise optuna.TrialPruned()

        def run():
            return fn(*args, **merged)

        out = run()
        torch.cuda.synchronize()

        if validate_fn is not None:
            validate_fn(output=out, *args, **merged)

        runtime_ms = _bench_callable(
            run,
            warmup=warmup,
            rep=rep,
        )
        trial.set_user_attr("runtime_ms", runtime_ms)
        return runtime_ms

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    return TuneResult(
        params={name: best.params[name] for name in configs},
        runtime_ms=float(best.user_attrs["runtime_ms"]),
    )


def relu_bat_c_reference(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return torch.relu(B @ A.T) @ C

def relu_bat_c_key(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> tuple:
    props = torch.cuda.get_device_properties(A.device)

    N, D = A.shape
    M, _ = B.shape

    return (
        props.major,
        props.minor,
        props.multi_processor_count,
        int(N),
        int(M),
        int(D),
    )

def relu_bat_c_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BK: int,
    num_ms: int,
) -> bool:
    if A.device != B.device or A.device != C.device:
        return False
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        return False
    if A.shape[1] != B.shape[1] or C.shape[1] != A.shape[1]:
        return False
    if not A.is_cuda or not B.is_cuda or not C.is_cuda:
        return False
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        return False
    if not A.is_contiguous() or not B.is_contiguous() or not C.is_contiguous():
        return False

    N, D = A.shape
    M, _ = B.shape

    if BM % 32 != 0:
        return False
    if BK not in (16, 32, 64, 128, 256):
        return False
    # if num_ms not in (1, 2, 4):
    #     return False


    if BM == 384 and num_ms == 4:
        return False

    V = D // 4
    R = D % 4
    if V not in (1, 2, 3, 4):
        return False
    if R not in (0, 1, 2, 3):
        return False

    if M < BM and num_ms > 1:
        return False
    if N <= 32 and BK > 32:
        return False
    if D > 64 and num_ms == 4:
        return False

    return True

def relu_bat_c_validate(
    output: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BK: int,
    num_ms: int,
) -> None:
    del BM, BK, num_ms
    ref = relu_bat_c_reference(A, B, C)
    torch.testing.assert_close(output, ref, rtol=1e-4, atol=1e-4)



# Environment flags:
#   KERNEL_AUTOTUNE_DISABLE=1   -  bypass autotuning, use first config in each list
#   KERNEL_AUTOTUNE_FORCE=1     -  ignore cache and retune
#   KERNEL_AUTOTUNE_VERBOSE=1   -  print tuning/cache diagnostics
#   KERNEL_AUTOTUNE_VALIDATE=1  -  validate correctness of candidate