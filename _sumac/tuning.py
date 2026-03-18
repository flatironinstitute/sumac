import functools
import inspect
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
    key_fn: Callable[..., Dict[str, Any]],
    constraint_fn: Optional[Callable[..., bool]] = None,
    validate_fn: Optional[Callable[..., None]] = None,
    cache_path: str = "kernel_autotune_cache.json",
    n_trials: int = 24,
    warmup: int = 25,
    rep: int = 100,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    force_env_var: str = "KERNEL_AUTOTUNE_FORCE",
    disable_env_var: str = "KERNEL_AUTOTUNE_DISABLE",
    validate_env_var: str = "KERNEL_AUTOTUNE_VALIDATE",
    verbose_env_var: str = "KERNEL_AUTOTUNE_VERBOSE",
):
    """
    Decorate a kernel launcher function whose signature looks like:

        fn(A, B, *, BM, BK, num_stages, num_ms)

    The decorated function can then be called as:

        y = fn(A, B)

    and the decorator will choose/tune the kernel parameters.
    """
    store = JsonConfigStore(cache_path)
    memo: Dict[str, Dict[str, Any]] = {}
    memo_lock = threading.Lock()
    tune_locks: Dict[str, threading.Lock] = {}
    tune_locks_guard = threading.Lock()

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=0)

    def get_tune_lock(cache_key: str) -> threading.Lock:
        with tune_locks_guard:
            if cache_key not in tune_locks:
                tune_locks[cache_key] = threading.Lock()
            return tune_locks[cache_key]

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this autotuned kernel")

            disable = os.environ.get(disable_env_var, "0") == "1"
            force = os.environ.get(force_env_var, "0") == "1"
            do_validate = os.environ.get(validate_env_var, "0") == "1"
            verbose = os.environ.get(verbose_env_var, "0") == "1"

            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            if disable:
                default_params = {k: v[0] for k, v in configs.items()}
                if verbose:
                    print(f"[autotune:{fn.__name__}] disabled, using defaults {default_params}")
                return fn(*args, **kwargs, **default_params)

            key_dict = key_fn(**bound.arguments)
            cache_key = _stable_key(key_dict)

            with memo_lock:
                cached = None if force else memo.get(cache_key)

            if cached is None and not force:
                cached = store.get(cache_key)
                if cached is not None:
                    with memo_lock:
                        memo[cache_key] = cached

            if cached is None:
                lock = get_tune_lock(cache_key)
                with lock:
                    with memo_lock:
                        cached = None if force else memo.get(cache_key)
                    if cached is None and not force:
                        cached = store.get(cache_key)
                        if cached is not None:
                            with memo_lock:
                                memo[cache_key] = cached

                    if cached is None:
                        result = _run_study(
                            fn=fn,
                            bound=bound,
                            configs=configs,
                            constraint_fn=constraint_fn,
                            validate_fn=validate_fn if do_validate else None,
                            n_trials=n_trials,
                            warmup=warmup,
                            rep=rep,
                            sampler=sampler,
                        )
                        payload = {
                            "function": fn.__name__,
                            "key": key_dict,
                            "params": result.params,
                            "runtime_ms": result.runtime_ms,
                        }
                        store.put(cache_key, payload)
                        with memo_lock:
                            memo[cache_key] = payload
                        cached = payload
                        if verbose:
                            print(
                                f"[autotune:{fn.__name__}] tuned key={key_dict} "
                                f"params={result.params} runtime_ms={result.runtime_ms:.4f}"
                            )

            chosen = cached["params"]
            if verbose:
                print(f"[autotune:{fn.__name__}] using params={chosen}")
            return fn(*args, **kwargs, **chosen)

        def pre_tune(*args, force: bool = False, **kwargs):
            old = os.environ.get(force_env_var)
            try:
                if force:
                    os.environ[force_env_var] = "1"
                return wrapper(*args, **kwargs)
            finally:
                if old is None:
                    os.environ.pop(force_env_var, None)
                else:
                    os.environ[force_env_var] = old

        def inspect_cache() -> Dict[str, Any]:
            return store._cache.copy()

        def clear_memo() -> None:
            with memo_lock:
                memo.clear()

        wrapper.pre_tune = pre_tune
        wrapper.inspect_cache = inspect_cache
        wrapper.clear_memo = clear_memo
        return wrapper

    return decorator


def _run_study(
    *,
    fn: Callable[..., Any],
    bound: inspect.BoundArguments,
    configs: Dict[str, list],
    constraint_fn: Optional[Callable[..., bool]],
    validate_fn: Optional[Callable[..., None]],
    n_trials: int,
    warmup: int,
    rep: int,
    sampler: optuna.samplers.BaseSampler,
) -> TuneResult:
    static_kwargs = dict(bound.arguments)

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: trial.suggest_categorical(name, values)
            for name, values in configs.items()
        }
        merged = {**static_kwargs, **params}

        if constraint_fn is not None and not constraint_fn(**merged):
            raise optuna.TrialPruned()

        def run():
            return fn(**merged)

        out = run()
        torch.cuda.synchronize()

        if validate_fn is not None:
            validate_fn(output=out, **merged)

        runtime_ms = _bench_callable(
            run,
            warmup=warmup,
            rep=rep,
        )
        trial.set_user_attr("runtime_ms", runtime_ms)
        return runtime_ms

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    return TuneResult(
        params=dict(study.best_params),
        runtime_ms=float(study.best_value),
    )


def relu_bat_c_reference(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return torch.relu(B @ A.T) @ C


def relu_bat_c_key(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> Dict[str, Any]:
    if A.device != B.device or A.device != C.device:
        raise ValueError("A, B and C must be on the same device")
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
        raise ValueError("A, B and C must all be 2D")
    if A.shape[1] != B.shape[1] or C.shape[1] != A.shape[1]:
        raise ValueError("A, B and C must have the same D dimension")

    key = _default_device_key(A)
    key.update(
        {
            "N": int(A.shape[0]),
            "M": int(B.shape[0]),
            "D": int(A.shape[1]),
            "V": int(A.shape[1] // 4),
            "R": int(A.shape[1] % 4),
        }
    )
    return key

def relu_bat_c_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BK: int,
    num_stages: int,
    num_ms: int,
    KNR: int,
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
    if num_stages not in (1, 2, 3, 4):
        return False
    if num_ms not in (1, 2, 4):
        return False
    if KNR not in (1, 2, 4, 8, 16):
        return False

    if KNR < 1 or KNR > BK:
        return False
    if BK % KNR != 0:
        return False

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
    num_stages: int,
    num_ms: int,
    KNR: int,
) -> None:
    del BM, BK, num_stages, num_ms, KNR
    ref = relu_bat_c_reference(A, B, C)
    torch.testing.assert_close(output, ref, rtol=1e-4, atol=1e-4)



# Environment flags:
#   KERNEL_AUTOTUNE_DISABLE=1   -  bypass autotuning, use first config in each list
#   KERNEL_AUTOTUNE_FORCE=1     -  ignore cache and retune
#   KERNEL_AUTOTUNE_VERBOSE=1   -  print tuning/cache diagnostics
#   KERNEL_AUTOTUNE_VALIDATE=1  -  validate correctness of candidate