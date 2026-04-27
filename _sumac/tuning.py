import functools
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
    mode: str 
    params: Dict[str, Any]
    runtime_ms: float


def autotune_cuda_kernel(
    *,
    configs,
    key_fn,
    fallback_fn: Optional[Callable[..., Any]] = None,
    constraint_fn: Optional[Callable[..., bool]] = None,
    cache_path="kernel_autotune_cache.json",
    n_trials=24,
    warmup=25,
    rep=100,
    sampler=None,
    force_env_var="KERNEL_AUTOTUNE_FORCE",
    disable_env_var="KERNEL_AUTOTUNE_DISABLE",
    verbose_env_var="KERNEL_AUTOTUNE_VERBOSE",
    force_fallback_env_var="KERNEL_AUTOTUNE_FORCE_FALLBACK",
    disable_fallback_env_var="KERNEL_AUTOTUNE_DISABLE_FALLBACK",
):
    store = JsonConfigStore(cache_path)
    memo: Dict[Any, Dict[str, Any]] = {}
    default_params = {k: v[0] for k, v in configs.items()}

    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=0)

    disable_default = os.getenv(disable_env_var, "0") == "1"
    force_default = os.getenv(force_env_var, "0") == "1"
    verbose_default = os.getenv(verbose_env_var, "0") == "1"
    force_fallback_default = os.getenv(force_fallback_env_var, "0") == "1"
    disable_fallback_default = os.getenv(disable_fallback_env_var, "0") == "1"

    def decorator(fn):
        def resolve_decision(*args, **kwargs) -> Dict[str, Any]:
            mem_key = key_fn(*args, **kwargs)

            if force_fallback_default:
                if fallback_fn is None:
                    raise ValueError(
                        f"{force_fallback_env_var}=1 but no fallback_fn was provided"
                    )
                decision = {
                    "mode": "fallback",
                    "params": dict(default_params),
                    "runtime_ms": float("inf"),
                }
                memo[mem_key] = decision
                return decision

            if disable_default:
                decision = {
                    "mode": "cuda",
                    "params": dict(default_params),
                    "runtime_ms": float("inf"),
                }
                memo[mem_key] = decision
                return decision

            decision = None if force_default else memo.get(mem_key)
            if decision is not None:
                return decision

            disk_key = json.dumps(_normalize_for_json(mem_key), separators=(",", ":"))

            payload = None if force_default else store.get(disk_key)
            if payload is not None:
               
                memo[mem_key] = payload
                return payload

            result = _run_study(
                fn=fn,
                fallback_fn=None if disable_fallback_default else fallback_fn,
                constraint_fn=constraint_fn,
                args=args,
                kwargs=kwargs,
                configs=configs,
                n_trials=n_trials,
                warmup=warmup,
                rep=rep,
                sampler=sampler,
                disk_key=disk_key,
            )

            decision = {
                "mode": result.mode,
                "params": result.params,
                "runtime_ms": result.runtime_ms,
            }
            memo[mem_key] = decision

            store.put(
                disk_key,
                {
                    "function": fn.__name__,
                    "mode": result.mode,
                    "params": result.params,
                    "runtime_ms": result.runtime_ms,
                },
            )

            if verbose_default:
                print(
                    f"[autotune:{fn.__name__}] tuned key={mem_key} "
                    f"mode={result.mode} params={result.params} "
                    f"runtime_ms={result.runtime_ms:.4f}"
                )

            return decision

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            decision = resolve_decision(*args, **kwargs)

            if decision["mode"] == "fallback":
                if fallback_fn is None:
                    raise RuntimeError(
                        f"Resolved to fallback mode for {fn.__name__}, "
                        "but no fallback_fn was provided"
                    )
                return fallback_fn(*args, **kwargs)

            return fn(*args, **kwargs, **decision["params"])

        wrapper.resolve_decision = resolve_decision
        wrapper.resolve_params = lambda *args, **kwargs: resolve_decision(*args, **kwargs)["params"]
        wrapper.clear_memo = memo.clear
        return wrapper

    return decorator


def _run_study(
    *,
    fn: Callable[..., Any],
    fallback_fn: Optional[Callable[..., Any]],
    constraint_fn: Optional[Callable[..., bool]],
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
    configs: Dict[str, list],
    n_trials: int,
    warmup: int,
    rep: int,
    sampler: optuna.samplers.BaseSampler,
    disk_key: str,
) -> TuneResult:
    static_kwargs = dict(kwargs)

    fallback_runtime_ms = float("inf")
    if fallback_fn is not None:
        def run_fallback():
            return fallback_fn(*args, **static_kwargs)

        try:
            run_fallback()
            fallback_runtime_ms = _bench_callable(
                run_fallback,
                warmup=warmup,
                rep=rep,
            )
        except Exception as e:
            print(f"[fallback] Error: {e}")
            fallback_runtime_ms = float("inf")

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

        try:
            run()
            runtime_ms = _bench_callable(run, warmup=warmup, rep=rep)
            trial.set_user_attr("runtime_ms", runtime_ms)
        except Exception as e:
            print(f"[Trial {trial.number}] Error: {e}")
            raise optuna.TrialPruned()

        return runtime_ms

    study_name = f"{fn.__module__}.{fn.__qualname__}:{disk_key}"
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_cuda_runtime_ms = float(best.user_attrs["runtime_ms"])
    best_cuda_params = {name: best.params[name] for name in configs}

    if fallback_runtime_ms <= best_cuda_runtime_ms:
        return TuneResult(
            mode="fallback",
            params={},
            runtime_ms=fallback_runtime_ms,
        )

    return TuneResult(
        mode="cuda",
        params=best_cuda_params,
        runtime_ms=best_cuda_runtime_ms,
    )


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


def relu_bat_reduce_key(
    A: torch.Tensor,
    B: torch.Tensor,
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

# Environment flags:
#   KERNEL_AUTOTUNE_DISABLE=1           -> bypass autotuning, use first config in each list
#   KERNEL_AUTOTUNE_FORCE=1             -> ignore cache and retune
#   KERNEL_AUTOTUNE_VERBOSE=1           -> print tuning/cache diagnostics
#   KERNEL_AUTOTUNE_FORCE_FALLBACK=1    -> always use fallback_fn
#   KERNEL_AUTOTUNE_DISABLE_FALLBACK=1  -> do not benchmark or select fallback_fn