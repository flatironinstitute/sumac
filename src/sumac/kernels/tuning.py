from __future__ import annotations

import functools
import itertools
import json
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Literal

import torch


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "CUDA kernel autotuning requires the optional dependency 'optuna'. "
            "Install the CUDA/autotune extras to use custom CUDA kernels."
        ) from exc
    return optuna


def _require_triton():
    try:
        import triton
    except ImportError as exc:
        raise RuntimeError(
            "CUDA kernel benchmarking requires the optional dependency 'triton'. "
            "Install the CUDA/autotune extras to use custom CUDA kernels."
        ) from exc
    return triton


class JsonConfigStore:
    def __init__(self, path: str | Path):
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
    triton = _require_triton()
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


def _brief_exception(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    return text.splitlines()[0]


def _print_trial_pruned(
    trial: Any,
    reason: str,
    params: Dict[str, Any],
    exc: Optional[Exception] = None,
    verbose: bool = False,
) -> None:
    if not verbose:
        return
    message = f"[Trial {trial.number}] pruned: {reason}; params={params}"
    if exc is not None:
        message += f"; error={_brief_exception(exc)}"
    print(message)


@dataclass(frozen=True)
class TuneResult:
    mode: str
    params: Dict[str, Any]
    runtime_ms: float


AutotuneMode = Literal["cache", "force", "disable", "fallback"]
AUTOTUNE_MODES: tuple[AutotuneMode, ...] = (
    "cache",
    "force",
    "disable",
    "fallback",
)


@dataclass(frozen=True)
class KernelAutotuneOptions:
    mode: AutotuneMode = "cache"
    cache_dir: Path | None = None
    cache_dir_key: str | None = None
    verbose: bool = False
    session_id: int = 0


AUTOTUNE_SESSION_IDS = itertools.count(1)

ACTIVE_AUTOTUNE_OPTIONS: ContextVar[KernelAutotuneOptions] = ContextVar(
    "ACTIVE_AUTOTUNE_OPTIONS",
    default=KernelAutotuneOptions(),
)


def normalize_autotune_mode(mode: str) -> AutotuneMode:
    if mode not in AUTOTUNE_MODES:
        raise ValueError(f"autotune must be one of {AUTOTUNE_MODES}, got {mode!r}")
    return mode


def default_kernel_autotune_cache_dir() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "sumac"
    return Path.home() / ".cache" / "sumac"


def active_kernel_autotune_options() -> KernelAutotuneOptions:
    return ACTIVE_AUTOTUNE_OPTIONS.get()


@contextmanager
def kernel_autotune_options(
    *,
    mode: str = "cache",
    cache_dir: str | Path | None = None,
    verbose: bool = False,
):
    normalized_mode = normalize_autotune_mode(mode)
    resolved_cache_dir = (
        default_kernel_autotune_cache_dir()
        if cache_dir is None
        else Path(cache_dir)
    )
    options = KernelAutotuneOptions(
        mode=normalized_mode,
        cache_dir=resolved_cache_dir,
        cache_dir_key=str(resolved_cache_dir),
        verbose=verbose,
        session_id=next(AUTOTUNE_SESSION_IDS) if normalized_mode == "force" else 0,
    )
    token = ACTIVE_AUTOTUNE_OPTIONS.set(options)
    try:
        yield options
    finally:
        ACTIVE_AUTOTUNE_OPTIONS.reset(token)


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
    autotune_options: KernelAutotuneOptions | None = None,
):
    memo: Dict[Any, Dict[str, Any]] = {}
    default_params = {k: v[0] for k, v in configs.items()}
    cache_path_obj = Path(cache_path)
    cache_path_key = str(cache_path_obj)
    cache_path_is_absolute = cache_path_obj.is_absolute()
    fixed_options = autotune_options
    fixed_cache_file = None
    fixed_store = None
    if fixed_options is not None and fixed_options.mode not in ("disable", "fallback"):
        fixed_cache_file = (
            cache_path_obj
            if cache_path_is_absolute
            else (fixed_options.cache_dir or default_kernel_autotune_cache_dir()) / cache_path_obj
        )
        fixed_store = JsonConfigStore(fixed_cache_file)

    if sampler is None:
        optuna = _require_optuna()
        sampler = optuna.samplers.TPESampler(seed=0)

    def decorator(fn):
        def resolve_decision(*args, **kwargs) -> Dict[str, Any]:
            options = fixed_options or active_kernel_autotune_options()

            if options.mode == "fallback":
                if fallback_fn is None:
                    raise ValueError(
                        f"autotune='fallback' was requested for {fn.__name__}, "
                        "but no fallback_fn was provided"
                    )
                decision = {
                    "mode": "fallback",
                    "params": dict(default_params),
                    "runtime_ms": float("inf"),
                }
                return decision

            mem_key = key_fn(*args, **kwargs)
            if fixed_options is not None:
                memo_key = mem_key
            else:
                cache_dir_key = options.cache_dir_key
                if cache_dir_key is None:
                    cache_dir_key = str(
                        options.cache_dir or default_kernel_autotune_cache_dir()
                    )
                memo_key = (
                    cache_path_key if cache_path_is_absolute else cache_dir_key,
                    cache_path_key,
                    options.mode,
                    options.session_id,
                    mem_key,
                )

            decision = memo.get(memo_key)
            if decision is not None:
                return decision

            if options.mode == "disable":
                decision = {
                    "mode": "cuda",
                    "params": dict(default_params),
                    "runtime_ms": float("inf"),
                }
                memo[memo_key] = decision
                return decision

            if fixed_options is not None:
                store = fixed_store
            else:
                cache_file = (
                    cache_path_obj
                    if cache_path_is_absolute
                    else (options.cache_dir or default_kernel_autotune_cache_dir()) / cache_path_obj
                )
                store = JsonConfigStore(cache_file)

            disk_key = json.dumps(_normalize_for_json(mem_key), separators=(",", ":"))

            payload = None if options.mode == "force" else store.get(disk_key)
            if payload is not None:
                memo[memo_key] = payload
                return payload

            result = _run_study(
                fn=fn,
                fallback_fn=fallback_fn,
                constraint_fn=constraint_fn,
                args=args,
                kwargs=kwargs,
                configs=configs,
                n_trials=n_trials,
                warmup=warmup,
                rep=rep,
                sampler=sampler,
                disk_key=disk_key,
                verbose=options.verbose,
            )

            decision = {
                "mode": result.mode,
                "params": result.params,
                "runtime_ms": result.runtime_ms,
            }
            memo[memo_key] = decision

            store.put(
                disk_key,
                {
                    "function": fn.__name__,
                    "mode": result.mode,
                    "params": result.params,
                    "runtime_ms": result.runtime_ms,
                },
            )

            if options.verbose:
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
        wrapper.resolve_params = (
            lambda *args, **kwargs: resolve_decision(*args, **kwargs)["params"]
        )
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
    sampler: Any,
    disk_key: str,
    verbose: bool,
) -> TuneResult:
    optuna = _require_optuna()
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
            if verbose:
                print(f"[fallback] Error: {e}")
            fallback_runtime_ms = float("inf")

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: trial.suggest_categorical(name, values)
            for name, values in configs.items()
        }
        merged = {**static_kwargs, **params}

        if constraint_fn is not None:
            try:
                constraint_ok = constraint_fn(*args, **merged)
            except Exception as e:
                _print_trial_pruned(
                    trial,
                    "constraint_fn error",
                    params,
                    e,
                    verbose=verbose,
                )
                raise optuna.TrialPruned() from e

            if not constraint_ok:
                _print_trial_pruned(
                    trial,
                    "constraint rejected config",
                    params,
                    verbose=verbose,
                )
                raise optuna.TrialPruned()

        def run():
            return fn(*args, **merged)

        try:
            run()
        except Exception as e:
            _print_trial_pruned(
                trial,
                "jit compile or kernel launch failure",
                params,
                e,
                verbose=verbose,
            )
            raise optuna.TrialPruned() from e

        try:
            torch.cuda.synchronize()
        except Exception as e:
            _print_trial_pruned(
                trial,
                "runtime failure after warmup launch",
                params,
                e,
                verbose=verbose,
            )
            raise optuna.TrialPruned() from e

        try:
            runtime_ms = _bench_callable(run, warmup=warmup, rep=rep)
            trial.set_user_attr("runtime_ms", runtime_ms)
        except Exception as e:
            _print_trial_pruned(
                trial,
                "runtime or benchmark failure",
                params,
                e,
                verbose=verbose,
            )
            raise optuna.TrialPruned() from e

        return runtime_ms

    study_name = f"{fn.__module__}.{fn.__qualname__}:{disk_key}"
    previous_optuna_verbosity = optuna.logging.get_verbosity()
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
        )
        study.optimize(objective, n_trials=n_trials)
    finally:
        if not verbose:
            optuna.logging.set_verbosity(previous_optuna_verbosity)

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
