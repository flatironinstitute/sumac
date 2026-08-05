from __future__ import annotations

import json
from dataclasses import dataclass, asdict, replace, fields
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import torch

from sumac.config import AutotuneMode
from sumac.kernels.cuda_utils import cuda_is_available

from .tuning_types import T_TuneConfig, TuneResult, KernelAutotuneOptions, TuneResultMode, T_FnParams, T_FnReturns, make_choices
from .json_cache import JsonConfigStore, normalize_for_json
from .option_handling import active_kernel_autotune_options, default_kernel_autotune_cache_dir

if TYPE_CHECKING:
    import optuna


def require_optuna():
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


def grid_size(config_dict: dict[str, list[int] | list[str]]) -> int:
    size = 1
    for values in config_dict.values():
        size += len(values)
    return size


def _bench_callable(fn: Callable[[], Any], *, warmup: int, rep: int) -> float:
    triton = _require_triton()
    return float(
        triton.testing.do_bench(    # type: ignore
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
    trial_number: int,
    reason: str,
    params: Dict[str, Any],
    exc: Optional[Exception] = None,
    verbose: bool = False,
) -> None:
    if not verbose:
        return
    message = f"[Trial {trial_number}] pruned: {reason}; params={params}"
    if exc is not None:
        message += f"; error={_brief_exception(exc)}"
    print(message)


@dataclass(frozen=True)
class ExperimentKey():
    major_version: int
    minor_version: int
    multi_processor_count: int
    N_A_outer_matrix_dimension: int
    M_B_outer_matrix_dimension: int
    D_inner_dimension: int


@dataclass(frozen=True)
class MemoryCacheKey():
    cache_directory: str
    cache_path: str
    mode: AutotuneMode
    disk_key: ExperimentKey


class AutotuneCudaKernel[T_Config: T_TuneConfig, T_Params: T_FnParams, T_Returns: T_FnReturns]():
    configs: list[T_Config]
    cache_path: Path
    n_trials: int
    warmup: int
    rep: int
    sampler: optuna.samplers.BaseSampler            # in practice always optuna.samplers.GridSampler(search_space=tune_config),
    wrapped_fn_name: str
    wrapped_fn_module: str
    decision_config: T_Config | None
    decision_runtime_ms: float
    interface_fn: Callable[[T_Params], T_Returns]
    decision_memo_cache: Dict[MemoryCacheKey, TuneResult]
    decision_disk_cache: JsonConfigStore
    options: KernelAutotuneOptions


    def __init__(self,
        configs: list[T_Config],
        wrapped_fn_name: str = "undef",
        wrapped_fn_module: str = "undef",
        cache_path: str = "kernel_autotune_cache.json",
        n_trials: int = 1000, #24, # the lower limit was never actually used
        warmup: int = 1, #25, # the higher limis are never actually used
        rep: int = 5, #100,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        autotune_options: KernelAutotuneOptions | None = None
    ):
        optuna = require_optuna()
        self.configs = configs
        self.cache_path = Path(cache_path)
        self.wrapped_fn_name = wrapped_fn_name
        self.wrapped_fn_module = wrapped_fn_module
        self.decision_runtime_ms = float("inf")
        self.n_trials = n_trials
        self.warmup = warmup
        self.rep = rep
        # QUERY: In practice, we never had a use where the sampler wasn't set to a grid search
        # over the parameters, so I'm just hard-coding that here.
        # Caller can still use a TPESampler if desired.
        # self.sampler = sampler or optuna.samplers.TPESampler(seed=0)
        self.sampler = sampler or optuna.samplers.GridSampler(search_space = make_choices(configs))
        self.decision_memo_cache = {}
        self.options = autotune_options or active_kernel_autotune_options()
        self.decision_disk_cache = self._get_disk_cache()
        self.decision_config = None


    def _get_key(self, params: T_Params) -> ExperimentKey:
        (A, B) = params[:2]
        props = torch.cuda.get_device_properties(A.device)
        N, D = A.shape
        M, _ = B.shape

        return ExperimentKey(
            props.major,
            props.minor,
            props.multi_processor_count,
            int(N),
            int(M),
            int(D)
        )


    def _get_memory_cache_key(self, disk_key: ExperimentKey):
        cache_dir_key = self.options.cache_dir_key or str(self.options.cache_dir or default_kernel_autotune_cache_dir())
        cache_directory = str(self.cache_path) if self.cache_path.is_absolute() else cache_dir_key
        return MemoryCacheKey(
            cache_directory,
            str(self.cache_path),
            self.options.mode,
            disk_key
        )


    def _get_disk_cache(self) -> JsonConfigStore:
        cache_file = (
            self.cache_path
            if self.cache_path.is_absolute()
            else (self.options.cache_dir or default_kernel_autotune_cache_dir()) / self.cache_path
        )
        return JsonConfigStore(cache_file)


    def resolve_decision(self, params: T_Params):
        must_fallback = self.options.mode == AutotuneMode.FALLBACK
        must_fallback = must_fallback or not cuda_is_available()
        for x in params:
            must_fallback = must_fallback or not x.is_cuda
            must_fallback = must_fallback or not x.dtype == torch.float32
            # TODO: Q: do we need to insist that they're all on the *same* device too?

        if must_fallback:   # by request or because cuda conditions aren't met
            self._set_interface_fn()
            return
        if self.options.mode == AutotuneMode.DISABLE:
            # If disable, we use the first config entry as the default
            self._set_interface_fn(decision = self.configs[0])
            return

        disk_cache_key = self._get_key(params)
        memo_cache_key = self._get_memory_cache_key(disk_cache_key)

        decision = self.decision_memo_cache.get(memo_cache_key)
        if decision is not None:
            self._set_interface_fn(decision)
            return

        normalized_disk_key = json.dumps(
            normalize_for_json(disk_cache_key),
            separators=(",", ":")
        )
        if self.options.mode == AutotuneMode.CACHE:
            cached_result = self.decision_disk_cache.get(normalized_disk_key)
            if cached_result is not None:
                _disk_decision = TuneResult(**cached_result)
                self.decision_memo_cache[memo_cache_key] = _disk_decision
                self._set_interface_fn(_disk_decision)
                return

        # Cannot serve from cache--run the study
        result = self._run_study(params, normalized_disk_key)

        self.decision_memo_cache[memo_cache_key] = result
        self.decision_disk_cache.put(normalized_disk_key, asdict(result))
        self._set_interface_fn(result)

        if self.options.verbose:
            print(
                f"[autotune:{self.wrapped_fn_name} tuned key={memo_cache_key}] "
                f"mode={result.mode.value} params={result.params} "
                f"runtime_ms={result.runtime_ms:.4f}"
            )


    def _run_study(self,
        params: T_Params,
        disk_key: str
    ) -> TuneResult:
        optuna = require_optuna()

        fallback_runtime_ms = self._bench_fallback(params)

        study_name = f"{self.wrapped_fn_module}.{self.wrapped_fn_name}:{disk_key}"
        objective = lambda trial: self._tuning_objective(trial, params, optuna)

        previous_optuna_verbosity = optuna.logging.get_verbosity()
        if not self.options.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        # TODO QUESTION: Do we potentially need to increase the verbosity
        # if self.options.verbose is True? Prior verbosity might've been too low
        try:
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                sampler=self.sampler,
            )
            study.optimize(objective, n_trials=self.n_trials)
        finally:
            optuna.logging.set_verbosity(previous_optuna_verbosity)

        best = study.best_trial
        best_cuda_runtime_ms = float(best.user_attrs["runtime_ms"])
        best_cuda_params = {name: best.params[name] for name in [f.name for f in fields(self.configs[0])]}

        if fallback_runtime_ms <= best_cuda_runtime_ms:
            return TuneResult(
                mode=TuneResultMode.FALLBACK,
                params={},
                runtime_ms=fallback_runtime_ms,
            )

        return TuneResult(
            mode=TuneResultMode.CUDA,
            params=best_cuda_params,
            runtime_ms=best_cuda_runtime_ms,
        )


    def _tuning_objective(self,
        trial: optuna.Trial,    # type: ignore
        params: T_Params,
        optuna: Any
    ) -> float:
        selected_config_dict = {
            name: trial.suggest_categorical(name, values)
            for name, values in make_choices(self.configs).items()
        }
        selected_config = replace(self.configs[0], **selected_config_dict)
        def _print_partial(msg: str, e: Exception | None = None):
            _print_trial_pruned(trial.number, msg, selected_config_dict, e, self.options.verbose)

        self._check_constraints(params, selected_config, optuna, _print_partial)

        def run(): return self._candidate_fn(params, selected_config)

        try:
            failure_reason = "jit compile or kernel launch failure"
            run()
            failure_reason = "runtime failure after warmup launch"
            torch.cuda.synchronize()
            failure_reason = "runtime or benchmark failure"
            runtime_ms = _bench_callable(run, warmup=self.warmup, rep=self.rep)
            trial.set_user_attr("runtime_ms", runtime_ms)
        except Exception as e:
            _print_partial(failure_reason, e)
        return runtime_ms


    def _check_constraints(self, params: T_Params, cfg: T_Config, optuna: Any, _print: Callable):
        try:
            constraint_ok = self._constraint(params, cfg)
        except Exception as e:
            _print("constraint_fn error", e)
            raise optuna.TrialPruned() from e

        if not constraint_ok:
            _print("Constraint rejected config")
            raise optuna.TrialPruned()


    def _bench_fallback(self, params: T_Params) -> float:
        fallback_runtime_ms = float("inf")
        def run_fallback(): return self._fallback(params)

        try:
            run_fallback()
            fallback_runtime_ms = _bench_callable(
                run_fallback,
                warmup=self.warmup,
                rep=self.rep,
            )
        except Exception as e:
            if self.options.verbose:
                print(f"[fallback] Error: {e}")
            fallback_runtime_ms = float("inf")
        return fallback_runtime_ms


    def _set_interface_fn(self, decision: TuneResult | T_Config | None = None):
        """Sets the __call__ interface for this instance based on the results of autotuning,
        and records the latest decision results in the self.decision_* variables.

        This is a little dense, so here's a summary of the intended situation.

        The purpose of this function is to set self.__call__ so that it always
        takes a tuple of the parameters expected by whatever kind of kernel we are
        (i.e. A, B, C for relu_bat_c and A, B for bat_reduce). self._fallback should
        always just take those parameters. Whatever our candidate function is
        (self._candidate_fn) will take those parameters, plus some config parameters;
        in that case, we'll actually set self.__call__ to a lambda that's closed over
        the selected configuration.

        Anyway, when this fn is done, __call__ refers to either:
        1) the fallback function,
        2) the wrapped kernel with default parameters, or
        3) the wrapped kernel with tuned parameters.

        We also need to set the observed runtime, and record what the selected config
        parameters actually were.

        In case 1 and 2, we didn't run a study, so self.decision_runtime_ms is INF.
        In case 1, there is no config, so we set self.decision_config to None.
        In case 2, we set self.decision_config to the default (first config in the list).

        In case 3, we actually did autotuning experiments (possibly from cache hit).
        Then, self.decision_runtime_ms is set to the study result, and self.decision_config
        is the parameters of the best result, EXCEPT in the case where the fallback function
        was actually faster--then we set the decision_config to None and __call__ to the
        fallback.
        
        For convenience, this information is all packed into the decision parameter:
          - If it's None, that means we do fallback
          - If it's a T_Config, that must be a default config
          - Otherwise it's a TuneResult, which we'll read for the run-time and config,
          again unless the fallback function outperformed.

        Args:
            decision (TuneResult | T_Config | None, optional): None (the default) for fallback
                mode, a default T_Config for default mode, or the result of empirical
                tuning testing (TuneResult) if we actually ran a study.
        """
        if isinstance(decision, TuneResult):
            self.decision_runtime_ms = decision.runtime_ms
            self.decision_config = replace(self.configs[0], **decision.params)
            if decision.mode == TuneResultMode.FALLBACK:
                self.decision_config = None
        else:
            # handles fallback ("None") and Disable ("T_Config") cases
            self.decision_config = decision
            self.decision_runtime_ms = float("inf")

        cfgs = self.decision_config
        if cfgs is None:    # incl case where decision is None (AutotuneMode.FALLBACK)
            self.interface_fn = self._fallback
        else:
            self.interface_fn = lambda params: self._candidate_fn(params, cfgs)


    def _candidate_fn(self, params: T_Params, config: T_Config) -> T_Returns:
        raise NotImplementedError("To be implemented by subclass.")


    def _fallback(self, params: T_Params) -> T_Returns:
        raise NotImplementedError("To be implemented by subclass.")


    def _constraint(self, params: T_Params, config: T_Config) -> bool:
        raise NotImplementedError("To be implemented by subclass.")


    def __call__(self, params: T_Params) -> T_Returns:
        return self.interface_fn(params)
