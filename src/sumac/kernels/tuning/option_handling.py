from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path

from .tuning_types import KernelAutotuneOptions
from sumac.config import AutotuneMode


ACTIVE_AUTOTUNE_OPTIONS: ContextVar[KernelAutotuneOptions] = ContextVar(
    "ACTIVE_AUTOTUNE_OPTIONS",
    default=KernelAutotuneOptions(),
)

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
    mode: AutotuneMode = AutotuneMode.CACHE,
    cache_dir: str | Path | None = None,
    verbose: bool = False,
):
    # normalized_mode = normalize_autotune_mode(mode)
    resolved_cache_dir = (
        default_kernel_autotune_cache_dir()
        if cache_dir is None
        else Path(cache_dir)
    )
    options = KernelAutotuneOptions(
        mode=AutotuneMode(mode),
        cache_dir=resolved_cache_dir,
        cache_dir_key=str(resolved_cache_dir),
        verbose=verbose,
    )
    token = ACTIVE_AUTOTUNE_OPTIONS.set(options)
    try:
        yield options
    finally:
        ACTIVE_AUTOTUNE_OPTIONS.reset(token)
