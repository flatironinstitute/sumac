from dataclasses import dataclass, fields
from enum import Enum
import math
import os
from pathlib import Path
import torch


def _validate_finite_number(name: str, value: int | float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


class SumacMethod(Enum):
    """Which strategy to use: SALSA (stochastic alternating least
    squares) or GD (gradient descent with choice of optimizer).
    """
    SALSA = 'salsa'
    GD = 'gd'


class OptimizerName(Enum):
    """Choice of supported optimizer for GD mode."""
    ADAM = 'adam'
    ADAMW = 'adamw'
    SGD = 'sgd'
    MUON = 'muon'


class AutotuneMode(Enum):
    """Controls autotuner cache usage."""
    CACHE = 'cache'
    FORCE = 'force'
    DISABLE = 'disable'
    FALLBACK = 'fallback'


def default_kernel_autotune_cache_dir() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "sumac"
    return Path.home() / ".cache" / "sumac"


@dataclass(kw_only=True)
class SumacConfig:
    """Unified configuration object for Sumac.

    All parameters have reasonable defaults. Supported fields
    include shared configuration parameters and those that are
    only relevant for specific modes of operation--irrelevant
    parameters will be ignored by the rest of the code.

    Attributes:
        method (SumacMethod): Which of the available Sumac
            factorization methods will be ueds. (Default: SALSA)
        rank (int): Target rank for the low-rank representation.
            (Default: 16)
        max_iterations (int): Maximum iterations before termination.
            (Default: 25)
        num_blocks (int | None): Number of blocks to use for blocking
            update algorithms. Defaults to None, in which case it will
            be computed from the available cache memory (cache_mb) and
            the size of the input matrix.
        cols_per_block (int | None): Columns to include per matrix
            block for blocked matrix algorithms. Defaults to None,
            in which case it will be computed based on the requested
            block count and input matrix size.
        seed (int | None): Random seed for reproducible results.
            By default set to None, to avoid restricting result space.
        cache_mb (int): Available megabytes of memory for caching, in MB.
            (Default: 5000)
        dtype (torch.dtype): Torch datatype for computation: makes
            the choice between single- and double-precision (or,
            if allowed, TF32 representation). (Default: float32)
        allow_tf32 (bool): Whether to use TF32 instead of IEEE
            single-precision representation during training. (Default:
            False) If True, and the dtype parameter is set to float32
            (single-precision), TF32 will be used if supported by the
            system.
        device (torch.device | None): Which device (CPU, GPU) to use for
            training. (Default: None) If unset, will be inferred from
            the location of the input sparse matrix.
        momentum (float | None): Momentum used in training. For GD methods,
            will be used by the chosen optimizer; for SALSA, used
            directly in updates. (Default: 0.7, unless GD is used
            with muon optimizer, in which case the default is 0.95.)
        learning_rate (float): Learning rate for training. For GD
            methods, used by the chosen optimizer; for SALSA, used
            in updates directly.
        verbose (bool): Whether to print lots of status messages.
            (Default: True)
        eval_interval (int | None): How many iterations to run
            between reporting loss values. If set to None
            (the default), will be set to 10 for SALSA or 100
            for GD modes.
        optimizer (OptimizerName): Determines the type of
            optimizer used for GD methods; ignored for SALSA.
            Available optimizers are defined by the OptimizerName
            enum. (Default: adam)
        adam_betas (tuple[float, float]): Used for configuring
            adam or adamw optimizer in GD mode. Ignored for
            SALSA or GD with other optimizers. (Default:
            (0.9, 0.999))
        adam_eps (float): Epsilon parameter for configuring
            adam or adamw optimizer in GD mode. Ignored for
            SALSA or GD with other optimizers. (Default: 1e-8)
        shuffle_blocks (bool): Whether to shuffle blocks
            through the data loader. Only relevant in GD
            mode; SALSA mode always shuffles blocks at every
            iteration. (Default: False)
        batch_blocks (int): Number of blocks per batch in
            GD mode. Ignored in SALSA mode. (Default: 1)
        autotune (AutotuneMode): How the autotuner will
            be used. Available options defined by the
            AutotuneMode enum. (Default: cache)
        autotune_cache_dir (str): Directory to use for
            the autotuner data cache. If None (the default),
            sumac will inspect the XDG_CACHE_HOME environment
            variable. If that is set, we will use a "sumac"
            directory under its value; otherwise, the default
            is ".cache/sumac" within the home directory.
        autotune_verbose (bool): Whether the autotuner should
            print verbose output. (Default: False)
        input_filename (str): Used by example code to identify
            a data file to load. Ignored in normal use.
        log_filename (str): Output filename for log data
            from examples. Ignored in normal use.
    """
    method: SumacMethod = SumacMethod.SALSA
    rank: int = 16
    max_iterations: int = 25
    num_blocks: int | None = None
    cols_per_block: int | None = None
    seed: int | None = None
    cache_mb: int = 5000
    dtype: torch.dtype = torch.float32
    allow_tf32: bool = False
    device: torch.device | None = None
    momentum: float = -1.
    learning_rate: float = 1e-1
    verbose: bool = True
    eval_interval: int | None = None
    # optimizer-specific
    optimizer: OptimizerName = OptimizerName.ADAM
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    shuffle_blocks: bool = False
    batch_blocks: int = 1
    # autotuning
    autotune: AutotuneMode = AutotuneMode.CACHE
    autotune_cache_dir: str | Path | None = None
    autotune_verbose: bool = False
    # consumed by examples only
    input_filename: str | None = None
    log_filename: str | None = None


    def __post_init__(self):
        self._validate_integer_options()
        self._validate_numeric_options()

        if self.eval_interval is None:
            self.eval_interval = 100 if self.method == SumacMethod.GD else 10
        if self.eval_interval <= 0:
            raise ValueError(
                "eval_interval must be a positive integer or None, "
                f"got {self.eval_interval!r}"
            )

        if self.autotune_cache_dir is None:
            self.autotune_cache_dir = default_kernel_autotune_cache_dir()
        self.autotune_cache_dir = str(self.autotune_cache_dir)

        if self.momentum < 0:
            self.momentum = 0.7
            if (self.optimizer == OptimizerName.MUON
                and self.method == SumacMethod.GD):
                self.momentum = 0.95
        if not 0 <= self.momentum < 1:
            raise ValueError(
                "momentum must satisfy 0 <= momentum < 1, "
                f"got {self.momentum!r}"
            )

        self._validate_adam_options()

        if self.seed is not None and self.verbose:
            print(f"seed = {self.seed}")


    def _validate_integer_options(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {self.rank!r}")
        if self.max_iterations < 0:
            raise ValueError(
                "max_iterations must be a nonnegative integer, "
                f"got {self.max_iterations!r}"
            )
        if self.num_blocks is not None and self.num_blocks < 1:
            raise ValueError(
                "num_blocks must be a positive integer or None, "
                f"got {self.num_blocks!r}"
            )
        if self.batch_blocks <= 0:
            raise ValueError(
                "batch_blocks must be a positive integer, "
                f"got {self.batch_blocks!r}"
            )


    def _validate_numeric_options(self) -> None:
        _validate_finite_number("cache_mb", self.cache_mb)
        if self.cache_mb <= 0:
            raise ValueError(f"cache_mb must be positive, got {self.cache_mb!r}")
        if self.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "dtype must be torch.float32 or torch.float64, "
                f"got {self.dtype!r}"
            )
        if self.dtype == torch.float64 and self.allow_tf32:
            raise ValueError(
                "dtype=torch.float64 and allow_tf32=True are mutually exclusive"
            )
        _validate_finite_number("learning_rate", self.learning_rate)
        if self.learning_rate < 0:
            raise ValueError(
                "learning_rate must be nonnegative, "
                f"got {self.learning_rate!r}"
            )
        _validate_finite_number("momentum", self.momentum)


    def _validate_adam_options(self) -> None:
        if self.method != SumacMethod.GD or self.optimizer not in (
            OptimizerName.ADAM,
            OptimizerName.ADAMW,
        ):
            return

        _validate_finite_number("adam_eps", self.adam_eps)
        if self.adam_eps <= 0:
            raise ValueError(f"adam_eps must be positive, got {self.adam_eps!r}")
        for beta in self.adam_betas:
            _validate_finite_number("adam_betas value", beta)
            if not 0 <= beta < 1:
                raise ValueError(
                    "adam_betas values must satisfy 0 <= beta < 1, "
                    f"got {self.adam_betas!r}"
                )


    def set_block_sizes(self, m: int, n: int):
        if self.num_blocks is None:
            max_bytes = self.cache_mb * 1e6
            bytes_per_dtype = 8 if self.dtype == torch.float64 else 4
            self.cols_per_block = max(1, int(max_bytes // (m * bytes_per_dtype)))
            self.num_blocks = math.ceil(n / self.cols_per_block)

        max_num_blocks = min(m, n)
        if self.num_blocks > max_num_blocks:
            raise ValueError(
                "num_blocks must satisfy "
                f"1 <= num_blocks <= min(effective shape)={max_num_blocks}, "
                f"got {self.num_blocks}"
            )
        self.cols_per_block = int(n // self.num_blocks)


    def print_prefactor_report(self, S_value: torch.Tensor, m: int, n: int, devcount: int):
        if not self.verbose: return
        nnz = len(S_value)
        print(f"\n  Input to SUMAC is a {m}x{n} sparse matrix with {nnz} nonzeros.")
        print(f"  Attempting matrix completion with rank {self.rank}.")
        print(f"  Available GPUs: {devcount}.")
        print(f"  Options:")
        opts = [f'    {m.name}: {getattr(self, m.name)}' for m in fields(self)]
        print("\n".join(opts))
        print()

    
    def get_generator(self):
        gen = torch.Generator(device=self.device)
        gen_rows = torch.Generator(device=self.device)
        tmp_device = self.device if self.method == SumacMethod.SALSA else "cpu"
        gen_aux = torch.Generator(device=tmp_device)
        generators = (gen, gen_rows, gen_aux)

        if self.seed is None:
            for generator in generators:
                generator.seed()
        else:
            for offset, generator in enumerate(generators):
                generator.manual_seed(self.seed + offset)

        return generators
