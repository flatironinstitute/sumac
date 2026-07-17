import argparse
from dataclasses import dataclass, fields
from enum import Enum
import math
import os
from pathlib import Path
import random
import torch

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
        eval_only (bool): Used in examples only. If True,
            the example will skip training and only provide
            an evaluation of provided matrix factors.
        eval_path (str | None): Used in examples only, as the
            directory to write the matrix factors into. In
            eval-only mode, this directory should contain
            the matrix factors to evaluate.
        eval_save (bool): Used in examples only. If True,
            the factors discovered through training will be
            saved in eval_path; otherwise they are discarded.
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
    eval_only: bool = False
    eval_path: str | None = None
    eval_save: bool = False


    def __post_init__(self):
        if self.eval_interval is None:
            self.eval_interval = 100 if self.method == SumacMethod.GD else 10
        if self.autotune_cache_dir is None:
            self.autotune_cache_dir = default_kernel_autotune_cache_dir()
        self.autotune_cache_dir = str(self.autotune_cache_dir)
        if self.momentum < 0:
            self.momentum = 0.7
            if (self.optimizer == OptimizerName.MUON
                and self.method == SumacMethod.GD):
                self.momentum = 0.95
        if self.seed is not None:
            if self.verbose:
                print(f"seed = {self.seed}")
            random.seed(self.seed)


    def set_block_sizes(self, m: int, n: int):
        if self.num_blocks is None:
            max_bytes = self.cache_mb * 1e6
            bytes_per_dtype = 8 if self.dtype == torch.float64 else 4
            self.cols_per_block = max(1, int(max_bytes // (m * bytes_per_dtype)))
            self.num_blocks = math.ceil(n / self.cols_per_block)
        else:
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
        gen = torch.Generator(device = self.device)
        gen_rows = None
        gen_cols = None
        if self.method == SumacMethod.SALSA:
            gen_rows = torch.Generator(device = self.device)
            gen_cols = torch.Generator(device = self.device)
        if self.seed is not None:
            gen.manual_seed(self.seed)
            torch.manual_seed(self.seed)
            if gen_rows is not None: gen_rows.manual_seed(self.seed + 1)
            if gen_cols is not None: gen_cols.manual_seed(self.seed + 2)
        return (gen, gen_rows, gen_cols)


def make_config_from_args() -> SumacConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default = None,
        help='If provided, should be the path to an input data file. For examples only.'
    )
    parser.add_argument('--mode', type=str, default='SALSA', choices=['SALSA', 'GD'],
        help='default (sumac): SALSA'
    )
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--num_blocks', type=int, default=100,
        help='number of blocks'
    )
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--momentum', type=float, default=0.9,
        help='use momentum to update W and H'
    )
    parser.add_argument('--float64', action='store_true',
        help='use torch.float64 (default use float32)'
    )
    parser.add_argument('--allow_tf32', action='store_true',
        help='allow PyTorch and SUMAC custom kernels to use TF32.'
    )
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='adam',
        choices=['adam','sgd', 'adamw', 'muon'],
        help='Optimizer to use for GD (ignored for SALSA)'
    )
    parser.add_argument('--eval_interval', type=int, default=None,
        help='Evaluation interval (default: 100 for GD, 10 for SALSA)'
    )
    parser.add_argument('--log_filename', type=str, default=None,
        help="filename for logging. If unset, stdout is used."
    )
    parser.add_argument('--eval_only', action='store_true',
        help='Only evaluate metrics for factors saved in eval_path, no training is performed if set.'
    )
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--eval_save',  action='store_true',
        help='Save evaluation results to .txt file.'
    )
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--autotune', type=str, default='cache',
        choices=['cache', 'force', 'disable', 'fallback'],
        help='CUDA kernel autotuning mode.'
    )
    parser.add_argument('--autotune_cache_dir', type=str, default=None,
        help='Directory for SUMAC kernel autotune cache files.'
    )
    parser.add_argument('--autotune_verbose', action='store_true',
        help='Print CUDA kernel autotuning decisions.'
    )
    parser.add_argument('--device', type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Training/evaluation device (defaults to cuda if a gpu is available)'
    )

    args = parser.parse_args()

    try:
        method = SumacMethod(str.lower(args.mode))
    except ValueError:
        print(f"Method {args.mode} is not supported (valid values are salsa and gd)")
    try:
        optimizer = OptimizerName(str.lower(args.optim))
    except ValueError:
        print(f"Optimizer type {args.optim} is not supported")
    try:
        autotune_mode = AutotuneMode(str.lower(args.autotune))
    except ValueError:
        print(f"Autotune mode {args.autotune} is not recognized " +\
              "(valid values are cache, force, disable, fallback)")

    return SumacConfig(
        method = method,
        rank = args.rank,
        max_iterations = args.iters,
        num_blocks = args.num_blocks,
        seed = args.seed,
        dtype = torch.float64 if args.float64 else torch.float32,
        allow_tf32 = args.allow_tf32 and not args.float64,
        device = None if args.device is None else torch.device(args.device),
        momentum = args.momentum,
        learning_rate = args.learning_rate,
        optimizer = optimizer,
        eval_interval = args.eval_interval,
        verbose = args.verbose,
        autotune = autotune_mode,
        autotune_cache_dir = args.autotune_cache_dir,
        autotune_verbose = args.autotune_verbose,
        input_filename = args.filename,
        log_filename = args.log_filename,
        eval_only = args.eval_only,
        eval_path = args.eval_path,
        eval_save = args.eval_save
    )
