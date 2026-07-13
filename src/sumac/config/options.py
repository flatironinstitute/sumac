import argparse
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import torch

class SumacMethod(Enum):
    SALSA = 'salsa'
    GD = 'gd'


class OptimizerName(Enum):
    ADAM = 'adam'
    ADAMW = 'adamw'
    SGD = 'sgd'
    MUON = 'muon'


# TODO: need to check if we need to support cuda, cpu, etc.
class AutotuneMode(Enum):
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
    method: SumacMethod = SumacMethod.SALSA
    rank: int = 16
    max_iterations: int = 25
    num_blocks: int | None = None
    seed: int | None = None
    cache_mb: int = 5000
    dtype: torch.dtype = torch.float32
    allow_tf32: bool = False
    device: torch.device | None = None
    momentum: float = 0.7
    learning_rate: float = 1e-1
    optimizer: OptimizerName = OptimizerName.ADAM
    # optimizer-specific
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    eval_interval: int | None = None
    verbose: bool = True
    autotune: AutotuneMode = AutotuneMode.CACHE
    autotune_cache_dir: str | Path | None = None
    autotune_verbose: bool = False
    # consumed by examples
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


