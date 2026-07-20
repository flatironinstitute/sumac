import argparse
import torch

from .options import SumacConfig, SumacMethod, OptimizerName, AutotuneMode

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
    )
