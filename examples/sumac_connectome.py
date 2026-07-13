import torch 
import argparse 
import os
import pickle
import numpy as np
import scipy.io as sio
from sumac.data import *
from torch.utils.data import DataLoader
from sumac import sumac_factorize
from sumac.datasets import RowBlockDataset, collate_blocks
from sumac.eval import eval
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_blocks', type=int, default=100, help='number of blocks')
    parser.add_argument('--filename', type=str)
    parser.add_argument('--momentum', type=float, default=0.9, help='use momentum to update W and H')
    parser.add_argument('--float64', action='store_true', help='use torch.float64 (default use float32)')
    parser.add_argument('--mode', type=str, default='SALSA', choices=['SALSA', 'GD'], help='default (sumac): SALSA')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='adam', choices=['adam','sgd', 'adamw', 'muon'], help='Optimizer to use for GD')
    parser.add_argument('--eval_interval', type=int, default=None, help='Evaluation interval (default: 100 for GD, 10 for SALSA)')
    parser.add_argument('--log_filename', type=str, default=None, help="filename for logging. If unset, stdout is used.")
    parser.add_argument('--allow_tf32', action='store_true', help='allow PyTorch and SUMAC custom kernels to use TF32.')
    parser.add_argument('--autotune', type=str, default='cache', choices=['cache', 'force', 'disable', 'fallback'], help='CUDA kernel autotuning mode.')
    parser.add_argument('--autotune_cache_dir', type=str, default=None, help='Directory for SUMAC kernel autotune cache files.')
    parser.add_argument('--autotune_verbose', action='store_true', help='Print CUDA kernel autotuning decisions.')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='training/evaluation device')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high' if args.allow_tf32 else 'highest')

    # log experiment configuration
    old_stdout = sys.stdout
    log_file = None

    if args.mode == "GD":
        save_dir = "connectome_GD"
    elif args.mode == 'SALSA':
        save_dir = "connectome_SALSA"
    else:
        save_dir = "connectome"
    save_path = f"./{save_dir}/sumac_rank={args.rank}_mom={args.momentum}_seed={args.seed}_iters={args.iters}_nblocks={args.num_blocks}_learning_rate={args.learning_rate}_optim={args.optim}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # logging
    if args.log_filename is not None:
        log_dir = os.path.dirname(args.log_filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = open(args.log_filename, "w")
        sys.stdout = log_file

    
    if args.filename.endswith('.mat'):
        mat = sio.loadmat(args.filename)
        S = mat['connectome']

        S = S.tocoo()

        S_index = torch.tensor(np.array([S.row, S.col]), dtype=torch.long)
        S_value = torch.tensor(S.data, dtype=torch.float64 if args.float64 else torch.float32)
        m, n = S.shape
        assert isinstance(m, int)
        assert isinstance(n, int)
        
    else:
        data = np.loadtxt(args.filename).T

        S_index = torch.LongTensor(data[0:2,:])
        S_value = torch.FloatTensor(data[2,:])

        m = n = int(S_index[0].max())
        print(f'm=n={m}, E={len(S_value)}')

        # normalize to start at zero-index
        S_index -= 1

    A, B, costs = sumac_factorize(
        S_index=S_index,
        S_value=S_value,
        shape=(m, n),
        rank=args.rank,
        max_iterations=args.iters,
        num_blocks=args.num_blocks,
        momentum=args.momentum,
        method=args.mode,
        learning_rate=args.learning_rate,
        optimizer=args.optim,
        eval_interval=args.eval_interval,
        seed=args.seed,
        allow_tf32=args.allow_tf32,
        device=args.device,
        autotune=args.autotune,
        autotune_cache_dir=args.autotune_cache_dir,
        autotune_verbose=args.autotune_verbose,
    )
    torch.save([A, B], f"{save_path}/AB.pt")
    with open(f"{save_path}/cost.pkl", "wb") as f:
        pickle.dump(costs, f)
    with open(f"{save_path}/opts.pkl", "wb") as f:
        pickle.dump(vars(args), f)

    eval_device = A.device
    A_eval = A.detach()
    B_eval = B.detach()
    S_index_eval = S_index.to(eval_device)
    S_value_eval = S_value.to(eval_device)

    ds = RowBlockDataset(
        S_index_eval,
        S_value_eval,
        m=m,
        num_blocks=args.num_blocks,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blocks)
    rmse, jacc, errZ = eval(
        A_eval,
        B_eval,
        S_index_eval,
        S_value_eval,
        m,
        n,
        num_blocks=args.num_blocks,
        full_block_loader=loader,
        device=eval_device,
    )
    print(f"Final metrics: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")

    if log_file is not None:
        sys.stdout = old_stdout
        log_file.close()

##launch scripts
#GPU: python sumac_connectome.py --iters 1000  --num_blocks 100 
