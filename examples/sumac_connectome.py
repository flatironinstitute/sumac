import torch 
import argparse 
import os
import numpy as np
import scipy.io as sio
from sumac.data import *
from torch.utils.data import DataLoader
from sumac import sumac 
from sumac.datasets import RowBlockDataset, collate_blocks
from sumac.eval import eval
import sys
from sumac.utils import _ensure_nccl_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_blocks', type=int, default=100, help='number of blocks')
    parser.add_argument('--filename', type=str)
    parser.add_argument('--momentum', type=float, default=0.9, help='use momentum to update W and H')
    parser.add_argument('--float64', action='store_true', help='use torch.float64 (default use float32)')
    parser.add_argument('--mode', type=str, default='SALSA', choices=['ALS','SALSA','GD_sumac', 'GDlatent_sumac', 'GDlatent_prec_sumac'], help='default (sumac): alternating LS')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='adam', choices=['adam','sgd', 'adamw', 'muon'], help='Optimizer to use for GD')
    parser.add_argument('--eval_interval', type=int, default=None, help='Evaluation interval (default: 100 for GD, 10 for SALSA)')
    parser.add_argument('--log_filename', type=str, default=None, help="filename for logging. If unset, stdout is used.")
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate metrics for factors saved in eval_path, no training is performed if set.')
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--eval_save',  action='store_true', help='Save evaluation results to .txt file.')
    parser.add_argument('--allow_tf32', action='store_true', help='allow PyTorch and SUMAC custom kernels to use TF32.')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high' if args.allow_tf32 else 'highest')

    # log experiment configuration
    args_dict = vars(args)
    ngpus = torch.cuda.device_count()
    old_stdout = sys.stdout
    log_file = None

    if not args.eval_only:
        if args.mode == "GD_sumac":
            save_dir = "connectome_GD_errZ"  
        elif args.mode == "GDlatent_sumac":
            save_dir = "connectome_GDlatent_errZ"
        elif args.mode == "GDlatent_prec_sumac":
            save_dir = "connectome_GDlatent_prec_errZ"
        elif args.mode == 'SALSA':
            save_dir = "connectome_SALSA"
        else:
            save_dir = "connectome"
        save_path = f"./{save_dir}/sumac_rank={args.rank}_mom={args.momentum}_seed={args.seed}_iters={args.iters}_ngpus={ngpus}_nblocks={args.num_blocks}_learning_rate={args.learning_rate}_optim={args.optim}"
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

    dtype = torch.float64 if args.float64 else torch.float32
    if not args.eval_only:
        method = args.mode if args.mode in ['ALS', 'SALSA'] else 'GD'
        GD_latent = args.mode in ['GDlatent_sumac', 'GDlatent_prec_sumac']
        precond = args.mode == 'GDlatent_prec_sumac'
        if method == 'GD':
            _ensure_nccl_env() #for cluster job run
            print("NCCL_SOCKET_IFNAME=", os.environ.get("NCCL_SOCKET_IFNAME"),
                " MASTER_ADDR=", os.environ.get("MASTER_ADDR"),
                " MASTER_PORT=", os.environ.get("MASTER_PORT"))

        opts = {'eval_interval': args.eval_interval} if args.eval_interval is not None else None
        A, B, costs = sumac(
            S_index,
            S_value,
            m=m,
            n=n,
            d=args.rank,
            max_iterate=args.iters,
            factor_init=True,
            num_blocks=args.num_blocks,
            sgd_momentum=args.momentum,
            method=method,
            lr=args.learning_rate,
            save_path=save_path,
            GD_latent=GD_latent,
            optim=args.optim,
            precondition=precond,
            opts=opts,
            seed=args.seed,
            allow_TF32=args.allow_tf32
        )

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

    else:
        eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A, B = torch.load(f"{args.eval_path}/AB.pt", map_location="cpu")
        A_eval = A.detach().to(eval_device)
        B_eval = B.detach().to(eval_device)
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
        print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")
        #saving to txt
        if args.eval_save:
            print(A.shape, B.shape)
            np.savetxt(f"{args.eval_path}/A.txt", A.cpu().numpy(), fmt="%.8f")
            np.savetxt(f"{args.eval_path}/B.txt", B.cpu().numpy(), fmt="%.8f")

##launch scripts
#GPU: python sumac_connectome.py --iters 1000  --num_blocks 100 
#EVAL: python sumac_connectome.py --eval_only
