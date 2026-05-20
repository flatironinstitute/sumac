import torch 
import argparse 
import os
import pickle
import numpy as np

from data import *
from torch.utils.data import Dataset, DataLoader, Subset

from sumac import sumac 
from _sumac.dataset import block_span, RowBlockDataset, collate_blocks
from _sumac.eval import eval 
import sys
from utils import _ensure_nccl_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_blocks', type=int, default=100, help='number of blocks')
    parser.add_argument('--filename', type=str, default='/mnt/home/lsaul/Datasets/flywire/connectome.txt')  #OLD: 'connectome.txt'
    parser.add_argument('--momentum', type=float, default=0.9, help='use momentum to update W and H')
    parser.add_argument('--float64', action='store_true', help='use torch.float64 (default use float32)')
    parser.add_argument('--mode', type=str, default='GDlatent_sumac', choices=['ALS','SALSA','GD_sumac', 'GDlatent_sumac', 'GDlatent_prec_sumac'], help='default (sumac): alternating LS')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='adam', choices=['adam','sgd', 'adamw', 'muon'], help='diffenrent optimizer for GD')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='clean-eval cadence (default: 100 for GD, 10 for SALSA/ALS)')
    parser.add_argument('--eval_only', action='store_true', help='eval only')
    parser.add_argument('--eval_path', type=str, default='connectome_GD/sumac_d=16_mom=0.7_seed=0_iters=1000_ngpus=1_nblocks=None_finit=True_v2')  #OLD: 'connectome.txt'
    parser.add_argument('--eval_save',  action='store_true', help='save to txt')  #OLD: 'connectome.txt'

    args = parser.parse_args()
    # log experiment configuration
    args_dict = vars(args)
    ngpus = torch.cuda.device_count()
    if args.eval_only == False:
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
        save_path = f"./{save_dir}/sumac_d={args.d}_mom={args.momentum}_seed={args.seed}_iters={args.iters}_ngpus={ngpus}_nblocks={args.num_blocks}_lr={args.lr}_optim={args.optim}_v2" #v2: dated 11/04/2025, testing new changes to matlab
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        # logging
        old_stdout = sys.stdout
        log_file = open(f"{save_path}/experiment.log","w")
        sys.stdout = log_file
        # print configs
        print('Experiment Setting:')
        for key, value in args_dict.items():
            print(f"| {key}: {value}")
    # load data (3, E)
    data = np.loadtxt(args.filename).T
    S_index = torch.LongTensor(data[0:2,:])
    S_value = torch.FloatTensor(data[2,:])
    m = n = int(S_index[0].max())
    print(f'm=n={m}, E={len(S_value)}')
    # normalize to start at zero-index
    S_index -= 1
    dtype = torch.float64 if args.float64 else torch.float32
    # NEW: EVAL ONLY
    if args.eval_only:
        A, B = torch.load(f"{args.eval_path}/AB.pt")
        ds = RowBlockDataset(S_index, S_value, m=m, num_blocks=args.num_blocks)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blocks)
        rmse, jacc, errZ = eval(A.detach().cpu(), B.detach().cpu(), S_index, S_value, 
                            m, n, num_blocks=args.num_blocks, 
                            full_block_loader=loader, device="cpu")
        print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")
        #saving to txt 
        if args.eval_save:
            print(A.shape, B.shape)
            np.savetxt(f"{args.eval_path}/A.txt", A.cpu().numpy(), fmt="%.8f")
            np.savetxt(f"{args.eval_path}/B.txt", B.cpu().numpy(), fmt="%.8f")

    else: #TRAIN FROM SCRATCH
        method = args.mode if args.mode in ['ALS', 'SALSA'] else 'GD'
        GD_latent = True if args.mode in ['GDlatent_sumac', 'GDlatent_prec_sumac'] else False
        precond = True if args.mode == 'GDlatent_prec_sumac' else False
        if method == 'GD':
            _ensure_nccl_env() #for cluster job run
            print("NCCL_SOCKET_IFNAME=", os.environ.get("NCCL_SOCKET_IFNAME"),
                " MASTER_ADDR=", os.environ.get("MASTER_ADDR"),
                " MASTER_PORT=", os.environ.get("MASTER_PORT"))

        opts = {'eval_interval': args.eval_interval} if args.eval_interval is not None else None
        sumac(S_index, S_value, m=m, n=m, d=args.d,
                max_iterate=args.iters, factor_init=True,
                num_blocks=args.num_blocks, mom=args.momentum,
                method=method, lr=args.lr, save_path=save_path,
                GD_latent=GD_latent, optim=args.optim, precondition=precond,
                opts=opts)

        sys.stdout = old_stdout
        log_file.close()

##launch scripts
#GPU: python sumac_connectome.py --iters 1000  --num_blocks 100 
#EVAL: python sumac_connectome.py --eval_only