import torch 
import argparse 
import os
import numpy as np
import scipy.io as sio
from data import *
from torch.utils.data import DataLoader
import h5py
from sumac import sumac 
from _sumac.dataset import RowBlockDataset, collate_blocks
from _sumac.eval import eval 
import sys
from utils import _ensure_nccl_env

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    #torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_blocks', type=int, default=100, help='number of blocks')
    parser.add_argument('--filename', type=str, default='/mnt/home/dbollweg/sumac_data/bigrams_250K.mat')  #OLD: 'bigrams.txt'
    parser.add_argument('--momentum', type=float, default=0.9, help='use momentum to update W and H')
    parser.add_argument('--float64', action='store_true', help='use torch.float64 (default use float32)')
    parser.add_argument('--mode', type=str, default='GDlatent_sumac', choices=['ALS','SALSA','GD_sumac', 'GDlatent_sumac', 'GDlatent_prec_sumac'], help='default (sumac): alternating LS')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--optim', type=str, default='adam', choices=['adam','sgd', 'adamw', 'muon'], help='diffenrent optimizer for GD')
    parser.add_argument('--eval_only', action='store_true', help='eval only')
    parser.add_argument('--eval_path', type=str, default='bigrams_GD/sumac_d=16_mom=0.7_seed=0_iters=1000_ngpus=1_nblocks=None_finit=True_v2')  #OLD: 'bigrams.txt'
    parser.add_argument('--eval_save',  action='store_true', help='save to txt')  #OLD: 'bigrams.txt'
    parser.add_argument('--compile_cache_path', type=str, default='sumac_compile_cache')
    args = parser.parse_args()
    
    # log experiment configuration
    args_dict = vars(args)
    ngpus = torch.cuda.device_count()
    if args.eval_only == False:
        if args.mode == "GD_sumac":
            save_dir = "bigrams_GD_errZ"  
        elif args.mode == "GDlatent_sumac":
            save_dir = "bigrams_GDlatent_errZ"
        elif args.mode == "GDlatent_prec_sumac":
            save_dir = "bigrams_GDlatent_prec_errZ"
        elif args.mode == 'SALSA':
            save_dir = "bigrams_SALSA"
        else:
            save_dir = "bigrams"
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

    compile_artifact_file = "sumac_compile_artifacts.bin"
    if os.path.exists(args.compile_cache_path):

        artifact_bytes = open(os.path.join(args.compile_cache_path,compile_artifact_file), "rb").read()
        torch.compiler.load_cache_artifacts(artifact_bytes)
    
    # load data (3, E)
    if args.filename.endswith('.mat'):
        mat = sio.loadmat(args.filename)
        S = mat['bigrams']

        S = S.tocoo()
        row_sums = np.asarray(S.sum(axis=1)).ravel().astype(np.float32)

        realmin = np.finfo(np.float32).tiny

        # Normalize each nonzero entry by the sum of its row
        S_data = S.data.astype(np.float32) / (row_sums[S.row] + realmin)
    
        S_index = torch.tensor(np.array([S.row, S.col]), dtype=torch.long)
        S_value = torch.tensor(S_data, dtype=torch.float64 if args.float64 else torch.float32)
        m, n = S.shape

    else:
        data = np.loadtxt(args.filename).T

        S_index = torch.LongTensor(data[0:2,:])
        S_value = torch.FloatTensor(data[2,:])
        S_index -= 1 #This is only correct if the indexing is 1-based in text file! 
        m = int(S_index[0].max().item()) + 1
        n = int(S_index[1].max().item()) + 1

    print(f'm=n={m}, E={len(S_value)}')
   
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
        print("s_index and value from outside of main loop:")
        print(S_index[:, 0], S_value[0])
        print(S_index[:, -1], S_value[-1])
        sumac(S_index, S_value, m=m, n=m, d=args.d, 
                max_iterate=args.iters, factor_init=True,
                num_blocks=args.num_blocks, mom=args.momentum,
                method=method, lr=args.lr, save_path=save_path,
                GD_latent=GD_latent, optim=args.optim, precondition=precond, seed=args.seed)

        sys.stdout = old_stdout
        log_file.close()
    
    if not os.path.exists(args.compile_cache_path):
            os.makedirs(args.compile_cache_path, exist_ok=True)
    artifact_bytes, cache_info = torch.compiler.save_cache_artifacts()
    open(os.path.join(args.compile_cache_path,compile_artifact_file), "wb").write(artifact_bytes)

##launch scripts
#GPU: python sumac_bigrams.py --iters 1000  --num_blocks 100 
#EVAL: python sumac_bigrams.py --eval_only