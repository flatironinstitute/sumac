import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

#from torch_sparse import transpose, spmm, spspmm
#from torch_sparse.tensor import * 
import argparse 
import pathlib
import os
import random
import pickle
import time
import numpy as np


##helper functions for DDP torchrun
def _pick_nccl_ifname():
    # keep user override if already set
    if os.environ.get("NCCL_SOCKET_IFNAME"):
        return os.environ["NCCL_SOCKET_IFNAME"]
    # try to pick a real ethernet iface
    for cand in ("eth0", "eno1", "ens3f0", "ens5f0", "enp134s0"):
        if os.path.exists(f"/sys/class/net/{cand}"):
            return cand
    # fallback: exclude virtual/loopback/IB so NCCL auto-picks a real NIC
    return "^lo,docker0,virbr0,veth*,ib*"

def _ensure_nccl_env():
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    # If you do NOT use InfiniBand on this cluster, leave this enabled:
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ["NCCL_SOCKET_IFNAME"] = _pick_nccl_ifname()

def set_seed(seed):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def memory_stats():
    print(f'allocated memory {torch.cuda.memory_allocated()/1024**2}')
    print(f'reserved memory {torch.cuda.memory_reserved()/1024**2}')



class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = int(result[:, 1].argmax().item())  # cast to int s.b. safe for argmax, which returns index
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            return r
        