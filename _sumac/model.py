import torch
import torch.nn as nn
from torch_sparse.tensor import * 
import torch.nn.functional as F


class FactorModel(nn.Module):
    # -----------------------
    # tiny module for DDP multi-gpu training
    # -----------------------
    def __init__(self, m: int, n: int, d: int, init_scale: float, device):
        super().__init__()
        self.A = nn.Parameter((torch.rand(m, d, device=device) - 0.5) * init_scale) #[-0.5, 0.5] * scale
        self.B = nn.Parameter((torch.rand(n, d, device=device) - 0.5) * init_scale) #[-0.5, 0.5] * scale
