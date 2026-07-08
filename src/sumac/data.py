import torch
import torch.nn.functional as F

from .utils import *


def generate_testdata(seed, m, n, r=3, val_max=10):
    '''
    m: number of rows
    n: number of columns
    E: number of nonzero entries (+) in the sparse matrix (m,n)
    '''
    set_seed(seed)
    W = torch.randint(-val_max, val_max, (m, r))
    H = torch.randint(1, val_max, (r, n))
    S = F.relu(W @ H)
    ##TODO: delete all zero rows to avoid degeneracy
    nonzero_mask = S.abs().sum(dim=1) != 0
    return S[nonzero_mask], W[nonzero_mask], H

def dense_to_sparse(S, sparse_rate=True):
    '''
    S: sparse matrix stored in a dense format (torch)
    Return: idx, val of the sparse matrix
    '''
    S_sp = S.to_sparse(layout=torch.sparse_coo)
    idx = S_sp.indices()
    val = S_sp.values()
    if sparse_rate:
        m,n = S.shape
        rate = len(val) / (m*n)
        print(f'sparse rate = {rate}')
    return idx, val.float()

def sparse_to_dense(idx: torch.Tensor, val: torch.Tensor, m: int, n: int):
    '''
    Given idx, val (COO) format, return a dense matrix
    '''
    s = torch.sparse_coo_tensor(idx, val, size=(m,n))
    return s.to_dense()

def prune_zero_rows_cols(S_index, shape):
    r, c = S_index
    m, n = shape
    if r.unique().numel() == m and c.unique().numel() == n:
        print(f"no all-zero row or col")
        return S_index, None, None # nothing to drop
    dev = S_index.device
    row_mask = torch.zeros(m, dtype=torch.bool, device=dev)
    row_mask[r] = True
    col_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    col_mask[c] = True
    row_new = torch.full((m,), -1, dtype=torch.long, device=dev)
    row_new[row_mask] = torch.arange(row_mask.sum().item(), device=dev)
    col_new = torch.full((n,), -1, dtype=torch.long, device=dev)
    col_new[col_mask] = torch.arange(col_mask.sum().item(), device=dev)
    new_idx = torch.stack([row_new[r], col_new[c]])
    print(f"{int(row_mask.sum())} rows, {int(col_mask.sum())} cols")
    return new_idx, row_mask, col_mask
