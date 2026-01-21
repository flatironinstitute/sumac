from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import math
import time
import random
from torch.nn.utils import clip_grad_norm_
import contextlib

from _sumac.dataset import block_span, RowBlockDataset

# ---------- config (uses num_blocks) ----------
@dataclass
class TrainConfig:
    d: int = 64
    num_blocks: int = 64       # NEW: number of row blocks
    epochs: int = 10
    lr: float = 1e-2
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    shuffle_blocks: bool = False
    batch_blocks: int = 1      # number of blocks per optimizer step ##TODO: multi-gpu check (use 4 for multi)
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 0
    eval_errZ_interval: int = 5 #eval errZ every 5 epochs
    optim: str = "adam"
    SGD_mom: float = 0.7
    precondition: bool = False #precondition
    prec_eps: float = 1e-6


#ablate the choice of optimizer
def make_optimizer(name, params, lr, weight_decay=0.0, SGD_momentum=0.0):
    name = name.lower()
    if name == "adam": 
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=SGD_momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


#main training loop; update [A,B] per block
def block_loss_and_pred(
    A: torch.Tensor,                 # (m, d)
    B: torch.Tensor,                 # (n, d)
    block_id: int,
    num_blocks: int,
    m: int,
    n: int,
    S_index: torch.LongTensor,       # (2, nnz)
    S_value: torch.Tensor,           # (nnz,)
    edge_idx: torch.Tensor,          # indices into S_index/S_value for this block
    errZ_obj: bool = False,          # whether use objective to min ||Z - L|| instead of ||S - Sr||
):
    """
    - Builds full block prediction Sr_I = ReLU(A_I @ B^T) (shape b x n).
    - Target is zero everywhere, except at observed entries (from edge_idx).
    - Loss: RMSE over the entire b*n matrix (sqrt of mean squared error) — used for backprop.
    - Metrics: returns scalar rmse (same as loss) and 1-Jaccard on the block.
    """
    start, end = block_span(block_id, m, num_blocks)
    b = end - start
    assert b > 0, "Empty block span"

    # 1) Dense prediction for the block vs all columns (all-zero negatives)
    A_block = A[start:end, :]                # (b, d)
    L = A_block @ B.T
    Sr_block = torch.clamp(L, min=0.0)   # (b, n)
    sumSr_block = Sr_block.sum() #float(Sr_block.sum().item())

    # 2) Sparse prediction for the block (at the edge index)
    target = Sr_block.new_zeros((b, n))      # (b, n)
    rows_all = S_index[0][edge_idx]         # global rows in [start, end)
    cols_all = S_index[1][edge_idx]         # global cols
    vals_all = S_value[edge_idx]            # (E_b,)
    local_r = rows_all - start              # shift to [0, b)
    target[local_r, cols_all] = vals_all

    # 3) MSE/jacc numerator over *all* entries in the block (loss used for backprop)
    mse_full = F.mse_loss(Sr_block, target, reduction="sum")  # sum over all b*n -> dense compute
    Sr_obs = Sr_block[local_r, cols_all]
    # ssqS_block = (vals_all * vals_all).sum()
    # mse_full = ssqS_block + ssqSr_block - 2.0 * (vals_all * Sr_obs).sum() #this turns out to be slightly slower
    jacc_num_block = torch.minimum(vals_all, Sr_obs).sum()
    errZ_num_block = None
    if errZ_obj: #make equivalent errZ objective; TODO: faster?
        # L at observed (local) coordinates
        L_obs = L[local_r, cols_all]              # (E_b,)
        neg_mask_pos = L_obs < 0                  # only entries that were clamped in Sr
        if neg_mask_pos.any():
            S_obs = vals_all[neg_mask_pos]        # S_{ij} at those coords
            L_neg = L_obs[neg_mask_pos]           # L_{ij} (negative values)
            # sum of (L^2 - 2 S L) over the intersection (observed & L<0)
            errZ_num_block = mse_full + (L_neg*L_neg - 2.0*S_obs*L_neg).sum()
   
    return mse_full, sumSr_block, jacc_num_block, errZ_num_block

##main eval code; reusing block_loss_and_pred() to compute metric
@torch.no_grad()
def eval(
    A, B, S_index, S_value,
    m, n, num_blocks,
    full_block_loader,   # yields (block_id, edge_idx) once per block_id
    device=None,
):
    ssqe = torch.zeros((), device=device, dtype=A.dtype)
    sumSr = torch.zeros((), device=device, dtype=A.dtype)
    num_j = torch.zeros((), device=device, dtype=A.dtype)
    errZ_num = torch.zeros((), device=device, dtype=A.dtype)

    for block in full_block_loader:  
        for (block_id, edge_idx) in block:
            edge_idx = edge_idx.to(device).view(-1)
            block_id = int(block_id)
            ssqe_b, sumSr_b, num_j_b, errZ_b = block_loss_and_pred(
                A, B, block_id, num_blocks, m, n,
                S_index, S_value, edge_idx
            )
            ssqe += ssqe_b
            sumSr += sumSr_b
            num_j += num_j_b
            errZ_num += errZ_b if errZ_b is not None else ssqe_b 
    
    sumSr = float(sumSr.item())
    S_norm = torch.norm(S_value)
    rmse = torch.sqrt(ssqe) / (S_norm + 1e-16)
    denom = S_value.sum() + sumSr - num_j
    jacc = 1.0 - num_j / (denom + 1e-16)
    errZ = torch.sqrt(errZ_num) / (S_norm + 1e-16)

    return float(rmse.item()), float(jacc.item()), float(errZ.item())

###Helpers for GD_loop
def select_devices(cfg, device: torch.device):
    """Return list of devices to use (master first). No DDP."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device.type == "cuda":
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [device]

def setup_replicas(A, B, S_index, S_value, devices):
    """
    Create per-device parameter replicas and per-device copies of S_index/S_value.
    Master params are the original A,B (tied to optimizer).
    """
    master = devices[0]
    S_index_devs = [S_index.to(dev, non_blocking=True) for dev in devices]
    S_value_devs = [S_value.to(dev, non_blocking=True) for dev in devices]

    A_devs = [A] + [torch.nn.Parameter(A.detach().to(dev)) for dev in devices[1:]]
    B_devs = [B] + [torch.nn.Parameter(B.detach().to(dev)) for dev in devices[1:]]

    streams = [torch.cuda.Stream(device=dev) if dev.type == "cuda" else None for dev in devices]
    return A_devs, B_devs, S_index_devs, S_value_devs, streams


def shard_blocks(blocks, num_shards: int):
    """Round-robin split of list[(block_id, edge_idx)] into num_shards lists."""
    shards = [[] for _ in range(num_shards)]
    for i, item in enumerate(blocks):
        shards[i % num_shards].append(item)
    return shards


def zero_replica_grads(opt, A_devs, B_devs):
    """Zero master grads via optimizer, and manually clear replica grads."""
    opt.zero_grad(set_to_none=True)
    for di in range(1, len(A_devs)):
        if A_devs[di].grad is not None:
            A_devs[di].grad = None
        if B_devs[di].grad is not None:
            B_devs[di].grad = None


def compute_backward_on_device(
    di: int,
    dev: torch.device,
    shard,
    A_devs, B_devs, S_index_devs, S_value_devs,
    streams,
    cfg, m, n,
    GD_latent: bool,
    block_loss_and_pred,
):
    """
    Compute loss + backward for a shard on a single device.
    Returns (loss_tensor, sumSr_tensor, numj_tensor) on that device.
    """
    if len(shard) == 0:
        z = torch.zeros((), device=dev, dtype=A_devs[0].dtype)
        return z, z, z

    stream = streams[di]
    ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with ctx:
        loss = torch.tensor(0.0, device=dev)
        sumSr = torch.tensor(0.0, device=dev)
        numj  = torch.tensor(0.0, device=dev)

        A_d = A_devs[di]
        B_d = B_devs[di]
        S_idx_d = S_index_devs[di]
        S_val_d = S_value_devs[di]

        for (block_id, edge_idx) in shard:
            block_id = int(block_id)
            edge_idx = edge_idx.to(dev).view(-1)

            mse_block, sumSr_block, jacc_num_block, errZ_block = block_loss_and_pred(
                A_d, B_d,
                block_id=block_id, num_blocks=cfg.num_blocks, m=m, n=n,
                S_index=S_idx_d, S_value=S_val_d, edge_idx=edge_idx,
                errZ_obj=GD_latent,
            )
            loss_block = errZ_block if errZ_block is not None else mse_block
            loss = loss + loss_block

            # block_loss_and_pred returns sumSr_block as Python float -> keep semantics
            sumSr = sumSr + torch.as_tensor(sumSr_block, device=dev, dtype=A_devs[0].dtype)
            numj  = numj + jacc_num_block

        loss.backward()
        return loss, sumSr, numj


def sync_all(devices):
    """Synchronize all CUDA devices (no-op on CPU)."""
    if devices[0].type == "cuda":
        for dev in devices:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

def wait_streams_before_reduce(devices, streams):
    """
    ### NEW
    Ensure all work enqueued on streams[di] is complete *w.r.t.* the default stream
    before we read/copy grads on that device.
    This replaces sync_all(devices); typically faster
    """
    for di, dev in enumerate(devices):
        if dev.type == "cuda" and streams[di] is not None:
            # default stream on that device waits for the replica stream
            torch.cuda.current_stream(dev).wait_stream(streams[di])

@torch.no_grad()
def reduce_grads_to_master(A, B, A_devs, B_devs, master: torch.device, average: bool = True):
    """
    Pattern 2: device 0 participates in backward (A_devs[0] is master param).
    Assumes:
      - opt.zero_grad(set_to_none=True) was called
      - compute_backward_on_device ran for di=0 and may have created A.grad/B.grad
      - replicas di>=1 have grads in A_devs[di].grad/B_devs[di].grad (or None)
    """

    # Ensure master grads exist (if shard[0] empty, grads might still be None)
    if A.grad is None:
        A.grad = torch.zeros_like(A, device=master)
        contrib = 0
    else:
        contrib = 1  # device 0 contributed

    if B.grad is None:
        B.grad = torch.zeros_like(B, device=master)
        # contrib already handled above; don't double-count
    else:
        contrib = max(contrib, 1)

    # Accumulate replica grads onto master
    for di in range(1, len(A_devs)):
        gA = A_devs[di].grad
        gB = B_devs[di].grad
        if gA is None and gB is None:
            continue

        if gA is not None:
            A.grad.add_(gA.detach().to(master, non_blocking=False))
        if gB is not None:
            B.grad.add_(gB.detach().to(master, non_blocking=False))

        contrib += 1

    # Average over contributing devices (not necessarily len(A_devs) if some shards empty)
    if average:
        denom = max(contrib, 1)
        A.grad.mul_(1.0 / denom)
        B.grad.mul_(1.0 / denom)


@torch.no_grad()
def broadcast_params_from_master(A, B, A_devs, B_devs, devices):
    """Copy updated master params to replicas."""
    for di in range(1, len(devices)):
        A_devs[di].copy_(A.detach().to(devices[di]))
        B_devs[di].copy_(B.detach().to(devices[di]))


def apply_precondition(A, B, cfg, master: torch.device):
    """Your existing right-preconditioning, unchanged."""
    if not cfg.precondition:
        return
    with torch.no_grad():
        I = torch.eye(cfg.d, device=master, dtype=A.dtype)
        G_B = B.T @ B + cfg.prec_eps * I
        G_A = A.T @ A + cfg.prec_eps * I
        L_B = torch.linalg.cholesky(G_B)
        L_A = torch.linalg.cholesky(G_A)
        if A.grad is not None:
            A.grad.copy_(torch.cholesky_solve(A.grad.T.contiguous(), L_B).T)
        if B.grad is not None:
            B.grad.copy_(torch.cholesky_solve(B.grad.T.contiguous(), L_A).T)


def apply_clip_and_step(opt, A, B, cfg):
    """Your existing clip+step, unchanged."""
    if cfg.optim.lower() == 'sgd':
        clip_grad_norm_([A, B], max_norm=1.0)
    opt.step()
