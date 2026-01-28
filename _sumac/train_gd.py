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

from _sumac.dataset import block_span

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
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    precondition: bool = False #precondition
    prec_eps: float = 1e-6
    muon_momentum: float = 0.95


#ablate the choice of optimizer
def make_optimizer(name, params, lr, weight_decay=0.0, SGD_momentum=0.0, adam_betas=(0.9, 0.999), adam_eps=1e-8, muon_momentum=0.95):
    name = name.lower()
    if name == "adam": 
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=adam_betas, eps=adam_eps)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=adam_betas, eps=adam_eps)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=SGD_momentum, weight_decay=weight_decay)
    elif name == "muon":
        return torch.optim.Muon(params, lr=lr, weight_decay=weight_decay, 
                                momentum=muon_momentum, adjust_lr_fn="match_rms_adamw")
    else:
        raise ValueError(f"Unknown optimizer: {name}")

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

        #for (block_id, edge_idx) in shard:
        for (block_id, edge_idx, row_indices) in shard:
            block_id = int(block_id)
            edge_idx = edge_idx.to(dev).view(-1)

            mse_block, sumSr_block, jacc_num_block, errZ_block = block_loss_and_pred(
                A_d, B_d,
                block_id=block_id, num_blocks=cfg.num_blocks, m=m, n=n,
                S_index=S_idx_d, S_value=S_val_d, edge_idx=edge_idx,
                row_indices=row_indices, errZ_obj=GD_latent,
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
