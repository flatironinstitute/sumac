import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import math
import time
import random
from data import dense_to_sparse, prune_zero_rows_cols
from dataclasses import dataclass
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
import pickle
from typing import Tuple

from _sumac.dataset import block_span, RowBlockDataset, collate_blocks
from _sumac.train_gd import TrainConfig, make_optimizer, block_loss_and_pred, eval, \
                            select_devices, setup_replicas, shard_blocks, \
                            zero_replica_grads, compute_backward_on_device, sync_all, wait_streams_before_reduce, \
                            reduce_grads_to_master, broadcast_params_from_master, apply_precondition, apply_clip_and_step
from _sumac.train_als import least_squares_update_fast, least_squares_update, refactor


def sumac(S_index, S_value, m, n, d, max_iterate=25, num_blocks=None,
          test_flag=False, dtype=torch.float32, opts=None, 
          mom=0.7, fast_flag=True, use_GD=False, lr=1e-1, factor_init=False,
          save_path=None, GD_latent=False, optim="adam", precondition=False):
    """
    PyTorch version of sumac algorithm driver function.

    Args:
      S_index : sparse index of (m, n)
      S_value : values at the indices
      d : target rank (int)
      max_iterate: maximum number of iteration
      test_flag: if true, run the test case for debugging
      dtype: default to torch.float32, accept torch.float64
      opts : (optional) dict of options; missing keys will be filled with defaults
      fast_flag: default True, use the fast alternating least squares
      use_GD: default False, use (block) gradient descent
      factor_init: default False, use per factor (A, B) specific scaling at initialiation
      GD_latent: default False. If true, correct the GD MSE object to align with ALS
      precondition: default False. If true, precondition the gradient of factors by A.grad = A.grad @ (B^T B)^{-1}
    Returns:
      A, B    : factors (torch.Tensors)
      costs   : list or tensor of cost history
      opts    : dict of options used
    """

    # verify input nonnegativity
    if (S_value < 0).any().item():
        raise ValueError("sumac: the input matrix should be nonnegative.")

    # default options / user-provided opts
    opts_default = {
        'time_limit': float('inf'),
        'tol_abs': 1e-2,
        'tol_rel': 1e-4,
        'tol_window': 20,
        'seed': random.randint(0, 2**31 - 1),
        'display': 1,
        'cache_MB': 5000,
        'exaggerate': mom, #0.7,
        'momentum_start_iter': 10,
        'refactor_interval': 25,
        'optim': optim, #default adam, can try sgd, adamw etc
        'max_iterate': max_iterate,
        'dtype': dtype,
        'factor_init': factor_init #if True: A,B specific initialization 
    }

    if opts is None:
        opts = opts_default.copy()
    else:
        # fill in any missing keys
        for k, v in opts_default.items():
            opts.setdefault(k, v)

    if num_blocks is None:
        max_bytes = opts['cache_MB'] * 1e6
        bytes_per_dtype = 8 if dtype == torch.float64 else 4
        cols_per_block = max(1, int(max_bytes // (m * bytes_per_dtype)))
        num_blocks   = math.ceil(n / cols_per_block)
    else:
        cols_per_block = int(n // num_blocks)
    opts['num_blocks'] = num_blocks
    opts['cols_per_block'] = cols_per_block

    # display info if desired
    if opts['display']:
        nnz = len(S_value)
        print(f"\n  Input to SUMAC is {m}×{n} sparse matrix with {nnz} nonzeros.")
        print(f"  Attempting to complete with rank {d}.")
        print(f"  Available GPUs: {torch.cuda.device_count()}.")
        print("  Options:")
        for k, v in opts.items():
            print(f"    {k}: {v}")
        print()

    ## NEW: remove all-zero rows and/or columns
    S_index, row_mask, col_mask = prune_zero_rows_cols(S_index, shape=(m,n))
    m_eff = row_mask.sum() if row_mask is not None else m
    n_eff = col_mask.sum() if col_mask is not None else n

    # set random seed for reproducibility
    random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])

    # call the core SUMAC loop 
    S_value = S_value.to(dtype)
    if use_GD:
        cfg = TrainConfig(d, num_blocks=num_blocks, epochs=max_iterate, lr=lr,
                          optim=optim, SGD_mom=mom, precondition=precondition)
        ## NEW: for multi-gpus, scale batch blocks and lr automatically
        if torch.cuda.device_count() > 1:
            cfg.batch_blocks = torch.cuda.device_count()
            cfg.lr = cfg.lr * torch.cuda.device_count() * 0.75 #TODO: test / better heuristic 
        A, B, costs = GD_loop(S_index, S_value, m_eff, n_eff, cfg, GD_latent)
    
    else:
        A, B, costs = sumac_loop(S_index, S_value, m_eff, n_eff, d, opts, test_flag, fast_flag)

    # NEW: restore zero rows and columns
    if row_mask is not None:
        A_ori = torch.zeros((m, d), dtype=A.dtype, device=A.device)
        A_ori[row_mask] = A 
    else:
        A_ori = A
    if col_mask is not None:
        B_ori = torch.zeros((n, d), dtype=B.dtype, device=B.device)
        B_ori[col_mask] = B 
    else:
        B_ori = B
    ## save
    if save_path is not None:
        torch.save([A_ori,B_ori], f"{save_path}/AB.pt")
        pickle.dump(costs, open(f"{save_path}/cost.pkl", "wb"))
        pickle.dump(opts, open(f"{save_path}/opts.pkl", "wb"))
    print(f'finish for {max_iterate} iterations!')

def GD_loop(
    S_index: torch.LongTensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    cfg: TrainConfig,
    GD_latent: bool = False,
):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device) #cfg.device = "cuda:0" by default, aka the master device

    # Move base data to master device first (replicas created later)
    S_index = S_index.to(device)
    S_value = S_value.to(device)

    # Parameters on master
    scale = 0.5 * math.sqrt(S_value.mean().item() / cfg.d)
    A = torch.nn.Parameter(torch.rand(m, cfg.d, device=device) * scale)
    B = torch.nn.Parameter(torch.rand(n, cfg.d, device=device) * scale)
    opt = make_optimizer(cfg.optim, [A, B], lr=cfg.lr, weight_decay=0.0, SGD_momentum=cfg.SGD_mom)

    # DataLoader
    ds = RowBlockDataset(S_index, S_value, m=m, num_blocks=cfg.num_blocks)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_blocks,
        shuffle=cfg.shuffle_blocks,
        collate_fn=collate_blocks
    )

    # Multi-GPU setup (manual DP, no DDP)
    devices = select_devices(cfg, device)
    master = devices[0]
    if master != device:
        # keep master consistent
        A.data = A.data.to(master)
        B.data = B.data.to(master)
        S_index = S_index.to(master)
        S_value = S_value.to(master)

    A_devs, B_devs, S_index_devs, S_value_devs, streams = setup_replicas(A, B, S_index, S_value, devices)

    history = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        total_loss, sumSr, num_jacc = 0.0, 0.0, 0.0
        t_start = time.time()

        for blocks in loader:
            shards = shard_blocks(blocks, len(devices))
            zero_replica_grads(opt, A_devs, B_devs)

            # compute+backward on each device
            losses = []
            sumSrs = []
            numjs  = []
            for di, dev in enumerate(devices):
                loss_d, sumSr_d, numj_d = compute_backward_on_device(
                    di, dev, shards[di],
                    A_devs, B_devs, S_index_devs, S_value_devs,
                    streams,
                    cfg, m, n,
                    GD_latent,
                    block_loss_and_pred,
                )
                losses.append(loss_d)
                sumSrs.append(sumSr_d)
                numjs.append(numj_d)
        
            #sync_all(devices)
            # ### CHANGED: replace device-wide synchronize with per-stream wait
            wait_streams_before_reduce(devices, streams)

            # reduce grads to master and step on master
            reduce_grads_to_master(A, B, A_devs, B_devs, master, average=True)
            apply_precondition(A, B, cfg, master)
            apply_clip_and_step(opt, A, B, cfg)

            # broadcast updated params
            broadcast_params_from_master(A, B, A_devs, B_devs, devices)

            # logging accumulators (sum per-device scalars onto master)
            total_loss += float(sum(ld.detach().to(master) for ld in losses).item())
            sumSr      += float(sum(sd.detach().to(master) for sd in sumSrs).item())
            num_jacc   += float(sum(nd.detach().to(master) for nd in numjs).item())

        time_step = time.time() - t_start
        denom_jacc = float(torch.sum(S_value).item()) + sumSr - num_jacc
        jacc = 1.0 - num_jacc / denom_jacc
        S_norm = float(torch.norm(S_value).item())
        rmse = math.sqrt(total_loss) / (S_norm + 1e-16)
        log = f"[epoch {epoch}/{cfg.epochs}]: rmse={rmse:.6f}, jacc={jacc:.6f}, factor_step ={time_step:6.4f}"
        print(log)
        history.append(log)

    total = time.time() - t0
    print(f"\nTotal elapsed time: {total:.2f} sec")

    rmse, jacc, errZ = eval(A, B, S_index, S_value,
                            m, n, num_blocks=cfg.num_blocks,
                            full_block_loader=loader, device=A.device)
    print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")

    return A.detach(), B.detach(), history

### OLD
# def GD_loop(
#     S_index: torch.LongTensor,
#     S_value: torch.Tensor,
#     m: int,
#     n: int,
#     cfg: TrainConfig,
#     GD_latent: bool = False,
# ):
#     torch.manual_seed(cfg.seed)
#     device = torch.device(cfg.device)

#     S_index = S_index.to(device)
#     S_value = S_value.to(device)

#     # Parameters
#     scale = 0.5*math.sqrt(S_value.mean()/cfg.d)
#     A = torch.nn.Parameter(torch.rand(m, cfg.d, device=device) * scale)
#     B = torch.nn.Parameter(torch.rand(n, cfg.d, device=device) * scale)
#     opt = make_optimizer(cfg.optim, [A, B], lr=cfg.lr, weight_decay=0.0, SGD_momentum=cfg.SGD_mom)
#     #opt = torch.optim.Adam([A, B], lr=cfg.lr) #TODO: can change to SGD? AdamW?

#     ds = RowBlockDataset(S_index, S_value, m=m, num_blocks=cfg.num_blocks)
#     loader = DataLoader(
#         ds,
#         batch_size=cfg.batch_blocks,
#         shuffle=cfg.shuffle_blocks,
#         collate_fn=collate_blocks
#     )
#     history = []
#     t0 = time.time()
#     for epoch in range(1, cfg.epochs + 1):
#         total_loss, sumSr, num_jacc = 0.0, 0.0, 0.0

#         t_start = time.time()

#         for blocks in loader:
#             loss = torch.tensor(0.0, device=device)
#             for (block_id, edge_idx) in blocks:
#                 block_id = int(block_id)
#                 edge_idx = edge_idx.to(device).view(-1)

#                 mse_block, sumSr_block, jacc_num_block, errZ_block = block_loss_and_pred(
#                     A, B,
#                     block_id=block_id, num_blocks=cfg.num_blocks, m=m, n=n,
#                     S_index=S_index, S_value=S_value, edge_idx=edge_idx,
#                     errZ_obj = GD_latent,
#                 )
#                 loss_block = errZ_block if errZ_block is not None else mse_block 
#                 loss = loss + loss_block
#                 sumSr   += sumSr_block
#                 num_jacc += jacc_num_block

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             #right precondition the gradisnt
#             if cfg.precondition:
#                 # Build Gram matrices with a small jitter; keep everything on-device/dtype
#                 with torch.no_grad():
#                     I = torch.eye(cfg.d, device=device, dtype=A.dtype)
#                     # for nuemrical stability
#                     G_B = B.T @ B + cfg.prec_eps * I
#                     G_A = A.T @ A + cfg.prec_eps * I

#                     # Cholesky factors (SPD by construction with eps)
#                     L_B = torch.linalg.cholesky(G_B)  # G_B = L_B @ L_B.T
#                     L_A = torch.linalg.cholesky(G_A)
#                     # We want: A.grad ← A.grad @ G_B^{-1}
#                     # Use cholesky_solve: (A.grad @ G_B^{-1}) = (G_B^{-1} @ A.grad^T)^T
#                     #    i.e., L_B L_B.T X = A_grad.T => X = (L_B L_B.T)^{-1} A_grad.T => transposing
#                     if A.grad is not None:
#                         A.grad.copy_(torch.cholesky_solve(A.grad.T.contiguous(), L_B).T)
#                     # And: B.grad ← B.grad @ G_A^{-1}
#                     if B.grad is not None:
#                         B.grad.copy_(torch.cholesky_solve(B.grad.T.contiguous(), L_A).T)

#             # SGD hacks: gradient clipping
#             if cfg.optim.lower() == 'sgd':
#                 clip_grad_norm_([A,B], max_norm=1.0)
#             opt.step()
#             total_loss += float(loss.item())
        
#         time_step = time.time() - t_start
#         #metrics
#         denom_jacc = float(torch.sum(S_value).item()) + sumSr - num_jacc
#         jacc = 1.0 - num_jacc / denom_jacc
#         S_norm = float(torch.norm(S_value).item())
#         rmse = math.sqrt(total_loss) / (S_norm + 1e-16)   
#         log = f"[epoch {epoch}/{cfg.epochs}]: rmse={rmse:.6f}, jacc={jacc:.6f}, factor_step ={time_step:6.4f}"
#         print(log)
#         history.append(log)

#     # 7) final timing display
#     total = time.time() - t0
#     print(f"\nTotal elapsed time: {total:.2f} sec")
#     rmse, jacc, errZ = eval(A, B, S_index, S_value, 
#                             m, n, num_blocks=cfg.num_blocks, 
#                             full_block_loader=loader, device=A.device)
#     print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")

#     return A.detach(), B.detach(), history

def _init_factors_testcase(
    m: int, d: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Init (A,B) deterministically for test purpose
    """
    np.random.seed(0)
    A = torch.FloatTensor(np.random.random((m, d)))
    B = -A
    return A, B

def _init_factors_factor_specific(
    S_index: torch.LongTensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    d: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factor-specific init based on row/col marginals of S.
    Produces A (m,d), B (n,d), then refactor().
    """
    # column sums
    sum1 = torch.zeros(n, dtype=S_value.dtype, device=S_value.device)
    sum1.index_add_(0, S_index[1], S_value)
    sum1 = sum1.reshape(-1, 1)  # (n,1)

    # row sums
    sum2 = torch.zeros(m, dtype=S_value.dtype, device=S_value.device)
    sum2.index_add_(0, S_index[0], S_value)
    sum2 = sum2.reshape(-1, 1)  # (m,1)

    # global sum
    sumS = S_value.sum()
    scaleA = torch.sqrt(n * sum2 / sumS)   # (m,1)
    scaleB = torch.sqrt(m * sum1 / sumS)   # (n,1)
    A = scaleA * (1 + torch.rand((m, d), dtype=torch.float32, device=S_value.device) / d) / 2
    B = -scaleB * (1 + torch.rand((n, d), dtype=torch.float32, device=S_value.device) / d) / 2
    A, B = refactor(A, B)
    return A, B


def _init_factors_factor_agnostic(
    S_value: torch.Tensor,
    m: int,
    n: int,
    d: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factor-agnostic init with your improved scaling (keeps your exact math).
    Produces A (m,d), B (n,d), then refactor().
    """
    scale = 0.5 * math.sqrt(S_value.mean() / d)
    A = (torch.rand((m, d), device=S_value.device) * scale)
    B = (-torch.rand((n, d), device=S_value.device) * scale)
    A, B = refactor(A, B)
    return A, B


def als_init_factors(
    S_index: torch.LongTensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    d: int,
    opts: dict,
    test_flag: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single entry point for init. Chooses:
      - testcase init if test_flag=True
      - factor-specific init if opts['factor_init']=True
      - factor-agnostic init otherwise
    """
    if test_flag:
        return _init_factors_testcase(m, d)
    if opts["factor_init"]:
        return _init_factors_factor_specific(S_index, S_value, m, n, d)
    else:
        return _init_factors_factor_agnostic(S_value, m, n, d)

def als_post_process_factors(
    rmse: float,
    it: int,
    rmse_hist: list,
    A: torch.Tensor,
    B: torch.Tensor,
    dA: torch.Tensor,
    dB: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    1) if rmse > 1.0: shrink weights, refactor, reset momentum buffers
     2) if it > 1 and rmse > rmse_hist[-2]: reset momentum buffers
    """
    if rmse > 1.0:
        A = 0.5 * A
        B = 0.5 * B
        A, B = refactor(A, B)
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)
    if it > 1 and rmse > rmse_hist[-2]:
        dA = torch.zeros_like(A)
        dB = torch.zeros_like(B)

    return A, B, dA, dB

def als_early_stop(
    it: int,
    elapsed: float,
    jacc: float,
    jacc_hist: list,
    opts: dict,
) -> bool:
    """
    Early stopping criteria, identical to your existing logic.
    """
    if elapsed > opts["time_limit"]:
        return True
    if jacc < opts["tol_abs"]:
        return True
    if it >= 2 * opts["tol_window"]:
        w = opts["tol_window"]
        m1 = sum(jacc_hist[-w:]) / w
        m2 = sum(jacc_hist[-2 * w : -w]) / w
        if (m2 - m1) / m2 < opts["tol_rel"]:
            return True
    return False

def sumac_loop(S_index: torch.LongTensor,
               S_value: torch.Tensor,
               m: int,
               n: int,
               d: int,
               opts: dict,
               test_flag: bool = False):
    """
    PyTorch version of the main sumac loop, that solves nonlinear low-rank factorization
    S \approx max(0, A B^T) where A is of shape (m, r), B is of shape (n, r)
    - S: sparse matrix of shape (m, n)
    - d: the chosen rank
    - opts: additional arguments
    Return: A, B, costs (evaluation metrics)
    """
    # 1) start clock
    print(f"test_flag={test_flag}")
    dtype = opts['dtype']

    # 2) prepare cost histories
    max_iter = opts['max_iterate']
    rmse_hist = []
    jacc_hist = []
    time_hist = []

    # 3) initialize A, B and set up blocks
    random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    A, B = als_init_factors(S_index, S_value, m, n, d, opts, test_flag=test_flag)
    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)
    num_blocks = opts['num_blocks']
    print(f'use {num_blocks} blocks')

    # 4) main loop: every 2 iterations finish update both factors A,B
    t0 = time.time()
    for it in range(max_iter): #range(1, max_iter+1):
        t_start = time.time()
        # update A or B 
        if it % 2 == 1:
            ## TODO: wrap it into a function, input arg - update A or B
            nextB, rmse, jacc = least_squares_update_fast(S_index, S_value, A, B, 
                                                     num_blocks, opts['cols_per_block'])
            dB = (nextB - B) + dB * opts['exaggerate']  * (it > opts['momentum_start_iter']) #apply momentum after 10 iterations
            B  = B + dB
        else:
            nextA, rmse, jacc = least_squares_update_fast(S_index[[1,0],:], S_value, B, A, 
                                                  num_blocks, opts['cols_per_block'])
            dA = (nextA - A) + dA * opts['exaggerate'] * (it > opts['momentum_start_iter']) #apply momentum after 10 iterations
            A  = A + dA

        # record costs
        torch.cuda.synchronize() ##timing on gpu
        time_step = time.time() - t_start
        elapsed = time.time() - t0
        rmse_hist.append(rmse)
        jacc_hist.append(jacc)
        time_hist.append(elapsed)

        # display progress
        if opts['display']:
            print(f"iter = {it:04d}, rmse = {rmse:.6f},  jacc = {jacc:.6f},  factor_step = {time_step:6.4f}")
        
        # post-processing and early stopping
        A, B, dA, dB = als_post_process_factors(rmse, it, rmse_hist, A, B, dA, dB)
        if als_early_stop(it, elapsed, jacc, jacc_hist, opts):
            break

    # 5) undo the partial update on break, so that the cost (precomputed before update) matches the same model
    if it % 2 == 1:
        # last update was on B
        B = B - dB
    else:
        A = A - dA

    # 6) assemble costs
    costs = {
        'rmse': rmse_hist,
        'jacc': jacc_hist,
        'time': time_hist
    }

    # 7) final timing display
    if opts['display']:
        total = time.time() - t0
        print(f"\nTotal elapsed time: {total:.2f} sec")

    return A, B, costs

##testing
# m = 100
# r = 8
# S_dense = torch.eye(m) #torch.eye(100)
# S_index, S_value = dense_to_sparse(S_dense)
# sumac(S_index, S_value, m=m, n=m, d=r, test_flag=True, 
#       max_iterate=30, mom=0.7, use_GD=True, num_blocks=2)
# sumac(S_index, S_value, m=m, n=m, d=r, test_flag=True, 
#       max_iterate=30, mom=0.7, use_GD=False, num_blocks=1)
