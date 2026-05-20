import numpy as np
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
from concurrent.futures import ThreadPoolExecutor

from _sumac.dataset import block_span, RowBlockDataset, collate_blocks, StochasticRowBlockDataset, \
                            MultiGPUStochasticRowBlockDataset, get_sharded_ds
from _sumac.train_gd import TrainConfig, make_optimizer, select_devices, setup_replicas, shard_blocks, \
                            zero_replica_grads, compute_backward_on_device, sync_all, wait_streams_before_reduce, \
                            reduce_grads_to_master, broadcast_params_from_master, apply_precondition, apply_clip_and_step
from _sumac.helper_als_salsa import als_init_factors, als_post_process_factors, als_early_stop
from _sumac.train_als import least_squares_update_fast, least_squares_update, refactor
from _sumac.train_salsa import update_factor_salsa
from _sumac.eval import block_loss_and_pred, eval

def sumac(S_index, S_value, m, n, d, max_iterate=25, num_blocks=None,
          test_flag=False, dtype=torch.float32, opts=None, 
          mom=0.7, method='GD', lr=1e-1, factor_init=False,
          save_path=None, GD_latent=False, optim="adam", precondition=False,
          adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, muon_momentum=0.95,
          multi_gpu=True, A_init=None, B_init=None):
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
      method: optimization routines, choices = ["GD", "SALSA", "ALS"]
      factor_init: default False, use per factor (A, B) specific scaling at initialiation for method="ALS"
      GD_latent: default False. If true, correct the GD MSE object to align with SALSA/ALS
      precondition: default False. If true, precondition the gradient of factors by A.grad = A.grad @ (B^T B)^{-1}
      optimizer args: adam_, muon_
      multi_gpu: If true, launch multiple streams, each per device
      A_init, B_init: allow user-specified factor initialization
    Saved Artifacts:
      A, B    : factors (torch.Tensors)
      costs   : list or tensor of cost history
      opts    : dict of options used
    """

    # verify input nonnegativity
    if (S_value < 0).any().item():
        raise ValueError("sumac: the input matrix should be nonnegative.")

    # default options / user-provided opts
    default_eval_interval = 100 if method == 'GD' else 10
    opts_default = {
        'max_iterate': max_iterate,
        'dtype': dtype,
        'method': method,
        'seed': random.randint(0, 2**31 - 1),
        'display': 1,
        'cache_MB': 5000,
        'time_limit': float('inf'), #ALS early stopping
        'tol_abs': 1e-2, #ALS early stopping
        'tol_rel': 1e-4, #ALS early stopping
        'tol_window': 20, #ALS early stopping
        'exaggerate': mom, #momentum for SALSA or ALS
        'momentum_start_iter': 10, #ALS
        'refactor_interval': 25, #ALS
        'eval_interval': default_eval_interval, #eval cadence: 100 for GD, 10 for SALSA/ALS
        'factor_init': factor_init, #ALS; if True: A,B specific initialization
        'optim': optim, #GD optimizer; default adam, also support sgd, adamw, muon
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
    if method == 'GD':
        cfg = TrainConfig(d, num_blocks=num_blocks, epochs=max_iterate, lr=lr,
                          optim=optim, SGD_mom=mom, precondition=precondition,
                          adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps,
                          muon_momentum=muon_momentum,
                          eval_interval=opts['eval_interval'])
        ## NEW: for multi-gpus, scale batch blocks and lr automatically
        if torch.cuda.device_count() > 1:
            cfg.batch_blocks = torch.cuda.device_count() #max(torch.cuda.device_count() * 4, 8) #
            cfg.lr = cfg.lr * torch.cuda.device_count() * 0.75 #TODO: test / better heuristic 
        A, B, costs = GD_loop(S_index, S_value, m_eff, n_eff, cfg, GD_latent, A_init, B_init)
    elif method == 'SALSA':
        A, B, costs = salsa_loop(S_index, S_value, m_eff, n_eff, d, opts, test_flag, A_init, B_init)
    elif method == 'ALS':
        A, B, costs = sumac_loop(S_index, S_value, m_eff, n_eff, d, opts, test_flag, A_init, B_init)
    else:
        raise NotImplementedError("method must be chosen from GD_ or SALSA or ALS")

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
    return A_ori, B_ori, costs

def GD_loop(
    S_index: torch.LongTensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    cfg: TrainConfig,
    GD_latent: bool = False,
    A_init: torch.Tensor = None,
    B_init: torch.Tensor = None
):
    torch.manual_seed(cfg.seed)
    base_device = torch.device(cfg.device)

    # Multi-GPU setup (manual DP, no DDP)
    devices = select_devices(cfg, base_device)
    master = devices[0]

    # Keep S on CPU for sharding (avoid parking full S on master GPU)
    S_index_cpu = S_index.detach().cpu()
    S_value_cpu = S_value.detach().cpu()

    # Parameters on master
    torch.cuda.nvtx.range_push("Declare params and optimizer")
    if A_init is None or B_init is None:
        scale = 0.5 * math.sqrt(S_value_cpu.mean().item() / cfg.d)
        A = torch.nn.Parameter(torch.rand(m, cfg.d, device=master) * scale)
        B = torch.nn.Parameter(torch.rand(n, cfg.d, device=master) * scale)
    else:
        A = torch.nn.Parameter(A_init.to(master))
        B = torch.nn.Parameter(B_init.to(master))
    opt = make_optimizer(
        cfg.optim, [A, B], lr=cfg.lr, weight_decay=0.0, SGD_momentum=cfg.SGD_mom,
        adam_betas=(cfg.adam_beta1, cfg.adam_beta2), adam_eps=cfg.adam_eps,
        muon_momentum=cfg.muon_momentum
    )
    torch.cuda.nvtx.range_pop()

    # Dataset (row-sharded; each shard pinned to its device)
    torch.cuda.nvtx.range_push("MultiGPUStochasticRowBlockDataset init")
    ds = MultiGPUStochasticRowBlockDataset(S_index_cpu, S_value_cpu, m, cfg.num_blocks, devices)
    torch.cuda.nvtx.range_pop()

    # Replicate params
    torch.cuda.nvtx.range_push("Multi-GPU setup replicas")
    A_devs = []
    B_devs = []
    streams = []
    for dev in devices:
        streams.append(torch.cuda.Stream(device=dev))
        if dev == master:
            A_devs.append(A)
            B_devs.append(B)
        else:
            A_devs.append(A.detach().to(dev).requires_grad_(True))
            B_devs.append(B.detach().to(dev).requires_grad_(True))
    # Per-device S tensors from dataset shards
    S_index_devs = [sh["S_index"] for sh in ds.shards]
    S_value_devs = [sh["S_value"] for sh in ds.shards]
    torch.cuda.nvtx.range_pop()


    # Hoisted eval setup (reused for both periodic and end-of-training eval)
    ds_rows_eval = StochasticRowBlockDataset(S_index_cpu, S_value_cpu, m, cfg.num_blocks)
    eval_loader = DataLoader(ds_rows_eval, batch_size=1, shuffle=False, collate_fn=collate_blocks)
    S_index_master = S_index_cpu.to(master, non_blocking=True)
    S_value_master = S_value_cpu.to(master, non_blocking=True)

    history = []
    t0 = time.time()
    eval_time_total = 0.0  # exclude eval overhead from abs_time

    all_block_ids = list(range(cfg.num_blocks))
    batch_blocks = int(cfg.batch_blocks)
    # Use ThreadPoolExecutor for concurrent CPU dispatch [speedup]
    executor = ThreadPoolExecutor(max_workers=len(devices))

    for epoch in range(1, cfg.epochs + 1):
        torch.cuda.nvtx.range_push(f"epoch: {epoch}")
        # reshuffle within each local shard 
        ds.reshuffle()
        # Sync worker streams with default streams to ensure reshuffle is complete
        for di, dev in enumerate(devices):
            if streams[di] is not None:
                streams[di].wait_stream(torch.cuda.default_stream(dev))

        # Use per-device tensors for metric accumulation to avoid cross-device sync issues
        total_loss_devs = [torch.tensor(0.0, device=d) for d in devices]
        sumSr_devs = [torch.tensor(0.0, device=d) for d in devices]
        num_jacc_devs = [torch.tensor(0.0, device=d) for d in devices]

        t_start = time.time()

        # manual block order (CPU) 
        if cfg.shuffle_blocks:
            order = torch.randperm(cfg.num_blocks).tolist()
            block_order = [all_block_ids[i] for i in order]
        else:
            block_order = all_block_ids

        for k in range(0, cfg.num_blocks, batch_blocks):
            torch.cuda.nvtx.range_push("shared blocks and zero grads")
            batch_block_ids = block_order[k:k + batch_blocks]
            # Build shards_per_device in the exact structure compute_backward_on_device expects:
            # list over devices, each is list of block tuples.
            shards_per_device = get_sharded_ds(ds, devices, batch_block_ids)
            zero_replica_grads(opt, A_devs, B_devs)
            torch.cuda.nvtx.range_pop()

            # compute+backward on each device (parallel CPU thread [speedup])
            torch.cuda.nvtx.range_push("compute_loss_and_grad")
            def device_job(di, dev):
                # Critical: Set the active device for the current thread
                torch.cuda.set_device(dev)
                # Each thread manages its own CUDA stream context
                with torch.cuda.stream(streams[di]):
                    loss_d, sumSr_d, numj_d = compute_backward_on_device(
                        di, dev, shards_per_device[di],
                        A_devs, B_devs, S_index_devs, S_value_devs,
                        streams,
                        cfg, m, n,
                        GD_latent,
                        block_loss_and_pred,
                    )
                    total_loss_devs[di].add_(loss_d.detach())
                    sumSr_devs[di].add_(sumSr_d.detach())
                    num_jacc_devs[di].add_(numj_d.detach())
            
            # Launch jobs concurrently; use a sequential warm-up for the first device
            # on the first batch to avoid torch.compile threading race conditions.
            if epoch == 1 and k == 0:
                # Run the first device job sequentially to trigger compilation/autotuning
                device_job(0, devices[0])
                # Then run the remaining devices in parallel
                futures = [executor.submit(device_job, di, dev) for di, dev in enumerate(devices[1:], 1)]
                for f in futures:
                    f.result()
            else:
                # Normal parallel dispatch for all other batches
                futures = [executor.submit(device_job, di, dev) for di, dev in enumerate(devices)]
                for f in futures:
                    f.result()
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("wait streams and reduce to master")
            wait_streams_before_reduce(devices, streams)
            reduce_grads_to_master(A, B, A_devs, B_devs, master, average=True)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("apply post processing")
            apply_precondition(A, B, cfg, master)
            apply_clip_and_step(opt, A, B, cfg)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("broadcast from master to device")
            broadcast_params_from_master(A, B, A_devs, B_devs, devices)
            #  Ensure worker streams don't race ahead of broadcast copies (otherwise CUDA error with 3,4 devices)
            for di, dev in enumerate(devices):
                streams[di].wait_stream(torch.cuda.default_stream(dev))
            torch.cuda.nvtx.range_pop()
            
        time_step = time.time() - t_start
        # Sum across devices only at the end of the epoch
        total_loss = sum(t.item() for t in total_loss_devs)
        sumSr = sum(t.item() for t in sumSr_devs)
        num_jacc = sum(t.item() for t in num_jacc_devs)

        denom_jacc = float(torch.sum(S_value_cpu).item()) + sumSr - num_jacc
        jacc = 1.0 - num_jacc / denom_jacc
        S_norm = float(torch.norm(S_value_cpu).item())
        rmse = math.sqrt(total_loss) / (S_norm + 1e-16)
        abs_time = time.time() - t0 - eval_time_total
        log = (f"[epoch {epoch}/{cfg.epochs}]: rmse={rmse:.6f}, jacc={jacc:.6f}, "
               f"factor_step={time_step:6.4f}, abs_time={abs_time:.2f}s")
        print(log)
        history.append(log)

        # Periodic clean eval
        if epoch % cfg.eval_interval == 0:
            t_eval = time.time()
            rmse_eval, jacc_eval, errZ_eval = eval(
                A, B, S_index_master, S_value_master,
                m, n, num_blocks=cfg.num_blocks,
                full_block_loader=eval_loader, device=A.device, errZ_obj=True
            )
            eval_time_total += time.time() - t_eval
            abs_time = time.time() - t0 - eval_time_total
            eval_log = (f"EVAL [epoch {epoch}]: rmse={rmse_eval:.6f}, "
                        f"jacc={jacc_eval:.6f}, errZ={errZ_eval:.6f}, "
                        f"abs_time={abs_time:.2f}s")
            print(eval_log)
            history.append(eval_log)

        torch.cuda.nvtx.range_pop()

    executor.shutdown()
    total = time.time() - t0
    print(f"\nTotal elapsed time: {total:.2f} sec")

    # Final eval (reuses hoisted eval_loader / S_*_master)
    rmse, jacc, errZ = eval(
        A, B, S_index_master, S_value_master,
        m, n, num_blocks=cfg.num_blocks,
        full_block_loader=eval_loader, device=A.device, errZ_obj=True
    )
    print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")

    return A.detach(), B.detach(), history

def sumac_loop(S_index: torch.LongTensor,
               S_value: torch.Tensor,
               m: int,
               n: int,
               d: int,
               opts: dict,
               test_flag: bool = False,
               A_init: torch.Tensor = None,
               B_init: torch.Tensor = None):
    """
    ALS subroutine to solve the nonlinear low-rank factorization
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
    if A_init is None or B_init is None:
        A, B = als_init_factors(S_index, S_value, m, n, d, opts, test_flag=test_flag)
    else:
        A, B = A_init, B_init
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
            print(f"iter = {it:04d}, rmse = {rmse:.6f},  jacc = {jacc:.6f},  factor_step = {time_step:6.4f}, abs_time = {elapsed:.2f}s")
        
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


def salsa_loop(S_index, S_value, m, n, d, opts, test_flag=False,
               A_init: torch.Tensor = None,
               B_init: torch.Tensor = None):
    """
    Minimal PyTorch version of the SALSA loop, reusing helpers from sumac.py.
    """
    dtype = opts['dtype']
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu" #S_value.device
    S_index = S_index.to(device)
    S_value = S_value.to(device)
    
    # Initialization using sumac helper
    random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    if  A_init is None or B_init is None:
        A, B = als_init_factors(S_index, S_value, m, n, d, opts, test_flag=test_flag)
    else:
        A, B = A_init, B_init
    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)

    # Datasets for row and column blocks
    #ds_rows = RowBlockDataset(S_index, S_value, m, opts['num_blocks'])
    ds_rows = StochasticRowBlockDataset(S_index, S_value, m, opts['num_blocks'])
    S_index_T = S_index[[1, 0], :]
    #ds_cols = RowBlockDataset(S_index_T, S_value, n, opts['num_blocks'])
    ds_cols = StochasticRowBlockDataset(S_index_T, S_value, n, opts['num_blocks'])

    ##init evaluation
    eval_loader = DataLoader(ds_rows, batch_size=1, shuffle=False, collate_fn=collate_blocks)
    rmse, jacc, errZ = eval(A.to(device), B.to(device), S_index, S_value, m, n, opts['num_blocks'], 
                            eval_loader, device=device, errZ_obj=True)
    print(f"iter = 0000, rmse = {rmse:.6f}, jacc = {jacc:.6f}, errZ = {errZ:.6}")

    rmse_hist = []
    jacc_hist = []
    time_hist = []
    
    t_start_loop = time.time()
    # --- PERSISTENT CUDA INFRASTRUCTURE (eliminates launch overhead) ---
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    multi_gpu_internal = (num_gpus > 1) 
    
    if num_gpus > 0:
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        streams = [torch.cuda.Stream(device=d) for d in devices]
        row_map_buffers = [torch.zeros(m, dtype=torch.long, device=d) for d in devices]
        col_map_buffers = [torch.zeros(n, dtype=torch.long, device=d) for d in devices]
    else:
        streams, row_map_buffers, col_map_buffers = None, None, None

    for iter_idx in range(1, opts['max_iterate'] + 1):
        t_start = time.time()
        # Truly stochastic sampling: reshuffle partitions every epoch
        ds_rows.reshuffle()
        ds_cols.reshuffle()
        block_order = list(range(opts['num_blocks']))
        #random.shuffle(block_order) -- only used for deterministic minibatch
        
        for mb_idx, block_id in enumerate(block_order):
            stepnum = mb_idx + 1 + (iter_idx - 1) * opts['num_blocks']
            # --- Update B ---
            B, dB = update_factor_salsa(S_index, S_value, ds_rows, block_id, A, B, dB, opts, stepnum,
                                        multi_gpu=multi_gpu_internal, streams=streams, map_buffers=row_map_buffers)
            # --- Update A ---
            A, dA = update_factor_salsa(S_index_T, S_value, ds_cols, block_id, B, A, dA, opts, stepnum,
                                        multi_gpu=multi_gpu_internal, streams=streams, map_buffers=col_map_buffers)

        # Metrics and Reporting
        if iter_idx % opts['eval_interval'] == 0:
            eval_loader = DataLoader(ds_rows, batch_size=1, shuffle=False, collate_fn=collate_blocks)
            rmse, jacc, errZ = eval(A.to(device), B.to(device), S_index, S_value, m, n, opts['num_blocks'], 
                                    eval_loader, device=device, errZ_obj=True)
            
            if num_gpus > 0:
                torch.cuda.synchronize() ##timing on gpu
            elapsed = time.time() - t_start
            rmse_hist.append(rmse)
            jacc_hist.append(jacc)
            time_hist.append(elapsed)
            
            if opts['display']:
                print(f"iter = {iter_idx:04d}, rmse = {rmse:.6f}, jacc = {jacc:.6f}, errZ = {errZ:.6}, time = {elapsed:.2f}s")

    # WRAP UP
    A, B = refactor(A, B)
    
    costs = {
        'rmse': rmse_hist,
        'jacc': jacc_hist,
        'time': time_hist
    }

    if opts['display']:
        total = time.time() - t_start_loop
        print(f"\nTotal elapsed time: {total:.2f} sec")

    return A, B, costs


#testing
# m = 100
# r = 8
# S_dense = torch.eye(m) #torch.eye(100)
# S_index, S_value = dense_to_sparse(S_dense)
# print(f"running SALSA")
# sumac(S_index, S_value, m=m, n=m, d=r, test_flag=True, 
#       max_iterate=30, mom=0.7, method='SALSA', num_blocks=2)
# print(f"running GD...")
# sumac(S_index, S_value, m=m, n=m, d=r, test_flag=True, 
#       max_iterate=30, mom=0.7, method='GD', num_blocks=2)
# print(f"running ALS...")
# sumac(S_index, S_value, m=m, n=m, d=r, test_flag=True, 
#       max_iterate=30, mom=0.7, method='ALS', num_blocks=1)
# sumac(S_index, S_value, m=m, n=m, d=r, test_flag=True, 
#       max_iterate=30, mom=0.7, use_GD=False, num_blocks=1)
