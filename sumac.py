import torch
from torch.utils.data import DataLoader

import math
import time
import random
from data import prune_zero_rows_cols
import pickle

from _sumac.dataset import collate_blocks, StochasticRowBlockDataset
from _sumac.train_gd import TrainConfig, make_optimizer, select_devices, setup_replicas, shard_blocks, \
                            zero_replica_grads, compute_backward_on_device, wait_streams_before_reduce, \
                            reduce_grads_to_master, broadcast_params_from_master, apply_precondition, apply_clip_and_step
from _sumac.helper_als_salsa import als_init_factors, als_post_process_factors, als_early_stop
from _sumac.train_als import least_squares_update_fast, refactor
from _sumac.train_salsa import update_factor_salsa
from _sumac.eval import block_loss_and_pred, eval

def sumac(S_index, S_value, m, n, d, max_iterate=25, num_blocks=None,
          test_flag=False, dtype=torch.float32, opts=None, 
          mom=0.7, method='GD', lr=1e-1, factor_init=False,
          save_path=None, GD_latent=False, optim="adam", precondition=False,
          adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8, muon_momentum=0.95,
          multi_gpu=True, seed=0):
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
    Saved Artifacts:
      A, B    : factors (torch.Tensors)
      costs   : list or tensor of cost history
      opts    : dict of options used
    """

    # verify input nonnegativity
    if (S_value < 0).any().item():
        raise ValueError("sumac: the input matrix should be nonnegative.")

    # default options / user-provided opts
    opts_default = {
        'max_iterate': max_iterate,
        'dtype': dtype,
        'method': method,
        'seed': seed,
        'display': 1,
        'cache_MB': 5000,  
        'time_limit': float('inf'), #ALS early stopping
        'tol_abs': 1e-2, #ALS early stopping
        'tol_rel': 1e-4, #ALS early stopping
        'tol_window': 20, #ALS early stopping
        'exaggerate': mom, #momentum for SALSA or ALS
        'momentum_start_iter': 10, #ALS 
        'refactor_interval': 25, #ALS
        'eval_interval': 10, #evaluation per interval for SALSA or ALS
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
    torch.cuda.nvtx.range_push("core SUMAC loop")
    if method == 'GD':
        cfg = TrainConfig(d, num_blocks=num_blocks, epochs=max_iterate, lr=lr,
                          optim=optim, SGD_mom=mom, precondition=precondition,
                          adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps,
                          muon_momentum=muon_momentum)
        ## NEW: for multi-gpus, scale batch blocks and lr automatically
        if torch.cuda.device_count() > 1:
            cfg.batch_blocks = torch.cuda.device_count()
            cfg.lr = cfg.lr * torch.cuda.device_count() * 0.75 #TODO: test / better heuristic 
        A, B, costs = GD_loop(S_index, S_value, m_eff, n_eff, cfg, GD_latent)
    elif method == 'SALSA':
        A, B, costs = salsa_loop(S_index, S_value, m_eff, n_eff, d, opts, test_flag)
    elif method == 'ALS':
        A, B, costs = sumac_loop(S_index, S_value, m_eff, n_eff, d, opts, test_flag)
    else:
        raise NotImplementedError("method must be chosen from GD_ or SALSA or ALS")
    torch.cuda.nvtx.range_pop()
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
    device = torch.device(cfg.device) 

    # Move base data to master device first (replicas created later)
    S_index = S_index.to(device)
    S_value = S_value.to(device)

    # Parameters on master
    scale = 0.5 * math.sqrt(S_value.mean().item() / cfg.d)
    A = torch.nn.Parameter(torch.rand(m, cfg.d, device=device) * scale)
    B = torch.nn.Parameter(torch.rand(n, cfg.d, device=device) * scale)
    opt = make_optimizer(cfg.optim, [A, B], lr=cfg.lr, weight_decay=0.0, SGD_momentum=cfg.SGD_mom,
                         adam_betas=(cfg.adam_beta1, cfg.adam_beta2), adam_eps=cfg.adam_eps,
                         muon_momentum=cfg.muon_momentum)

    # DataLoader
    ds = StochasticRowBlockDataset(S_index, S_value, m, cfg.num_blocks) 
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
        torch.cuda.nvtx.range_push("epoch: " + str(epoch))
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
        torch.cuda.nvtx.range_pop()
    total = time.time() - t0
    print(f"\nTotal elapsed time: {total:.2f} sec")

    rmse, jacc, errZ = eval(A, B, S_index, S_value,
                            m, n, num_blocks=cfg.num_blocks,
                            full_block_loader=loader, device=A.device, errZ_obj=True)
    print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")

    return A.detach(), B.detach(), history


def sumac_loop(S_index: torch.LongTensor,
               S_value: torch.Tensor,
               m: int,
               n: int,
               d: int,
               opts: dict,
               test_flag: bool = False):
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
    A, B = als_init_factors(S_index, S_value, m, n, d, opts, test_flag=test_flag)
    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)
    num_blocks = opts['num_blocks']
    print(f'use {num_blocks} blocks')

    # 4) main loop: every 2 iterations finish update both factors A,B
    t0 = time.time()
    for it in range(max_iter): #range(1, max_iter+1):
        torch.cuda.nvtx.range_push("Iteration " + str(it) + " start")
        t_start = time.time()
        # update A or B 
        if it % 2 == 1:
            torch.cuda.nvtx.range_push("Least Square Update B")
            ## TODO: wrap it into a function, input arg - update A or B
            nextB, rmse, jacc = least_squares_update_fast(S_index, S_value, A, B, 
                                                     num_blocks, opts['cols_per_block'])
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("apply momentum")
            dB = (nextB - B) + dB * opts['exaggerate']  * (it > opts['momentum_start_iter']) #apply momentum after 10 iterations
            B  = B + dB
            torch.cuda.nvtx.range_pop()
        else:
            torch.cuda.nvtx.range_push("Least Square Update A")
            nextA, rmse, jacc = least_squares_update_fast(S_index[[1,0],:], S_value, B, A, 
                                                  num_blocks, opts['cols_per_block'])
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("apply momentum")
            dA = (nextA - A) + dA * opts['exaggerate'] * (it > opts['momentum_start_iter']) #apply momentum after 10 iterations
            A  = A + dA
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("record costs")
        # record costs
        torch.cuda.synchronize() ##timing on gpu
        time_step = time.time() - t_start
        elapsed = time.time() - t0
        rmse_hist.append(rmse)
        jacc_hist.append(jacc)
        time_hist.append(elapsed)
        torch.cuda.nvtx.range_pop()
        # display progress
        if opts['display']:
            print(f"iter = {it:04d}, rmse = {rmse:.6f},  jacc = {jacc:.6f},  factor_step = {time_step:6.4f}")

        # post-processing and early stopping
        torch.cuda.nvtx.range_push("post_process_and_stop")
        A, B, dA, dB = als_post_process_factors(rmse, it, rmse_hist, A, B, dA, dB)
        torch.cuda.nvtx.range_pop()

        if als_early_stop(it, elapsed, jacc, jacc_hist, opts):
            break

        torch.cuda.nvtx.range_pop()
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


def salsa_loop(S_index, S_value, m, n, d, opts, test_flag=False):
    """
    Minimal PyTorch version of the SALSA loop, reusing helpers from sumac.py.
    """
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu" #S_value.device
    S_index = S_index.to(device)
    S_value = S_value.to(device)
    
    # Initialization using sumac helper
    random.seed(opts['seed'])
    gen = torch.Generator(device=device)
    print(f"seed = {opts['seed']}")
    gen.manual_seed(opts['seed'])
    gen_rows = torch.Generator(device=device)
    gen_rows.manual_seed(opts['seed'] + 1)
    gen_cols = torch.Generator(device=device)
    gen_cols.manual_seed(opts['seed'] + 2)
    torch.cuda.nvtx.range_push("als_init_factors")
    A, B = als_init_factors(S_index, S_value, m, n, d, opts, test_flag=test_flag, gen=gen)
    torch.cuda.nvtx.range_pop()

    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)

    
    # Datasets for row and column blocks

    torch.cuda.nvtx.range_push("StochasitcRowBlockDataset rows")
    ds_rows = StochasticRowBlockDataset(S_index, S_value, m, opts['num_blocks'], gen=gen_rows)
    S_index_T = S_index[[1, 0], :] 
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("StochasticRowBlockDataset cols")
    ds_cols = StochasticRowBlockDataset(S_index_T, S_value, n, opts['num_blocks'], gen=gen_cols)
    torch.cuda.nvtx.range_pop()
    ##init evaluation

    torch.cuda.nvtx.range_push("eval_loader init")
    eval_loader = DataLoader(ds_rows, batch_size=1, shuffle=False, collate_fn=collate_blocks)
    rmse, jacc, errZ = eval(A.to(device), B.to(device), S_index, S_value, m, n, opts['num_blocks'], 
                            eval_loader, device=device, errZ_obj=True)
    print(f"iter = 0000, rmse = {rmse:.6f}, jacc = {jacc:.6f}, errZ = {errZ:.6}")
    torch.cuda.nvtx.range_pop()
    rmse_hist = []
    jacc_hist = []
    time_hist = []
    
    t_start_loop = time.time()    
    
    momentum = torch.tensor(opts.get("exaggerate", 0.7), device=A.device, dtype=A.dtype)
    t_start = time.time()
    for iter_idx in range(1, opts['max_iterate'] + 1):
        torch.cuda.nvtx.range_push("Iteration " + str(iter_idx))

        # Truly stochastic sampling: reshuffle partitions every epoch
        torch.cuda.nvtx.range_push("reshuffle")
        ds_rows.reshuffle()
        ds_cols.reshuffle()
        block_order = list(range(opts['num_blocks']))
        torch.cuda.nvtx.range_pop()
        #random.shuffle(block_order) -- only used for deterministic minibatch
        
        for mb_idx, block_id in enumerate(block_order):

            stepnum = mb_idx + 1 + (iter_idx - 1) * opts['num_blocks']
            unbias = 1 - (momentum ** stepnum)

            torch.cuda.nvtx.range_push("update_factor_salsa B")
            # --- Update B ---
            B, dB = update_factor_salsa(S_index, S_value, ds_rows, block_id, A, B, dB, momentum, unbias)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("update_factor_salsa A")
            # --- Update A ---
            A, dA = update_factor_salsa(S_index_T, S_value, ds_cols, block_id, B, A, dA, momentum, unbias)
            torch.cuda.nvtx.range_pop()

        # Metrics and Reporting
        if iter_idx % opts['eval_interval'] == 0:
            eval_loader = DataLoader(ds_rows, batch_size=1, shuffle=False, collate_fn=collate_blocks)
            rmse, jacc, errZ = eval(A.to(device), B.to(device), S_index, S_value, m, n, opts['num_blocks'], 
                                    eval_loader, device=device, errZ_obj=True)
            
            #if num_gpus > 0:
            torch.cuda.synchronize() ##timing on gpu
            elapsed = time.time() - t_start
            rmse_hist.append(rmse)
            jacc_hist.append(jacc)
            time_hist.append(elapsed)
            
            if opts['display']:
                print(f"iter = {iter_idx:04d}, rmse = {rmse:.6f}, jacc = {jacc:.6f}, errZ = {errZ:.6}, time = {elapsed:.2f}s")
            t_start = time.time()
        torch.cuda.nvtx.range_pop()
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
