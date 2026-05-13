import torch
from torch.utils.data import DataLoader, Subset
import math
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from _sumac.dataset import block_span, RowBlockDataset
from _sumac.model import FactorModel
from _sumac.train_gd import TrainConfig
from _sumac.eval import block_loss_and_pred, eval

# -----------------------------
# TODO: DEBUGGGGG - DDP training loop (FP32)
# -----------------------------
def GD_loop_ddp(
    rank: int,
    world_size: int,
    S_index: torch.Tensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    cfg: TrainConfig,
    GD_latent: bool = False,
):
       # --- DDP init ---
    print("WARNING: DEBUGGING IN PROCESS -- per-epoch metrics differ from eval!")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.manual_seed(cfg.seed)
    device = torch.device(f"cuda:{rank}")

    # --- Keep COO tensors on device (num_workers=0 -> OK) ---
    S_index = S_index.to(device, non_blocking=True)
    S_value = S_value.to(device, non_blocking=True)

    # --- Model / opt (identical init on all ranks) ---
    scale = 0.5 * math.sqrt(S_value.mean().item() / cfg.d)
    model = FactorModel(m, n, cfg.d, scale, device)
    ddp   = DDP(model, device_ids=[rank], output_device=rank)

    # Optional but robust: force-sync initial params from rank 0
    for p in ddp.module.parameters():
        dist.broadcast(p.data, src=0)

    opt = torch.optim.Adam(ddp.parameters(), lr=cfg.lr)

    # --- Dataset (row blocks) ---
    ds = RowBlockDataset(S_index, S_value, m=m, num_blocks=cfg.num_blocks)

    # Manual strided shard of block indices
    all_idx = torch.arange(cfg.num_blocks, device=S_index.device)
    ## TODO: removed?
    # shard = all_idx[rank::world_size].tolist()
    # target_len = (cfg.num_blocks + world_size - 1) // world_size
    # pad = target_len - len(shard)
    # if pad > 0:
    #     shard += shard[:pad]
    # rank_idx = torch.tensor(shard, device=all_idx.device)
    rank_idx = all_idx[rank::world_size]
    subset = Subset(ds, rank_idx.tolist())

    def collate_blocks(batch):  # list[(block_id, edge_idx), ...]
        return batch

    loader = DataLoader(
        subset,
        batch_size=cfg.batch_blocks,      # 1 block per step by default
        shuffle=cfg.shuffle_blocks,       # local shuffle ok; ranks are disjoint already
        num_workers=0,                    # keep workers=0 since dataset lives on CUDA
        pin_memory=cfg.pin_memory,
        collate_fn=collate_blocks,        # each batch consists of many blocks
    )

    history = []
    #cfg.epochs *= world_size #OPTIONAL:so that the effective total gradient steps remain invariant

    # --- Metrics constants (||S||_F and sum(S)) computed once and broadcast ---
    if rank == 0:
        S_norm = torch.norm(S_value).detach()
        S_sum  = torch.sum(S_value).detach()
    else:
        S_norm = torch.tensor(0.0, device=device)
        S_sum  = torch.tensor(0.0, device=device)
    dist.broadcast(S_norm, src=0)
    dist.broadcast(S_sum,  src=0)
    t0 = time.time()
    # --- Training loop ---
    for epoch in range(1, cfg.epochs + 1):
        ddp.train()

        # Accumulators for epoch metrics (data term only; reg excluded from RMSE)
        mse_epoch   = torch.tensor(0.0, device=device)
        sumSr_epoch = torch.tensor(0.0, device=device)
        jaccN_epoch = torch.tensor(0.0, device=device)
        t_start = time.time()

        for blocks in loader:
            mse_step = torch.tensor(0.0, device=device)
            reg = torch.tensor(0.0, device=device)
            touched_rows, touched_cols = [], []

            for (block_id, edge_idx) in blocks:
                # NOTE: 'block_id' returned by Subset is the original block id, good.
                block_id = int(block_id)
                edge_idx = edge_idx.to(device, non_blocking=True).view(-1)

                A, B = ddp.module.A, ddp.module.B
                mse_block, sumSr_block, jacc_num_block, _ = block_loss_and_pred(
                    A, B,
                    block_id=block_id, num_blocks=cfg.num_blocks, m=m, n=n,
                    S_index=S_index, S_value=S_value, edge_idx=edge_idx,
                    errZ_obj=GD_latent,
                )
                mse_step      += mse_block
                sumSr_epoch   += sumSr_block
                jaccN_epoch   += jacc_num_block

                # TODO: THIS DOES NOT EXIST
                if cfg.l2_reg > 0:
                    start, end = block_span(block_id, m, cfg.num_blocks)
                    touched_rows.append(torch.arange(start, end, device=device))
                    if edge_idx.numel() > 0:
                        touched_cols.append(torch.unique(S_index[1][edge_idx]))

            if touched_rows:
                tr = torch.cat(touched_rows)
                reg = reg + ddp.module.A[tr].pow(2).mean()
                if touched_cols:
                    tc = torch.unique(torch.cat(touched_cols))
                    reg = reg + ddp.module.B[tc].pow(2).mean()

            # Metrics use unscaled MSE (data-only)
            mse_epoch += mse_step.detach()

            # Training loss = (data + λ·reg); scale by world_size to counter DDP grad averaging
            step_loss = mse_step + (cfg.l2_reg * reg if cfg.l2_reg > 0 else 0.0)
            step_loss = step_loss * world_size

            opt.zero_grad(set_to_none=True)
            step_loss.backward()
            opt.step()

        # All-reduce metrics across ranks
        dist.all_reduce(mse_epoch,   op=dist.ReduceOp.SUM)
        dist.all_reduce(sumSr_epoch, op=dist.ReduceOp.SUM)
        dist.all_reduce(jaccN_epoch, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize() ##timing on gpu
        dist.barrier()
        time_step = time.time() - t_start

        if rank == 0:
            rmse = math.sqrt(max(float(mse_epoch.item()), 0.0)) / (float(S_norm.item()) + 1e-16)
            denom = float(S_sum.item()) + float(sumSr_epoch.item()) - float(jaccN_epoch.item())
            jacc = 1.0 - (float(jaccN_epoch.item()) / (denom + 1e-16)) if denom > 0 else 1.0
            log = f"[epoch {epoch}/{cfg.epochs}] rmse={rmse:.6f}, jacc={jacc:.6f}, factor_step ={time_step:6.4f}"
            print(log)
            history.append(log)

            # DEBUG: 1) Recompute the SAME quantities via a standalone pass that mirrors training math
            with torch.no_grad():
                A0, B0 = ddp.module.A, ddp.module.B
                sse_full_eval = torch.tensor(0.0, device=device)
                sumSr_eval    = torch.tensor(0.0, device=device)
                jaccN_eval    = torch.tensor(0.0, device=device)

                # iterate all blocks; build edge_idx by masking S_index rows into the block span
                for b_id in range(cfg.num_blocks):
                    start, end = block_span(b_id, m, cfg.num_blocks)
                    # mask for edges in this row-block (S_index lives on device already)
                    mask = (S_index[0] >= start) & (S_index[0] < end)
                    edge_idx = torch.nonzero(mask, as_tuple=False).view(-1)

                    mse_full_b, sumSr_b, jaccN_b, _ = block_loss_and_pred(
                        A0, B0,
                        block_id=b_id, num_blocks=cfg.num_blocks, m=m, n=n,
                        S_index=S_index, S_value=S_value, edge_idx=edge_idx
                    )
                    sse_full_eval += mse_full_b
                    sumSr_eval    += sumSr_b
                    jaccN_eval    += jaccN_b

                # 2) Compute RMSE/Jacc from BOTH paths and print side-by-side
                S_norm_val = float(S_norm.item())
                rmse_loop  = math.sqrt(max(float(mse_epoch.item()), 0.0)) / (S_norm_val + 1e-16)
                rmse_eval  = math.sqrt(max(float(sse_full_eval.item()), 0.0)) / (S_norm_val + 1e-16)

                denom_loop = float(S_sum.item()) + float(sumSr_epoch.item()) - float(jaccN_epoch.item())
                denom_eval = float(S_sum.item()) + float(sumSr_eval.item())  - float(jaccN_eval.item())
                jacc_loop  = 1.0 - (float(jaccN_epoch.item()) / (denom_loop + 1e-16)) if denom_loop > 0 else 1.0
                jacc_eval  = 1.0 - (float(jaccN_eval.item())  / (denom_eval + 1e-16)) if denom_eval > 0 else 1.0

                print(
                    "[DIAG] sse_loop={:.6e}, sse_eval={:.6e}, S_norm^2={:.6e}".format(
                        float(mse_epoch.item()),
                        float(sse_full_eval.item()),
                        S_norm_val**2
                    )
                )
                print(
                    "[DIAG] rmse_loop={:.6f}, rmse_eval={:.6f} | jacc_loop={:.6f}, jacc_eval={:.6f}".format(
                        rmse_loop, rmse_eval, jacc_loop, jacc_eval
                    )
                )

            # if epoch % 5 == 0:
            #     rmse, jacc = eval(S_index.to("cpu"), S_value.to("cpu"), ddp.module.A.detach().cpu(), ddp.module.B.detach().cpu(), 
            #                       [torch.device("cuda:0")], num_blocks=16)
            #     log = f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}"
            #     print(log)
            #     history.append(log)

    # 7) final timing display
    total = time.time() - t0
    print(f"\nTotal elapsed time: {total:.2f} sec")

    # >>> NEW: ensure all ranks finished their last optimizer step
    torch.cuda.synchronize()
    dist.barrier()

    # Return final params on rank 0
    A0 = ddp.module.A.detach().cpu() if rank == 0 else None
    B0 = ddp.module.B.detach().cpu() if rank == 0 else None

    # EVAL
    if rank == 0:
        devices = [torch.device("cuda:0")]
        # TODO CHECK THIS FIX
        m = n = int(S_index[0].max())
        rmse, jacc, _ = eval(
            ddp.module.A.detach().cpu(),
            ddp.module.B.detach().cpu(),
            S_index.to("cpu"),
            S_value.to("cpu"),
            m,                  # TODO CHECK ME
            n,                  # TODO CHECK ME
            cfg.num_blocks,     # TODO CHECK ME
            loader,             # TODO CHECK ME
            devices[0]          # TODO CHECK ME
        )
        print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}")

    dist.destroy_process_group()
    return A0, B0, history
