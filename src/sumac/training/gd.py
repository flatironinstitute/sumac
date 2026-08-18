import math
import time
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch import Tensor, Generator, device

from sumac.config import OptimizerName, SumacConfig
from sumac.datasets import collate_blocks, StochasticRowBlockDataset
from sumac.eval import block_loss_and_pred, eval
from sumac.kernels.cuda_utils import nvtx_range_pop, nvtx_range_push
from sumac.kernels.tuning import AutotuneReluBatReduce
from sumac.training.salsa import init_salsa_factors


# TODO:
# - further review
# - remove performance reporting instrumentation?

def make_optimizer(
    cfg: SumacConfig,
    params: list[torch.nn.Parameter],
    weight_decay: float = 0.0,
):
    lr = cfg.learning_rate
    if cfg.optimizer == OptimizerName.ADAM:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=cfg.adam_betas, eps=cfg.adam_eps)
    if cfg.optimizer == OptimizerName.ADAMW:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=cfg.adam_betas, eps=cfg.adam_eps)
    if cfg.optimizer == OptimizerName.SGD:
        return torch.optim.SGD(params, lr=lr, momentum=cfg.momentum, weight_decay=weight_decay)
    if cfg.optimizer == OptimizerName.MUON:
        return torch.optim.Muon(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=cfg.momentum,
            adjust_lr_fn="match_rms_adamw",
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def apply_clip_and_step(opt, A, B, cfg):
    if cfg.optimizer == OptimizerName.SGD:
        clip_grad_norm_([A, B], max_norm=1.0)
    opt.step()


def _init_factors(
    S_index: Tensor,
    S_value: Tensor,
    cfg: SumacConfig,
    m: int,
    n: int,
    gen: Generator,
    A_init: Tensor | None,
    B_init: Tensor | None,
):
    if A_init is None or B_init is None:
        A, B = init_salsa_factors(
            S_index,
            S_value,
            m,
            n,
            cfg.rank,
            gen,
        )
        A = torch.nn.Parameter(A.detach())
        B = torch.nn.Parameter(B.detach())
    else:
        A = torch.nn.Parameter(A_init.detach().to(cfg.device).clone())
        B = torch.nn.Parameter(B_init.detach().to(cfg.device).clone())
    return (A, B)


def _init_block_loss(
    eval_kernel: AutotuneReluBatReduce,
    S_index: Tensor,
    S_value: Tensor,
    device: device
):
    def inner(
        A: Tensor,
        B: Tensor,
        edge_idx: Tensor,
        row_indices: Tensor,
    ):
        edge_idx = edge_idx.to(device).view(-1)
        mse_block, sumSr_block, jacc_num_block, errZ_block = block_loss_and_pred(
            eval_kernel,
            A,
            B,
            S_index=S_index,
            S_value=S_value,
            edge_idx=edge_idx,
            row_indices=row_indices,
        )
        loss_block = errZ_block if errZ_block is not None else mse_block
        return (
            loss_block,
            mse_block,
            torch.as_tensor(sumSr_block, device=device, dtype=A.dtype),
            jacc_num_block,
        )

    return inner


# TODO: Can we slim this down more?
def GD_loop(
    S_index: Tensor,
    S_value: Tensor,
    m: int,
    n: int,
    cfg: SumacConfig,
    A_init: Tensor | None = None,
    B_init: Tensor | None = None,
):
    assert cfg.num_blocks is not None
    assert cfg.device is not None
    assert cfg.eval_interval is not None

    (gen, gen_blocks, gen_loader) = cfg.get_generator()
    eval_kernel = AutotuneReluBatReduce()
    _block_eval = _init_block_loss(eval_kernel, S_index, S_value, cfg.device)
    
    # Move data to the training device.
    S_index = S_index.to(cfg.device)
    S_value = S_value.to(cfg.device)
    (A, B) = _init_factors(S_index, S_value, cfg, m, n, gen, A_init, B_init)
    opt = make_optimizer(cfg, [A, B], weight_decay=0.0)

    # DataLoader
    ds = StochasticRowBlockDataset(S_index, S_value, m, cfg.num_blocks, gen_blocks)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_blocks,
        shuffle=cfg.shuffle_blocks,
        collate_fn=collate_blocks,
        generator=gen_loader,
    )
    eval_generator = torch.Generator()
    eval_generator.manual_seed(gen_loader.initial_seed())
    eval_loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_blocks,
        generator=eval_generator,
    )

    rmse, jacc, errZ = eval(
        eval_kernel,
        A,
        B,
        S_index,
        S_value,
        full_block_loader=eval_loader,
        device=A.device,
    )
    if cfg.verbose:
        print(
            f"[epoch 0/{cfg.max_iterations}]: cost = {errZ:.6f}, "
            f"rmse = {rmse:.6f}, jacc = {jacc:.6f}, time = 0.0000s"
        )

    cost_hist: list[float] = []
    rmse_hist: list[float] = []
    jacc_hist: list[float] = []
    time_hist: list[float] = []
    t0 = time.time()

    for epoch in range(1, cfg.max_iterations + 1):
        nvtx_range_push("epoch: " + str(epoch))
        report_epoch = epoch % cfg.eval_interval == 0
        total_errZ, total_mse, sumSr, num_jacc = 0.0, 0.0, 0.0, 0.0

        for blocks in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=cfg.device, dtype=A.dtype)
            if report_epoch:
                mse_batch = torch.zeros((), device=cfg.device, dtype=A.dtype)
                sumSr_batch = torch.zeros((), device=cfg.device, dtype=A.dtype)
                num_jacc_batch = torch.zeros((), device=cfg.device, dtype=A.dtype)

            for block_id, edge_idx, row_indices in blocks:
                (
                    loss_block,
                    mse_block,
                    sumSr_block,
                    jacc_num_block,
                ) = _block_eval(A, B, edge_idx, row_indices)
                loss = loss + loss_block
                if report_epoch:
                    mse_batch = mse_batch + mse_block.detach()
                    sumSr_batch = sumSr_batch + sumSr_block
                    num_jacc_batch = num_jacc_batch + jacc_num_block

            loss.backward()
            apply_clip_and_step(opt, A, B, cfg)

            if report_epoch:
                total_errZ += float(loss.detach().item())
                total_mse += float(mse_batch.item())
                sumSr += float(sumSr_batch.detach().item())
                num_jacc += float(num_jacc_batch.detach().item())

        if report_epoch:
            denom_jacc = float(torch.sum(S_value).item()) + sumSr - num_jacc
            jacc = 1.0 - num_jacc / denom_jacc
            S_norm = float(torch.norm(S_value).item())
            rmse = math.sqrt(total_mse) / (S_norm + 1e-16)
            errZ = math.sqrt(total_errZ) / (S_norm + 1e-16)
            report_time = time.time()
            elapsed = report_time - t0
            log = (
                f"[epoch {epoch}/{cfg.max_iterations}]: cost = {errZ:.6f}, "
                f"rmse = {rmse:.6f}, jacc = {jacc:.6f}, "
                f"time = {elapsed:.4f}s"
            )
            if cfg.verbose:
                print(log)
            cost_hist.append(errZ)
            rmse_hist.append(rmse)
            jacc_hist.append(jacc)
            time_hist.append(elapsed)
        nvtx_range_pop()
    total = time.time() - t0
    if cfg.verbose:
        print(f"\nTotal elapsed time: {total:.2f} sec")

    rmse, jacc, errZ = eval(
        eval_kernel,
        A,
        B,
        S_index,
        S_value,
        full_block_loader=loader,
        device=A.device,
    )
    if cfg.verbose:
        print(
            f"EVAL: cost={errZ:.6f}, rmse={rmse:.6f}, "
            f"jacc={jacc:.6f}, time={total:.4f}s"
        )

    history = {
        "cost": cost_hist,
        "rmse": rmse_hist,
        "jacc": jacc_hist,
        "time": time_hist,
    }
    return A.detach(), B.detach(), history
