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
    S_value: Tensor,
    cfg: SumacConfig,
    m: int,
    n: int,
    gen: Generator,
    A_init: Tensor | None,
    B_init: Tensor | None,
):
    if A_init is None or B_init is None:
        scale = 0.5 * math.sqrt(S_value.mean().item() / cfg.rank)
        A = torch.nn.Parameter(
            torch.rand(
                m,
                cfg.rank,
                device=cfg.device,
                dtype=cfg.dtype,
                generator=gen,
            ) * scale
        )
        B = torch.nn.Parameter(
            torch.rand(
                n,
                cfg.rank,
                device=cfg.device,
                dtype=cfg.dtype,
                generator=gen,
            ) * scale
        )
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
        return (loss_block, torch.as_tensor(sumSr_block, device=device, dtype=A.dtype), jacc_num_block)

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

    (gen, gen_blocks, gen_loader) = cfg.get_generator()
    eval_kernel = AutotuneReluBatReduce()
    _block_eval = _init_block_loss(eval_kernel, S_index, S_value, cfg.device)
    
    # Move data to the training device.
    S_index = S_index.to(cfg.device)
    S_value = S_value.to(cfg.device)
    (A, B) = _init_factors(S_value, cfg, m, n, gen, A_init, B_init)
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

    history = []
    t0 = time.time()

    for epoch in range(1, cfg.max_iterations + 1):
        nvtx_range_push("epoch: " + str(epoch))
        total_loss, sumSr, num_jacc = 0.0, 0.0, 0.0
        t_start = time.time()

        for blocks in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=cfg.device, dtype=A.dtype)
            sumSr_batch = torch.zeros((), device=cfg.device, dtype=A.dtype)
            num_jacc_batch = torch.zeros((), device=cfg.device, dtype=A.dtype)

            for block_id, edge_idx, row_indices in blocks:
                (loss_block, sumSr_block, jacc_num_block) = _block_eval(A, B, edge_idx, row_indices)
                loss = loss + loss_block
                sumSr_batch = sumSr_batch + sumSr_block
                num_jacc_batch = num_jacc_batch + jacc_num_block

            loss.backward()
            apply_clip_and_step(opt, A, B, cfg)

            total_loss += float(loss.detach().item())
            sumSr += float(sumSr_batch.detach().item())
            num_jacc += float(num_jacc_batch.detach().item())

        time_step = time.time() - t_start
        denom_jacc = float(torch.sum(S_value).item()) + sumSr - num_jacc
        jacc = 1.0 - num_jacc / denom_jacc
        S_norm = float(torch.norm(S_value).item())
        rmse = math.sqrt(total_loss) / (S_norm + 1e-16)
        log = f"[epoch {epoch}/{cfg.max_iterations}]: rmse={rmse:.6f}, jacc={jacc:.6f}, factor_step ={time_step:6.4f}"
        if cfg.verbose:
            print(log)
        history.append(log)
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
        print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")

    return A.detach(), B.detach(), history
