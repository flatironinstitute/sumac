import math
import time
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch import Tensor

from sumac.config.options import OptimizerName, SumacConfig
from sumac.datasets import collate_blocks, StochasticRowBlockDataset
from sumac.eval import block_loss_and_pred, eval
from sumac.kernels.cuda_utils import nvtx_range_pop, nvtx_range_push


# TODO:
# - further review
# - remove performance reporting instrumentation

def make_optimizer(
    name: OptimizerName,
    params: list[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
):
    if name == OptimizerName.ADAM:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=adam_betas, eps=adam_eps)
    if name == OptimizerName.ADAMW:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=adam_betas, eps=adam_eps)
    if name == OptimizerName.SGD:
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == OptimizerName.MUON:
        return torch.optim.Muon(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            adjust_lr_fn="match_rms_adamw",
        )
    raise ValueError(f"Unknown optimizer: {name}")


def apply_clip_and_step(opt, A, B, cfg):
    if cfg.optimizer == OptimizerName.SGD:
        clip_grad_norm_([A, B], max_norm=1.0)
    opt.step()


# TODO: Can we slim this down
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

    (gen, _, _) = cfg.get_generator()
    
    # Move data to the training device.
    S_index = S_index.to(cfg.device)
    S_value = S_value.to(cfg.device)
    if A_init is None or B_init is None:
        scale = 0.5 * math.sqrt(S_value.mean().item() / cfg.rank)
        A = torch.nn.Parameter(torch.rand(m, cfg.rank, device=cfg.device) * scale)
        B = torch.nn.Parameter(torch.rand(n, cfg.rank, device=cfg.device) * scale)
    else:
        A = torch.nn.Parameter(A_init.to(cfg.device))
        B = torch.nn.Parameter(B_init.to(cfg.device))
    opt = make_optimizer(
        cfg.optimizer,
        [A, B],
        lr=cfg.learning_rate,
        weight_decay=0.0,
        momentum=cfg.momentum,
        adam_betas=cfg.adam_betas,
        adam_eps=cfg.adam_eps,
    )

    # DataLoader
    ds = StochasticRowBlockDataset(S_index, S_value, m, cfg.num_blocks, gen)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_blocks,
        shuffle=cfg.shuffle_blocks,
        collate_fn=collate_blocks
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
                edge_idx = edge_idx.to(cfg.device).view(-1)
                mse_block, sumSr_block, jacc_num_block, errZ_block = block_loss_and_pred(
                    A,
                    B,
                    block_id=int(block_id),
                    num_blocks=cfg.num_blocks,
                    m=m,
                    n=n,
                    S_index=S_index,
                    S_value=S_value,
                    edge_idx=edge_idx,
                    row_indices=row_indices,
                    errZ_obj=True,
                )
                loss_block = errZ_block if errZ_block is not None else mse_block
                loss = loss + loss_block
                sumSr_batch = sumSr_batch + torch.as_tensor(sumSr_block, device=cfg.device, dtype=A.dtype)
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
        print(log)
        history.append(log)
        nvtx_range_pop()
    total = time.time() - t0
    print(f"\nTotal elapsed time: {total:.2f} sec")

    rmse, jacc, errZ = eval(
        A,
        B,
        S_index,
        S_value,
        m,
        n,
        num_blocks=cfg.num_blocks,
        full_block_loader=loader,
        device=A.device,
        errZ_obj=True
    )
    print(f"EVAL: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")

    return A.detach(), B.detach(), history
