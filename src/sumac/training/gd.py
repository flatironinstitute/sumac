from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn.utils import clip_grad_norm_

from ..kernels.cuda_utils import cuda_is_available
from sumac.config.options import OptimizerName

# OptimizerName = Literal["adam", "adamw", "sgd", "muon"]


@dataclass
class TrainConfig:
    d: int = 64
    num_blocks: int = 64
    epochs: int = 10
    lr: float = 1e-2
    seed: int = 0
    device: torch.device | str = "cuda" if cuda_is_available() else "cpu"
    shuffle_blocks: bool = False
    batch_blocks: int = 1
    eval_interval: int = 100
    optimizer: OptimizerName = OptimizerName.ADAM
    momentum: float = 0.7
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95


def make_optimizer(
    name: OptimizerName,
    params: list[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-8,
    muon_momentum: float = 0.95
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
            momentum=muon_momentum,
            adjust_lr_fn="match_rms_adamw",
        )
    raise ValueError(f"Unknown optimizer: {name}")


def apply_clip_and_step(opt, A, B, cfg):
    if cfg.optimizer.lower() == 'sgd':
        clip_grad_norm_([A, B], max_norm=1.0)
    opt.step()
