import numpy as np
import os
import pickle
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from typing import Literal

from sumac import sumac_factorize
from sumac.datasets import RowBlockDataset, collate_blocks
from sumac.eval import eval
from sumac.config.options import SumacConfig


example_type = Literal['digits'] | Literal['bigrams'] | Literal['connectome']


def make_paths(cfg: SumacConfig, etype: example_type) -> tuple[str, str | None]:
    save_dir = f"{etype}_{cfg.method.value}"
    chunks = [
        f"sumac_rank={cfg.rank}",
        f"mom={cfg.momentum}",
        f"seed={cfg.seed}",
        f"iters={cfg.max_iterations}",
        f"nblocks={cfg.num_blocks}",
        f"learning_rate={cfg.learning_rate}",
        f"optim={cfg.optimizer.value}"
    ]
    computed_path = "_".join(chunks)
    save_path = f"./{save_dir}/{computed_path}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if cfg.log_filename is not None:
        log_dir = os.path.dirname(cfg.log_filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    return (save_path, cfg.log_filename)


## Shared data loading

def load_data(filename: str, ex: example_type, dtype: torch.dtype = torch.float32):
    if filename.endswith('.mat'):
        (m, n, S_index, S_value) = _load_matlab(filename, ex, dtype)
    else:
        (m, n, S_index, S_value) = _load_np_compatible(filename, ex)
    assert isinstance(m, int)
    assert isinstance(n, int)

    # TODO: Gate this on verbosity
    print(f'm={m}, n={n}, E={len(S_value)}')
    return (m, n, S_index, S_value)


def _load_matlab(filename: str, ex: example_type, dtype: torch.dtype = torch.float32):
    matrix_name = 'S' if ex == 'digits' else ex
    mat = sio.loadmat(filename)
    S = mat[matrix_name]
    S = S.tocoo()

    if ex == 'bigrams':
        # need to normalize each nonzero entry by the sum of its row
        row_sums = np.asarray(S.sum(axis=1)).ravel().astype(np.float32)
        realmin = np.finfo(np.float32).tiny
        S_data = S.data.astype(np.float32) / (row_sums[S.row] + realmin)
    else:
        S_data = S.data
    S_index = torch.tensor(np.array([S.row, S.col]), dtype=torch.long)
    S_value = torch.tensor(S_data, dtype=dtype)
    m, n = S.shape
    return (m, n, S_index, S_value)


def _load_np_compatible(filename: str, ex: example_type):
    data = np.loadtxt(filename).T
    S_index = torch.LongTensor(data[0:2,:])
    S_value = torch.FloatTensor(data[2,:])

    if ex == 'connectome':
        m = n = int(S_index[0].max())
    else:
        # bigram, digit files aren't necessarily square
        m = int(S_index[0].max().item())
        n = int(S_index[1].max().item())
    # normalize to start at zero-index
    S_index -= 1
    return (m, n, S_index, S_value)


## Main operation

def load_factorize_save(config: SumacConfig, etype: example_type, save_path: str):
    assert config.input_filename is not None

    (m, n, S_index, S_value) = load_data(config.input_filename, etype, config.dtype)

    A, B, costs = sumac_factorize(
        S_index=S_index,
        S_value=S_value,
        shape=(m, n),
        rank=config.rank,
        max_iterations=config.max_iterations,
        num_blocks=config.num_blocks,
        momentum=config.momentum,
        method=config.method,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        eval_interval=config.eval_interval,
        seed=config.seed if config.seed is not None else 0,
        allow_tf32=config.allow_tf32,
        device=config.device,
        autotune=config.autotune.value,
        autotune_cache_dir=config.autotune_cache_dir,
        autotune_verbose=config.autotune_verbose,
    )
    torch.save([A, B], f"{save_path}/AB.pt")
    with open(f"{save_path}/cost.pkl", "wb") as f:
        pickle.dump(costs, f)
    with open(f"{save_path}/opts.pkl", "wb") as f:
        pickle.dump(vars(config), f)

    eval_device = A.device
    do_eval(A, B, S_index, S_value, m, n, config, eval_device)



## Final/Sole Error Evaluation

def eval_only(cfg: SumacConfig, etype: example_type):
    assert cfg.input_filename is not None

    # ensure defaults
    eval_device = cfg.device
    cfg.eval_path = '' if cfg.eval_path is None else cfg.eval_path
    cfg.num_blocks = 100 if cfg.num_blocks is None else cfg.num_blocks
    if eval_device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        eval_device = torch.device(device_str)

    (m, n, S_index, S_value) = load_data(cfg.input_filename, etype, cfg.dtype)
    A, B = torch.load(f"{cfg.eval_path}/AB.pt", map_location="cpu")
    do_eval(A, B, S_index, S_value, m, n, cfg, eval_device, eval_only=True)

    #saving to txt
    if cfg.eval_save:
        print(A.shape, B.shape)
        np.savetxt(f"{cfg.eval_path}/A.txt", A.cpu().numpy(), fmt="%.8f")
        np.savetxt(f"{cfg.eval_path}/B.txt", B.cpu().numpy(), fmt="%.8f")
    exit()


def do_eval(
    A: torch.Tensor,
    B: torch.Tensor,
    S_index: torch.Tensor,
    S_value: torch.Tensor,
    m: int,
    n: int,
    cfg: SumacConfig,
    eval_device: torch.device,
    eval_only: bool = False
):
    assert cfg.num_blocks is not None
    A_eval = A.detach().to(eval_device)
    B_eval = B.detach().to(eval_device)
    S_index_eval = S_index.to(eval_device)
    S_value_eval = S_value.to(eval_device)

    ds = RowBlockDataset(
        S_index_eval,
        S_value_eval,
        m=m,
        num_blocks=cfg.num_blocks,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blocks)
    rmse, jacc, errZ = eval(
        A_eval,
        B_eval,
        S_index_eval,
        S_value_eval,
        m,
        n,
        num_blocks=cfg.num_blocks,
        full_block_loader=loader,
        device=eval_device,
    )
    label = "EVAL" if eval_only else "Final metrics"
    print(f"{label}: rmse={rmse:.6f}, jacc={jacc:.6f}, errZ={errZ:.6f}")
