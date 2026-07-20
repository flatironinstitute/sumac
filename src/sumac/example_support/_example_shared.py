import numpy as np
import os
import pickle
import scipy.io as sio
import torch
from typing import Literal

from sumac import sumac_factorize
from sumac.config import SumacConfig


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
        config=config
    )
    torch.save([A, B], f"{save_path}/AB.pt")
    with open(f"{save_path}/cost.pkl", "wb") as f:
        pickle.dump(costs, f)
    with open(f"{save_path}/opts.pkl", "wb") as f:
        pickle.dump(vars(config), f)
