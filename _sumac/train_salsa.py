import torch
from functools import lru_cache
from torch import Tensor
from _sumac.dataset import StochasticRowBlockDataset
from relu_batc_jit.api import relu_bat_c_fused
from relu_batc_tf32_jit.jit_kernel_tf32_sync import launch_relu_batc_mma_sync_tf32
from relu_batc_tf32_jit.jit_kernel_tf32_wgmma import launch_relu_bat_c_wgmma_tf32_tma
from _sumac.tuning import *
import triton
import triton.language as tl

def relu_bat_c_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BK: int,
    num_ms: int,
) -> bool:
    if A.shape[1] >= 32 and num_ms > 2:
        return False

    if A.shape[1] >= 64 and num_ms > 1:  
        return False  
    
    props = torch.cuda.get_device_properties(torch.cuda.current_device())

    if props.shared_memory_per_block < 8 * BK * A.shape[1]:
        return False

    return True


@torch.compile(mode='max-autotune-no-cudagraphs')
def relu_bat_c_fallback(A: torch.Tensor,
                        B: torch.Tensor,
                        C: torch.Tensor):
    return torch.relu(B @ A.T) @ C


def relu_bat_c_cuda_launcher():
    tune_config = {
        "BM": [32, 64, 128],
        "BK": [16, 32, 64],
        "num_ms": [1, 2, 4, 6],
    }

    @autotune_cuda_kernel(
        configs=tune_config,
        fallback_fn=relu_bat_c_fallback,
        constraint_fn=relu_bat_c_constraints,
        key_fn=relu_bat_c_key,
        cache_path="relu_bat_c_jit_autotune.json",
        n_trials=1000,
        warmup=1,
        rep=5,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_batc(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BK: int,
        num_ms: int,
    ) -> torch.Tensor:
        return relu_bat_c_fused(
            A,
            B,
            C,
            BM=BM,
            BK=BK,
            MS=num_ms,
        )

    return relu_batc


def grid_size(config: dict) -> int:
    size = 1
    for values in config.values():
        size *= len(values)
    return size


def max_dynamic_smem_bytes(props) -> int:
    candidates = [
        int(getattr(props, "shared_memory_per_block", 0) or 0),
        int(getattr(props, "shared_memory_per_block_optin", 0) or 0),
    ]
    if getattr(props, "major", 0) == 9:
        candidates.append(227 * 1024)
    return max(candidates)


def relu_bat_c_tf32_sync_tune_config(D: int) -> dict:
    if D == 64:
        return {
            "BM": [128, 256],
            "BN": [32, 64, 128],
            "M_TILES": [1, 2, 4],
            "num_stages": [1, 2, 3],
        }
    if D == 128:
        return {
            "BM": [64, 128, 256],
            "BN": [8, 16, 32],
            "M_TILES": [1, 2, 4],
            "num_stages": [1, 2, 3],
        }
    if D == 256:
        return {
            "BM": [64, 128, 256],
            "BN": [8, 16],
            "M_TILES": [1, 2],
            "num_stages": [1, 2],
        }

    return {
        "BM": [128, 256],
        "BN": [16, 32, 64],
        "M_TILES": [2, 4],
        "num_stages": [1, 2, 3],
    }


def relu_bat_c_tf32_sync_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    M_TILES: int,
    num_stages: int,
) -> bool:
    _, D = A.shape
    props = torch.cuda.get_device_properties(A.device)

    if props.major < 8:
        return False
    if D % 8 != 0:
        return False
    if BN % 8 != 0:
        return False
    if num_stages < 1:
        return False

    warp_m_rows = M_TILES * 16
    if BM % warp_m_rows != 0:
        return False

    compute_warps = BM // warp_m_rows
    if compute_warps < 1 or compute_warps > 8:
        return False

    max_smem = getattr(
        props,
        "shared_memory_per_block_optin",
        props.shared_memory_per_block,
    )
    smem_bytes = 2 * num_stages * BN * D * 4 + 127
    return smem_bytes <= max_smem


@lru_cache(maxsize=None)
def relu_bat_c_tf32_sync_launcher(D: int):
    tune_config = relu_bat_c_tf32_sync_tune_config(D)

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_tf32_sync_constraints,
        cache_path="relu_bat_c_tf32_mma_autotune.json",
        n_trials=grid_size(tune_config),
        warmup=1,
        rep=5,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_bat_c_tf32_sync_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BN: int,
        M_TILES: int,
        num_stages: int,
    ) -> torch.Tensor:
        return launch_relu_batc_mma_sync_tf32(
            A,
            B,
            C,
            BM=BM,
            BN=BN,
            M_TILES=M_TILES,
            num_stages=num_stages,
        )

    return relu_bat_c_tf32_sync_cuda


def relu_bat_c_tf32_wgmma_tune_config(D: int) -> dict:
    if D == 16:
        return {
            "BM": [256, 320],
            "BN": [64, 128],
            "WGMMA_S_N": [64],
            "WGMMA_Y_N": [16],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }
    if D == 32:
        return {
            "BM": [192, 256, 320],
            "BN": [64, 128],
            "WGMMA_S_N": [64],
            "WGMMA_Y_N": [32],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }
    if D == 64:
        return {
            "BM": [128, 192, 256],
            "BN": [64, 128],
            "WGMMA_S_N": [64],
            "WGMMA_Y_N": [64],
            "num_stages": [2],
            "wgmma_mode": ["RS"],
        }
    if D == 128:
        return {
            "BM": [128, 192, 256],
            "BN": [32, 64, 128],
            "WGMMA_S_N": [32, 64],
            "WGMMA_Y_N": [64, 128],
            "num_stages": [2],
            "wgmma_mode": ["SS"],
        }
    if D == 256:
        return {
            "BM": [64, 128, 192],
            "BN": [16, 32, 64],
            "WGMMA_S_N": [16, 32],
            "WGMMA_Y_N": [64, 128],
            "num_stages": [2],
            "wgmma_mode": ["SS"],
        }

    wgmma_n_values = [16, 32, 64, 128]
    y_shapes = [n for n in wgmma_n_values if D % n == 0] or [16]
    return {
        "BM": [64, 128, 192, 256, 320],
        "BN": [16, 32, 64, 128, 256],
        "WGMMA_S_N": wgmma_n_values,
        "WGMMA_Y_N": y_shapes,
        "num_stages": [1, 2],
        "wgmma_mode": ["RS", "SS"],
    }


def relu_bat_c_tf32_wgmma_constraints(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    BM: int,
    BN: int,
    WGMMA_S_N: int,
    WGMMA_Y_N: int,
    num_stages: int,
    wgmma_mode: str,
) -> bool:
    _, D = A.shape
    props = torch.cuda.get_device_properties(A.device)

    if props.major != 9:
        return False
    if WGMMA_S_N not in (16, 32, 64, 128):
        return False
    if WGMMA_Y_N not in (16, 32, 64, 128):
        return False
    if BN % WGMMA_S_N != 0:
        return False
    if D % WGMMA_Y_N != 0:
        return False
    if num_stages not in (1, 2, 3):
        return False
    if wgmma_mode not in ("RS", "SS"):
        return False
    if BM % 64 != 0:
        return False

    compute_warpgroups = BM // 64
    threads_per_block = (compute_warpgroups + 1) * 128
    if compute_warpgroups < 1:
        return False
    if threads_per_block > getattr(props, "max_threads_per_block", 1024):
        return False

    smem_elems = num_stages * 2 * BN * D
    if wgmma_mode == "SS":
        smem_elems += BM * D
    smem_bytes = smem_elems * 4 + 127
    return smem_bytes <= max_dynamic_smem_bytes(props)


@lru_cache(maxsize=None)
def relu_bat_c_tf32_wgmma_launcher(D: int):
    tune_config = relu_bat_c_tf32_wgmma_tune_config(D)

    @autotune_cuda_kernel(
        configs=tune_config,
        key_fn=relu_bat_c_key,
        constraint_fn=relu_bat_c_tf32_wgmma_constraints,
        cache_path="relu_bat_c_tf32_wgmma_mode_autotune.json",
        n_trials=grid_size(tune_config),
        warmup=1,
        rep=5,
        sampler=optuna.samplers.GridSampler(search_space=tune_config),
    )
    def relu_bat_c_tf32_wgmma_cuda(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        BM: int,
        BN: int,
        WGMMA_S_N: int,
        WGMMA_Y_N: int,
        num_stages: int,
        wgmma_mode: str,
    ) -> torch.Tensor:
        return launch_relu_bat_c_wgmma_tf32_tma(
            A,
            B,
            C,
            BM=BM,
            BN=BN,
            WGMMA_S_N=WGMMA_S_N,
            WGMMA_Y_N=WGMMA_Y_N,
            num_stages=num_stages,
            wgmma_mode=wgmma_mode,
        )

    return relu_bat_c_tf32_wgmma_cuda


def _select_relu_bat_c_kernel_mode(allow_tf32: bool, device) -> str:
    if not allow_tf32 or not torch.cuda.is_available():
        return "fp32_cuda"

    props = torch.cuda.get_device_properties(device)
    if props.major == 9:
        return "tf32_wgmma"
    if props.major >= 8:
        return "tf32_mma_sync"
    return "fp32_cuda"


relu_bat_c_tuned = relu_bat_c_cuda_launcher()
_relu_bat_c_kernel_mode = "fp32_cuda"
_relu_bat_c_kernel_d = None


def configure_kernel_prec(
    *,
    allow_tf32: bool,
    device,
    D: int,
) -> None:
    global relu_bat_c_tuned
    global _relu_bat_c_kernel_mode
    global _relu_bat_c_kernel_d

    mode = _select_relu_bat_c_kernel_mode(allow_tf32, device)
    if mode == _relu_bat_c_kernel_mode and D == _relu_bat_c_kernel_d:
        return

    if mode == "tf32_wgmma":
        relu_bat_c_tuned = relu_bat_c_tf32_wgmma_launcher(D)
    elif mode == "tf32_mma_sync":
        relu_bat_c_tuned = relu_bat_c_tf32_sync_launcher(D)
    else:
        relu_bat_c_tuned = relu_bat_c_cuda_launcher()

    _relu_bat_c_kernel_mode = mode
    _relu_bat_c_kernel_d = D



def lsq_update_single_gpu(
    Ar_dev: torch.Tensor,
    B_blk_dev: torch.Tensor,
    pinvAt_dev: torch.Tensor,
    dB_blk_dev: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    blk_vals: torch.Tensor,  
    momentum: torch.Tensor,
    unbias: torch.Tensor,
    lrate: Tensor | float,
) -> tuple[Tensor, Tensor]:

    stepM_blk = relu_bat_c_tuned(Ar_dev, B_blk_dev, pinvAt_dev)

    Lij_blk = torch.sum(Ar_dev[edge_i, :] * B_blk_dev[edge_j, :], dim=1)
    Mij_blk = torch.relu(Lij_blk)
    Ct_vals = blk_vals - Lij_blk + Mij_blk

    
    stepC_blk = torch.zeros_like(B_blk_dev)
    
    stepC_blk.index_add_(
        0,
        edge_j,
        Ct_vals[:, None] * pinvAt_dev[edge_i, :],
    )

    lsqB_blk = B_blk_dev - stepM_blk + stepC_blk
    dB_blk_new = (lsqB_blk - B_blk_dev) * (1 - momentum) + dB_blk_dev * momentum
    B_blk_new = B_blk_dev + lrate * dB_blk_new / unbias
    return B_blk_new, dB_blk_new


@torch.compile(mode='max-autotune-no-cudagraphs')
def batch_update_single_gpu(
    S_idx_full: torch.Tensor,
    S_val_full: torch.Tensor,
    edge_idx: torch.Tensor,
    Factor_fixed: torch.Tensor,
    row_indices: torch.Tensor,
    B: torch.Tensor,
    dB: torch.Tensor,
    momentum,
    unbias,
    lrate,
    m_fixed: int,
):
    
    dev = B.device

    m_batch = row_indices.shape[0]

    Ar_dev = Factor_fixed[row_indices, :]
    GramA = Ar_dev.T @ Ar_dev
    pinvAt_dev = torch.linalg.solve(GramA, Ar_dev.T).T

    blk_idx = S_idx_full[:, edge_idx]
    blk_vals = S_val_full[edge_idx]

    local_map = torch.zeros(m_fixed, dtype=torch.long, device=dev)

    local_map[row_indices] = torch.arange(m_batch, device=dev)
    edge_i = local_map[blk_idx[0]]
    edge_j = blk_idx[1]
    


    B_new, dB_new = lsq_update_single_gpu(
        Ar_dev=Ar_dev,
        B_blk_dev=B,
        pinvAt_dev=pinvAt_dev,
        dB_blk_dev=dB,
        edge_i=edge_i,
        edge_j=edge_j,
        blk_vals=blk_vals,
        momentum=momentum,
        unbias=unbias,
        lrate=lrate,
    )

    return B_new, dB_new


def update_factor_salsa(
    S_idx_full: Tensor,
    S_val_full: Tensor,
    dataset: StochasticRowBlockDataset,
    block_id: int,
    Factor_fixed: Tensor,
    Factor_update: Tensor,
    dFactor: Tensor,
    momentum: Tensor,
    unbias: Tensor,
    lrate: float | Tensor,
):
    m_fixed = Factor_fixed.shape[0]
    _, edge_idx, row_indices = dataset[block_id]

    # TODO: NOTE: Unused
    params = relu_bat_c_tuned.resolve_params(Factor_fixed[row_indices, :], Factor_update, Factor_fixed[row_indices, :])
    #need to resolve params outside of the compiled region

    nextF, dF = batch_update_single_gpu(
        S_idx_full=S_idx_full,
        S_val_full=S_val_full,
        edge_idx=edge_idx,
        Factor_fixed=Factor_fixed,
        row_indices=row_indices,
        B=Factor_update,
        dB=dFactor,
        momentum=momentum,
        unbias=unbias,
        lrate=lrate,
        m_fixed=m_fixed,
    )
    
    return nextF, dF
