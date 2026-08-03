import torch

from sumac.utils import round_up
from sumac.kernels.tuning.autotune import AutotuneCudaKernel
from sumac.kernels.relu_batc_tf32_jit.api import relu_bat_c_tf32_mma_sync
from sumac.kernels.tuning.tuning_types import ReluBatCTf32SyncTuneConfig, T_ReluBatCParams, T_ReluBatCReturn, make_config_list
from sumac.kernels.tuning.relu_bat_c.relu_bat_c_base import relu_bat_c_fallback


def relu_bat_c_tf32_sync_tune_config(D: int) -> list[ReluBatCTf32SyncTuneConfig]:
    if D == 64:
        base = ReluBatCTf32SyncTuneConfig(BM = 128, BN = 32, M_TILES = 1, num_stages = 1)
        return make_config_list(
            base,
            {
                'BM': [128, 256],
                'BN': [32, 64, 128],
                'M_TILES': [1, 2, 4],
                'num_stages': [1, 2, 3]
            }
        )
    if D == 128:
        base = ReluBatCTf32SyncTuneConfig(BM = 64, BN = 8, M_TILES = 1, num_stages = 1)
        return make_config_list(
            base,
            {
                'BM': [64, 128, 256],
                'BN': [8, 16, 32],
                'M_TILES': [1, 2, 4],
                'num_stages': [1, 2, 3]
            }
        )
    if D == 256:
        base = ReluBatCTf32SyncTuneConfig(BM = 64, BN = 8, M_TILES = 1, num_stages = 1)
        return make_config_list(
            base,
            {
                'BM': [64, 128, 256],
                'BN': [8, 16],
                'M_TILES': [1, 2],
                'num_stages': [1, 2]
            }
        )
    base = ReluBatCTf32SyncTuneConfig(BM = 128, BN = 16, M_TILES = 2, num_stages = 1)
    return make_config_list(
        base,
        {
            'BM': [128, 256],
            'BN': [16, 32, 64],
            'M_TILES': [2, 4],
            'num_stages': [1, 2, 3]
        }
    )


class AutotuneReluBatCTf32Sync(AutotuneCudaKernel[ReluBatCTf32SyncTuneConfig, T_ReluBatCParams, T_ReluBatCReturn]):
    def __init__(self, *args, **kwargs):
        kwargs['wrapped_fn_name'] = relu_bat_c_tf32_mma_sync.__name__
        kwargs['wrapped_fn_module'] = relu_bat_c_tf32_mma_sync.__module__
        super().__init__(*args, **kwargs)


    def _candidate_fn(self, params: T_ReluBatCParams, config: ReluBatCTf32SyncTuneConfig):
        (A, B, C) = params
        return relu_bat_c_tf32_mma_sync(
            A,
            B,
            C,
            BM=config.BM,
            BN=config.BN,
            M_TILES=config.M_TILES,
            num_stages=config.num_stages
        )


    def _fallback(self, params: T_ReluBatCParams) -> T_ReluBatCReturn:
        (A, B, C) = params
        return relu_bat_c_fallback(A, B, C)


    def _constraint(self, params: T_ReluBatCParams, config: ReluBatCTf32SyncTuneConfig) -> bool:
        (A, _, _) = params
        props = torch.cuda.get_device_properties(A.device)
        _, D = A.shape

        if props.major < 8: return False
        if D < 1: return False
        if config.BN % 8 != 0: return False
        if config.num_stages < 1: return False

        warp_m_rows = config.M_TILES * 16
        if config.BM % warp_m_rows != 0: return False

        compute_warps = config.BM // warp_m_rows
        if compute_warps < 1 or compute_warps > 8: return False

        _block_mem = getattr(props, "shared_memory_per_block", 0)
        max_smem = getattr(props, "shared_memory_per_block_optin", _block_mem)
        smem_bytes = 2 * config.num_stages * config.BN * round_up(D, 8) * 4 + 127
        return smem_bytes <= max_smem
