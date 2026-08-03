import torch

from sumac.utils import round_up
from sumac.kernels.tuning.autotune import AutotuneCudaKernel
from sumac.kernels.relu_batc_tf32_jit.api import relu_bat_c_tf32_wgmma
from sumac.kernels.tuning.tuning_types import ReluBatCTf32WgmmaTuneConfig, T_ReluBatCParams, T_ReluBatCReturn, make_config_list
from sumac.kernels.tuning.relu_bat_c.relu_bat_c_base import relu_bat_c_fallback


def relu_bat_c_tf32_wgmma_tune_config(D: int) -> list[ReluBatCTf32WgmmaTuneConfig]:
    # Don't worry, we don't actually use these
    base = ReluBatCTf32WgmmaTuneConfig(BM = 192, BN = 64, WGMMA_S_N=32, WGMMA_Y_N=16, num_stages=2, wgmma_mode='RS')

    if D <= 16:
        return make_config_list(
            base,
            {
                'BM': [192, 256, 320],
                'BN': [64, 128, 256],
                'WGMMA_S_N': [32, 64],
                'WGMMA_Y_N': [16],
                'num_stages': [2],
                'wgmma_mode': ['RS']
            }
    )
    if D <= 32:
        return make_config_list(
            base,
            {
                'BM': [192, 256, 320],
                'BN': [64, 128, 256],
                'WGMMA_S_N': [32, 64],
                'WGMMA_Y_N': [16, 32],
                'num_stages': [2],
                'wgmma_mode': ['RS']
            }
    )
    if D <= 72:
        return make_config_list(
            base,
            {
                'BM': [128, 192, 256],
                'BN': [64, 128, 256],
                'WGMMA_S_N': [32, 64, 128],
                'WGMMA_Y_N': [32, 64],
                'num_stages': [2],
                'wgmma_mode': ['RS']
            }
    )
    if D <= 144:
        return make_config_list(
            base,
            {
                'BM': [128, 192, 256],
                'BN': [32, 64, 128],
                'WGMMA_S_N': [32, 64, 128],
                'WGMMA_Y_N': [64, 128],
                'num_stages': [2],
                'wgmma_mode': ['RS', 'SS']
            }
    )

    return make_config_list(
        base,
        {
            'BM': [64, 128, 192, 256],
            'BN': [16, 32, 64, 128],
            'WGMMA_S_N': [16, 32, 64],
            'WGMMA_Y_N': [64, 128],
            'num_stages': [2],
            'wgmma_mode': ['SS']
        }
    )


def relu_bat_c_tf32_wgmma_available(A: torch.Tensor) -> bool:
    props = torch.cuda.get_device_properties(A.device)
    return props.major == 9


class AutotuneReluBatCTf32Wgmma(AutotuneCudaKernel[ReluBatCTf32WgmmaTuneConfig, T_ReluBatCParams, T_ReluBatCReturn]):
    def __init__(self, *args, **kwargs):
        kwargs['wrapped_fn_name'] = relu_bat_c_tf32_wgmma.__name__
        kwargs['wrapped_fn_module'] = relu_bat_c_tf32_wgmma.__module__
        super().__init__(*args, **kwargs)


    def _candidate_fn(self, params: T_ReluBatCParams, config: ReluBatCTf32WgmmaTuneConfig) -> T_ReluBatCReturn:
        (A, B, C) = params

        return relu_bat_c_tf32_wgmma(
            A,
            B,
            C,
            BM=config.BM,
            BN=config.BN,
            WGMMA_S_N=config.WGMMA_S_N,
            WGMMA_Y_N=config.WGMMA_Y_N,
            num_stages=config.num_stages,
            wgmma_mode=config.wgmma_mode
        )


    def _fallback(self, params: T_ReluBatCParams) -> T_ReluBatCReturn:
        (A, B, C) = params
        return relu_bat_c_fallback(A, B, C)


    def _constraint(self, params: T_ReluBatCParams, config: ReluBatCTf32WgmmaTuneConfig) -> bool:
        (A, _, _) = params
        _, D = A.shape
        props = torch.cuda.get_device_properties(A.device)

        if props.major != 9: return False
        if D < 1: return False
        # TODO: These would invalidate the config, move to the config definition as post_init
        if config.WGMMA_S_N not in (16, 32, 64, 128): return False
        if config.WGMMA_Y_N not in (16, 32, 64, 128): return False
        if config.BN % config.WGMMA_S_N != 0: return False
        if config.num_stages not in (1, 2, 3): return False
        if config.wgmma_mode not in ("RS", "SS"): return False
        if config.BM % 64 != 0: return False

        compute_warpgroups = config.BM // 64
        if compute_warpgroups < 1: return False
        threads_per_block = (compute_warpgroups + 1) * 128
        if threads_per_block > getattr(props, "max_threads_per_block", 1024): return False

        max_dynamic_smem_bytes = max([
            int(getattr(props, "shared_memory_per_block", 0) or 0),
            int(getattr(props, "shared_memory_per_block_optin", 0) or 0),
            # we know props.major == 9, b/c we can't have hit this point by now otherwise
            227 * 1024
        ])

        D_k_pad = round_up(D, 8)
        D_y_pad = round_up(D, config.WGMMA_Y_N)
        smem_elems = config.num_stages * config.BN * (D_k_pad + D_y_pad)
        if config.wgmma_mode == "SS":
            smem_elems += config.BM * D_k_pad
        smem_bytes = smem_elems * 4 + 127
        return smem_bytes <= max_dynamic_smem_bytes
