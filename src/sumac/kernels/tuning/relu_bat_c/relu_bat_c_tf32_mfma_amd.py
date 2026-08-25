from __future__ import annotations

import torch

from sumac.kernels.relu_batc_tf32_jit_amd.api import (
    relu_bat_c_tf32_mfma,
    relu_bat_c_tf32_mfma_available,
)
from sumac.kernels.tuning.autotune import AutotuneCudaKernel
from sumac.kernels.tuning.tuning_types import (
    ReluBatCTf32MfmaTuneConfig,
    T_ReluBatCParams,
    T_ReluBatCReturn,
    make_config_list,
)
from sumac.utils import round_up

from .relu_bat_c_base import relu_bat_c_fallback


def relu_bat_c_tf32_mfma_tune_config(
    D: int,
) -> list[ReluBatCTf32MfmaTuneConfig]:

    if D >= 256:
        stage_choices = [1, 2, 3] if D > 256 else [2, 1, 3]
        template = ReluBatCTf32MfmaTuneConfig(
            BM=16, BN=16, M_TILES=1, num_stages=stage_choices[0]
        )
        choices = {
            "BM": [16, 32, 64],
            "BN": [16, 32],
            "M_TILES": [1, 2],
            "num_stages": stage_choices,
        }
    elif D >= 128:
        template = ReluBatCTf32MfmaTuneConfig(
            BM=32, BN=16, M_TILES=1, num_stages=2
        )
        choices = {
            "BM": [32, 64, 128],
            "BN": [16, 32, 64],
            "M_TILES": [1, 2, 4],
            "num_stages": [2, 1, 3],
        }
    elif D >= 64:
        template = ReluBatCTf32MfmaTuneConfig(
            BM=32, BN=16, M_TILES=1, num_stages=2
        )
        choices = {
            "BM": [32, 64, 128, 256],
            "BN": [16, 32, 64, 128],
            "M_TILES": [1, 2, 4],
            "num_stages": [2, 1, 3],
        }
    else:
        template = ReluBatCTf32MfmaTuneConfig(
            BM=64, BN=16, M_TILES=1, num_stages=2
        )
        choices = {
            "BM": [64, 128, 256],
            "BN": [16, 32, 64, 128],
            "M_TILES": [1, 2, 4],
            "num_stages": [2, 1, 3],
        }
    return make_config_list(template, choices)


class AutotuneReluBatCTf32MfmaAMD(
    AutotuneCudaKernel[
        ReluBatCTf32MfmaTuneConfig,
        T_ReluBatCParams,
        T_ReluBatCReturn,
    ]
):
    def __init__(self, *args, **kwargs):
        kwargs["wrapped_fn_name"] = relu_bat_c_tf32_mfma.__name__
        kwargs["wrapped_fn_module"] = relu_bat_c_tf32_mfma.__module__
        super().__init__(*args, **kwargs)

    def _candidate_fn(
        self,
        params: T_ReluBatCParams,
        config: ReluBatCTf32MfmaTuneConfig,
    ):
        A, B, C = params
        return relu_bat_c_tf32_mfma(
            A,
            B,
            C,
            BM=config.BM,
            BN=config.BN,
            M_TILES=config.M_TILES,
            num_stages=config.num_stages,
        )

    def _fallback(self, params: T_ReluBatCParams) -> T_ReluBatCReturn:
        A, B, C = params
        return relu_bat_c_fallback(A, B, C)

    def _constraint(
        self,
        params: T_ReluBatCParams,
        config: ReluBatCTf32MfmaTuneConfig,
    ) -> bool:
        A, _, _ = params
        _, D = A.shape
        if not relu_bat_c_tf32_mfma_available(A.device, D):
            return False
        if config.BM <= 0 or config.BN <= 0 or config.M_TILES <= 0:
            return False
        if config.num_stages not in (1, 2, 3):
            return False
        if config.BN % 16 != 0:
            return False

        wave_m_rows = config.M_TILES * 16
        if config.BM % wave_m_rows != 0:
            return False
        waves_per_block = config.BM // wave_m_rows
        if waves_per_block < 1 or waves_per_block > 16:
            return False

        props = torch.cuda.get_device_properties(A.device)
        threads_per_block = waves_per_block * 64
        if threads_per_block > int(
            getattr(props, "max_threads_per_block", 1024) or 1024
        ):
            return False

        if D < 1:
            return False
        D_f = round_up(D, 16)
        smem_bytes = 2 * config.num_stages * config.BN * D_f * 4
        default_smem = int(
            getattr(props, "shared_memory_per_block", 0) or 0
        )
        max_smem = max(
            default_smem,
            int(
                getattr(props, "shared_memory_per_block_optin", None)
                or default_smem
            ),
        )
        if smem_bytes > max_smem:
            return False

        b_words = config.M_TILES * D_f // 4
        y_words = config.M_TILES * D_f // 4
        score_words = 4 * config.M_TILES
        if b_words + score_words > 256 or y_words > 256:
            return False
        register_floor_per_lane = b_words + y_words + score_words
        if register_floor_per_lane > 512:
            return False

        register_budget = int(
            getattr(props, "regs_per_block", 0)
            or getattr(props, "regs_per_multiprocessor", 0)
            or 0
        )
        if (
            register_budget > 0
            and threads_per_block * register_floor_per_lane > register_budget
        ):
            return False
        return True


__all__ = [
    "AutotuneReluBatCTf32MfmaAMD",
    "relu_bat_c_tf32_mfma_tune_config",
]
