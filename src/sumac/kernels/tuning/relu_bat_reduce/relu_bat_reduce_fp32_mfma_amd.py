from __future__ import annotations

import torch

from sumac.kernels.relu_bat_reduce_jit_amd.api import (
    relu_bat_reduce_fp32_mfma,
    relu_bat_reduce_fp32_mfma_available,
)
from sumac.kernels.tuning.autotune import AutotuneCudaKernel, grid_size
from sumac.kernels.tuning.tuning_types import (
    ReluBatReduceMfmaTuneConfig,
    T_ReluBatReduceParams,
    T_ReluBatReduceReturn,
    make_choices,
    make_config_list,
)
from sumac.utils import is_power_of_two, round_up

from .relu_bat_reduce import relu_bat_reduce_fallback


_DEFAULT_CONFIG = ReluBatReduceMfmaTuneConfig(
    BM=16,
    BN=16,
    M_TILES=1,
)

relu_bat_reduce_mfma_tune_config = make_config_list(
    _DEFAULT_CONFIG,
    {
        "BM": [16, 32, 64, 128, 256],
        "BN": [16, 32, 64, 128],
        "M_TILES": [1, 2, 4],
    },
)


class AutotuneReluBatReduceMfmaAMD(
    AutotuneCudaKernel[
        ReluBatReduceMfmaTuneConfig,
        T_ReluBatReduceParams,
        T_ReluBatReduceReturn,
    ]
):
    def __init__(self, *args, **kwargs):
        configs = (
            kwargs.get("configs", relu_bat_reduce_mfma_tune_config)
            or relu_bat_reduce_mfma_tune_config
        )
        kwargs["wrapped_fn_name"] = relu_bat_reduce_fp32_mfma.__name__
        kwargs["wrapped_fn_module"] = relu_bat_reduce_fp32_mfma.__module__
        kwargs["configs"] = configs
        kwargs["cache_path"] = (
            kwargs.get(
                "cache_path",
                "relu_bat_reduce_hip_mfma_runtime_autotune.json",
            )
            or "relu_bat_reduce_hip_mfma_runtime_autotune.json"
        )
        kwargs["n_trials"] = kwargs.get(
            "n_trials",
            grid_size(make_choices(configs)),
        )
        super().__init__(*args, **kwargs)

    def resolve_decision(self, params: T_ReluBatReduceParams):
        A, _ = params
        D = int(A.shape[1]) if A.ndim == 2 else None
        if not relu_bat_reduce_fp32_mfma_available(A.device, D):
            self._set_interface_fn()
            return
        return super().resolve_decision(params)

    def _candidate_fn(
        self,
        params: T_ReluBatReduceParams,
        config: ReluBatReduceMfmaTuneConfig,
    ) -> T_ReluBatReduceReturn:
        A, B = params
        return relu_bat_reduce_fp32_mfma(
            A,
            B,
            BM=config.BM,
            BN=config.BN,
            M_TILES=config.M_TILES,
        )

    def _fallback(
        self,
        params: T_ReluBatReduceParams,
    ) -> T_ReluBatReduceReturn:
        A, B = params
        return relu_bat_reduce_fallback(A, B)

    def _constraint(
        self,
        params: T_ReluBatReduceParams,
        config: ReluBatReduceMfmaTuneConfig,
    ) -> bool:
        A, _ = params
        _, D = A.shape
        if not relu_bat_reduce_fp32_mfma_available(A.device, D):
            return False
        if config.BM <= 0 or config.BN <= 0 or config.M_TILES <= 0:
            return False
        if config.BN % 16 != 0:
            return False

        wave_m_rows = config.M_TILES * 16
        if config.BM % wave_m_rows != 0:
            return False
        waves_per_block = config.BM // wave_m_rows
        if waves_per_block < 1 or waves_per_block > 16:
            return False

        threads_per_block = waves_per_block * 64
        if not is_power_of_two(threads_per_block):
            return False

        props = torch.cuda.get_device_properties(A.device)
        if threads_per_block > int(
            getattr(props, "max_threads_per_block", 1024) or 1024
        ):
            return False

        if D < 1:
            return False
        D_f = round_up(D, 4)
        panel_smem_bytes = config.BN * D_f * 4
        reduction_smem_bytes = 2 * threads_per_block * 4
        smem_bytes = panel_smem_bytes + reduction_smem_bytes
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

        a_words = config.M_TILES * D_f // 4
        score_words = 4 * config.M_TILES
        reduction_words = 2
        register_floor_per_lane = a_words + score_words + reduction_words
        if register_floor_per_lane > 256:
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
    "AutotuneReluBatReduceMfmaAMD",
    "relu_bat_reduce_mfma_tune_config",
]
