
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from torch import Tensor

from sumac.config import AutotuneMode

@dataclass
class ReluBatCFp32TuneConfig:
    BM: list[int]
    BK: list[int]
    num_ms: list[int]


@dataclass
class ReluBatCTf32SyncTuneConfig:
    BM: list[int]
    BN: list[int]
    M_TILES: list[int]
    num_stages: list[int]


@dataclass
class ReluBatCTf32WgmmaTuneConfig:
    BM: list[int]
    BN: list[int]
    WGMMA_S_N: list[int]
    WGMMA_Y_N: list[int]
    num_stages: list[int]
    wgmma_mode: list[str]


@dataclass
class ReluBatReduceTuneConfig:
    BM: list[int]
    BK: list[int]
    num_ms: list[int]


T_TuneConfig = ReluBatCFp32TuneConfig | ReluBatCTf32SyncTuneConfig | ReluBatCTf32WgmmaTuneConfig | ReluBatReduceTuneConfig

T_ReluBatCParams = Tuple[Tensor, Tensor, Tensor]
T_ReluBatReduceParams = Tuple[Tensor, Tensor]
T_FnParams = T_ReluBatCParams | T_ReluBatReduceParams

T_ReluBatCReturn = Tensor
T_ReluBatReduceReturn = Tuple[Tensor, Tensor]
T_FnReturns = T_ReluBatCReturn | T_ReluBatReduceReturn

class TuneResultMode(Enum):
    CUDA = "cuda"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class TuneResult:
    mode: TuneResultMode
    # TODO: This can be done better, some kind of covariant stuff or something idk
    params: Dict[str, Any]      ## TODO: This can be a TuneConfig subclass?
    runtime_ms: float


@dataclass(frozen=True)
class KernelAutotuneOptions:
    mode: AutotuneMode = AutotuneMode.CACHE
    cache_dir: Path | None = None
    cache_dir_key: str | None = None
    verbose: bool = False
