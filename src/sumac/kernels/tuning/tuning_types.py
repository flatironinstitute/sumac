
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Tuple

from torch import Tensor

from sumac.config import AutotuneMode

@dataclass
class ReluBatCFp32TuneConfig:
    BM: int
    BK: int
    num_ms: int


@dataclass
class ReluBatCTf32SyncTuneConfig:
    BM: int
    BN: int
    M_TILES: int
    num_stages: int


@dataclass
class ReluBatCTf32WgmmaTuneConfig:
    BM: int
    BN: int
    WGMMA_S_N: int
    WGMMA_Y_N: int
    num_stages: int
    wgmma_mode: str


@dataclass
class ReluBatReduceTuneConfig:
    BM: int
    BK: int
    num_ms: int


T_TuneConfig = ReluBatCFp32TuneConfig | ReluBatCTf32SyncTuneConfig | ReluBatCTf32WgmmaTuneConfig | ReluBatReduceTuneConfig

T_ReluBatCParams = Tuple[Tensor, Tensor, Tensor]
T_ReluBatReduceParams = Tuple[Tensor, Tensor]
T_FnParams = T_ReluBatCParams | T_ReluBatReduceParams

T_ReluBatCReturn = Tensor
T_ReluBatReduceReturn = Tuple[Tensor, Tensor]
T_FnReturns = T_ReluBatCReturn | T_ReluBatReduceReturn


def make_choices(cfgs: Sequence[T_TuneConfig]) -> dict[str, list[int] | list[str]]:
    base = {f.name: [getattr(cfgs[0], f.name)] for f in fields(cfgs[0])}
    for c in cfgs[1:]:
        for f in fields(c):
            v = getattr(c, f.name)
            if v in base[f.name]: continue
            base[f.name].append(v)
    return base


def make_config_list[T_Cfg: T_TuneConfig](base: T_Cfg, d: dict[str, list[int] | list[str]]) -> list[T_Cfg]:
    res = []
    fieldnames = [f.name for f in fields(base)]
    vals = {fname: getattr(base, fname) for fname in fieldnames}    # won't actually be used
    while True:
        done = True
        for f in fieldnames:
            l = d.get(f, [])
            if len(l) > 0:
                done = False
                vals.update({f: l.pop(0)})
        if done:
            break
        res.append(replace(base, **vals))
    return res


class TuneResultMode(StrEnum):
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
