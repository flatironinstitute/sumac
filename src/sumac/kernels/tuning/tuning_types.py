
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
    """Turns a list of individual configs into a set of lists of their parameters, so
    that autotuning can treat them as a matrix of options.

    We want the Autotune class (and its descendants) to operate on an individual Config
    with defined fields. We would like to keep strong typing for the fields of the
    total parameter matrix used for tuning. However, there's no natural way to go from
    a Config type where each field is a list (e.g. BM: list[int]...) to a Config type
    where each field is the individual value (BM: int, ...).

    So to define the matrix of config parameters to test, we pass a list of Configs
    to the Autotune class, and it uses this function to create a matrix of the
    superset of every value that appears for each of the parameter keys.

    Args:
        cfgs (Sequence[T_TuneConfig]): A collection of configurations; we'll take
        every value that appears in any of them as part of the test matrix.

    Returns:
        dict[str, list[int] | list[str]]: A dictionary which points each field
        in the Config type to a list of all the values that appeared in any of
        the "candidate" configurations.
    """
    base = {f.name: [getattr(cfgs[0], f.name)] for f in fields(cfgs[0])}
    for c in cfgs[1:]:
        for f in fields(c):
            v = getattr(c, f.name)
            if v in base[f.name]: continue
            base[f.name].append(v)
    return base


def make_config_list[T_Cfg: T_TuneConfig](base: T_Cfg, d: dict[str, list[int] | list[str]]) -> list[T_Cfg]:
    """This is the inverse of make_choices; it allows a parameter-configuration search matrix to
    be defined by lists as expected, but represents that as a hopefully-minimal list of configurations
    to test.

    Args:
        base (T_Cfg): A configuration object instance. Whatever configuration it represents WILL NOT
        appear in the actual matrix; it's just used so the dataclass replace function can generate
        the right type of object.

        d (dict): A dictionary with a list of candidate values for each of the hyperparameters
        that would appear in the relevant configuration type. In the event of a field mismatch
        between the dictionary fields and the configuration object type, we expect to see an
        error (rather than strange behavior farther down the stack).

    Returns:
        list[T_Cfg]: A list of tuning configuration objects which collectively represent all of
        the values that appear in the input dictionary-of-lists, so that a proper hyperparameter
        search matrix can be recreated later.
    """
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
    params: Dict[str, Any]
    runtime_ms: float


@dataclass(frozen=True)
class KernelAutotuneOptions:
    mode: AutotuneMode = AutotuneMode.CACHE
    cache_dir: Path | None = None
    cache_dir_key: str | None = None
    verbose: bool = False
