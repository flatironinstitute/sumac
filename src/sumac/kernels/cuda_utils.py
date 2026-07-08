from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def cuda_is_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, AssertionError, RuntimeError):
        return False


def cuda_device_count() -> int:
    if not cuda_is_available():
        return 0
    try:
        return int(torch.cuda.device_count())
    except (AttributeError, AssertionError, RuntimeError):
        return 0


def current_cuda_device() -> int:
    if not cuda_is_available():
        return 0
    return int(torch.cuda.current_device())


def is_cuda_device(device) -> bool:
    if device is None:
        return cuda_is_available()
    return torch.device(device).type == "cuda"


def synchronize_if_cuda(device=None) -> None:
    if not cuda_is_available():
        return
    if device is not None and not is_cuda_device(device):
        return
    torch.cuda.synchronize(device)


def empty_cache_if_cuda() -> None:
    if cuda_is_available():
        torch.cuda.empty_cache()


def nvtx_range_push(message: str) -> None:
    if not cuda_is_available():
        return
    try:
        torch.cuda.nvtx.range_push(message)
    except RuntimeError:
        pass


def nvtx_range_pop() -> None:
    if not cuda_is_available():
        return
    try:
        torch.cuda.nvtx.range_pop()
    except RuntimeError:
        pass


@contextmanager
def nvtx_range(message: str) -> Iterator[None]:
    pushed = False
    if cuda_is_available():
        try:
            torch.cuda.nvtx.range_push(message)
            pushed = True
        except RuntimeError:
            pushed = False

    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except RuntimeError:
                pass
