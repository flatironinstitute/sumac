from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Callable


_KernelArguments = list[Any] | tuple[Any, ...]


def _decode(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else "<unknown>"


def _prepend_header(kernel_source: str, header_code: str) -> str:
    if not header_code:
        return kernel_source
    separator = "" if header_code.endswith("\n") else "\n"
    return header_code + separator + kernel_source


def _as_3tuple(value: Any, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (value, 1, 1)
    else:
        items = tuple(value)
        if len(items) == 1:
            result = (int(items[0]), 1, 1)
        elif len(items) == 2:
            result = (int(items[0]), int(items[1]), 1)
        elif len(items) == 3:
            result = (int(items[0]), int(items[1]), int(items[2]))
        else:
            raise ValueError(f"{name} must have 1, 2, or 3 dimensions")

    if any(dimension <= 0 for dimension in result):
        raise ValueError(f"{name} dimensions must be positive, got {result}")
    return result


def _resolve_launch_args(
    launch_args: tuple[Any, ...],
    *,
    grid: Any | None,
    block: Any | None,
    args: _KernelArguments | None,
) -> tuple[Any, Any, _KernelArguments]:
    if launch_args:
        if len(launch_args) > 3:
            raise TypeError("kernel launch accepts at most grid, block, args")
        if len(launch_args) >= 1:
            if grid is not None:
                raise TypeError("kernel launch got multiple values for grid")
            grid = launch_args[0]
        if len(launch_args) >= 2:
            if block is not None:
                raise TypeError("kernel launch got multiple values for block")
            block = launch_args[1]
        if len(launch_args) >= 3:
            if args is not None:
                raise TypeError("kernel launch got multiple values for args")
            args = launch_args[2]

    if grid is None or block is None or args is None:
        raise TypeError("kernel launch requires grid, block, and args")
    if not isinstance(args, (list, tuple)):
        raise TypeError("kernel launch args must be a list or tuple")
    return grid, block, args


@dataclass(slots=True)
class _CStringArray:
    encoded: tuple[bytes, ...]
    pointers: Any

    @property
    def count(self) -> int:
        return len(self.encoded)


def _encode_c_strings(values: tuple[str, ...]) -> _CStringArray:
    encoded = tuple(value.encode("utf-8") for value in values)
    pointers = (ctypes.c_char_p * len(encoded))(*encoded)
    return _CStringArray(encoded=encoded, pointers=pointers)


@dataclass(slots=True)
class _PackedKernelArgs:
    arguments: list[Any]
    values: list[Any]
    pointers: Any


def _pack_kernel_args(
    arguments: list[Any],
    convert: Callable[[Any], Any],
) -> _PackedKernelArgs:
    values = [convert(argument) for argument in arguments]
    pointers = (ctypes.c_void_p * len(values))()
    for index, value in enumerate(values):
        pointers[index] = ctypes.c_void_p(ctypes.addressof(value))
    return _PackedKernelArgs(
        arguments=arguments,
        values=values,
        pointers=pointers,
    )
