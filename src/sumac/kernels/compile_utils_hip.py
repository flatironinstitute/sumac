"""Compatibility facade for the HIP runtime compilation utilities."""

from __future__ import annotations

from ._compile_utils_hip import (
    HipJitError,
    HipKernel,
    compile_hip_kernel,
    hip_device_arch,
)


__all__ = [
    "HipJitError",
    "HipKernel",
    "compile_hip_kernel",
    "hip_device_arch",
]
