"""Public entry points for CUDA and HIP runtime kernel compilation."""

from __future__ import annotations

from ._compile_utils_cuda import CudaJitError, CudaKernel, compile_cuda_kernel
from ._compile_utils_hip import (
    HipJitError,
    HipKernel,
    compile_hip_kernel,
    hip_device_arch,
)


__all__ = [
    "CudaJitError",
    "CudaKernel",
    "HipJitError",
    "HipKernel",
    "compile_cuda_kernel",
    "compile_hip_kernel",
    "hip_device_arch",
]
