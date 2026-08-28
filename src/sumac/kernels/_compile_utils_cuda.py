from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch

from ._compile_utils_common import (
    _as_3tuple,
    _decode,
    _pack_kernel_args,
    _prepend_header,
    _resolve_launch_args,
    _run_rtc,
)


CUDA_SUCCESS = 0
NVRTC_SUCCESS = 0
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
_get_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


class CudaJitError(RuntimeError):
    pass


def _find_library(candidates: list[str], names: list[str]) -> str:
    for name in names:
        found = ctypes.util.find_library(name)
        if found:
            return found

    for pattern in candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]

    raise CudaJitError(
        "Could not find " + " or ".join(names) +
        ". Set LD_LIBRARY_PATH or install the CUDA runtime wheels/toolkit."
    )


@lru_cache(maxsize=1)
def _load_driver() -> ctypes.CDLL:
    lib = ctypes.CDLL(_find_library([], ["cuda"]))

    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    lib.cuGetErrorString.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.cuGetErrorString.restype = ctypes.c_int
    lib.cuModuleLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib.cuModuleLoadData.restype = ctypes.c_int
    lib.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.cuModuleGetFunction.restype = ctypes.c_int
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cuLaunchKernel.restype = ctypes.c_int
    lib.cuFuncSetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.cuFuncSetAttribute.restype = ctypes.c_int

    _check_cuda(lib.cuInit(0))
    return lib


def _check_cuda(code: int) -> None:
    if code == CUDA_SUCCESS:
        return

    msg = ctypes.c_char_p()
    lib = _load_driver()
    if lib.cuGetErrorString(code, ctypes.byref(msg)) == CUDA_SUCCESS:
        raise CudaJitError(f"CUDA driver error {code}: {_decode(msg.value)}")
    raise CudaJitError(f"CUDA driver error {code}")


def _nvrtc_candidates() -> list[str]:
    patterns: list[str] = []
    roots = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
    ]
    for root in roots:
        if root:
            patterns.append(str(Path(root) / "lib64" / "libnvrtc.so*"))
            patterns.append(str(Path(root) / "lib" / "libnvrtc.so*"))

    for entry in sys.path:
        patterns.append(str(Path(entry) / "nvidia" / "cu*" / "lib" / "libnvrtc.so*"))
        patterns.append(
            str(Path(entry) / "nvidia" / "cuda_nvrtc" / "lib" / "libnvrtc.so*")
        )

    try:
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        patterns.append(str(torch_lib / "libnvrtc.so*"))
    except Exception:
        pass

    return patterns


@lru_cache(maxsize=1)
def _load_nvrtc() -> ctypes.CDLL:
    path = _find_library(_nvrtc_candidates(), ["nvrtc"])
    try:
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        builtins = sorted(glob.glob(str(Path(path).with_name("libnvrtc-builtins.so*"))))
        if builtins:
            ctypes.CDLL(builtins[-1], mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    lib.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nvrtcCreateProgram.restype = ctypes.c_int
    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.nvrtcDestroyProgram.restype = ctypes.c_int
    lib.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nvrtcCompileProgram.restype = ctypes.c_int
    lib.nvrtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.nvrtcGetProgramLogSize.restype = ctypes.c_int
    lib.nvrtcGetProgramLog.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.nvrtcGetProgramLog.restype = ctypes.c_int
    lib.nvrtcGetPTXSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.nvrtcGetPTXSize.restype = ctypes.c_int
    lib.nvrtcGetPTX.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.nvrtcGetPTX.restype = ctypes.c_int
    lib.nvrtcGetErrorString.argtypes = [ctypes.c_int]
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p

    if hasattr(lib, "nvrtcGetCUBINSize"):
        lib.nvrtcGetCUBINSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.nvrtcGetCUBINSize.restype = ctypes.c_int
        lib.nvrtcGetCUBIN.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.nvrtcGetCUBIN.restype = ctypes.c_int

    return lib


def _check_nvrtc(code: int, lib: ctypes.CDLL | None = None) -> None:
    if code == NVRTC_SUCCESS:
        return
    if lib is None:
        lib = _load_nvrtc()
    raise CudaJitError(f"NVRTC error {code}: {_decode(lib.nvrtcGetErrorString(code))}")


def _normalize_arch(compute_capability: Optional[str]) -> str:
    if compute_capability is None:
        torch.cuda._lazy_init()
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        return f"sm_{major}{minor}"

    cc = str(compute_capability).strip().lower()
    if cc.startswith(("sm_", "compute_")):
        return cc
    if cc.startswith(("sm", "compute")):
        prefix = "sm_" if cc.startswith("sm") else "compute_"
        return prefix + cc.removeprefix("sm").removeprefix("compute")
    return "sm_" + cc.replace(".", "")


def _has_arch_option(options: list[str]) -> bool:
    for opt in options:
        if opt in {"-arch", "--gpu-architecture"}:
            return True
        if opt.startswith(("-arch=", "--gpu-architecture=")):
            return True
    return False


def _nvrtc_options(
    *,
    arch: str,
    cuda_include_dirs: Optional[list[str]],
    nvcc_options: Optional[list[str]],
) -> list[str]:
    options = list(nvcc_options or [])

    if not any(opt.startswith("--std=") or opt.startswith("-std=") for opt in options):
        options.append("--std=c++17")
    if not _has_arch_option(options):
        options.append(f"--gpu-architecture={arch}")

    for inc in cuda_include_dirs or []:
        options.append(f"-I{inc}")

    return options


def _extract_nvrtc_image(
    lib: ctypes.CDLL,
    program: ctypes.c_void_p,
    *,
    arch: str,
) -> bytes:
    if hasattr(lib, "nvrtcGetCUBINSize") and arch.startswith("sm_"):
        size = ctypes.c_size_t()
        _check_nvrtc(lib.nvrtcGetCUBINSize(program, ctypes.byref(size)), lib)
        if size.value:
            buffer = ctypes.create_string_buffer(size.value)
            _check_nvrtc(lib.nvrtcGetCUBIN(program, buffer), lib)
            return bytes(buffer.raw)

    size = ctypes.c_size_t()
    _check_nvrtc(lib.nvrtcGetPTXSize(program, ctypes.byref(size)), lib)
    buffer = ctypes.create_string_buffer(size.value)
    _check_nvrtc(lib.nvrtcGetPTX(program, buffer), lib)
    return bytes(buffer.raw)


@lru_cache(maxsize=None)
def _compile_nvrtc_image(
    full_source: str,
    *,
    kernel_name: str,
    arch: str,
    options: tuple[str, ...],
) -> bytes:
    lib = _load_nvrtc()
    return _run_rtc(
        source=full_source,
        program_name=f"{kernel_name}.cu",
        options=options,
        success_code=NVRTC_SUCCESS,
        create_program=lib.nvrtcCreateProgram,
        compile_program=lib.nvrtcCompileProgram,
        get_program_log_size=lib.nvrtcGetProgramLogSize,
        get_program_log=lib.nvrtcGetProgramLog,
        destroy_program=lib.nvrtcDestroyProgram,
        check=lambda code: _check_nvrtc(code, lib),
        make_compile_error=lambda log: CudaJitError(
            f"NVRTC failed compiling {kernel_name} for {arch}\n"
            f"options: {' '.join(options)}\n{log}"
        ),
        extract_result=lambda program, _: _extract_nvrtc_image(
            lib,
            program,
            arch=arch,
        ),
    )


def _kernel_device(args: list[Any]) -> int:
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            return arg.device.index if arg.device.index is not None else torch.cuda.current_device()
    return torch.cuda.current_device()


def _kernel_param(arg: Any) -> ctypes._SimpleCData:
    if isinstance(arg, torch.Tensor):
        return ctypes.c_void_p(arg.data_ptr())
    if arg is None:
        return ctypes.c_void_p(0)
    if isinstance(arg, bool):
        return ctypes.c_bool(arg)
    if isinstance(arg, int):
        if -(2**31) <= arg < 2**31:
            return ctypes.c_int(arg)
        return ctypes.c_longlong(arg)
    if isinstance(arg, float):
        return ctypes.c_float(arg)
    if isinstance(arg, ctypes._SimpleCData):
        return arg
    if hasattr(arg, "__index__"):
        return _kernel_param(int(arg))
    raise TypeError(f"Unsupported CUDA kernel argument type: {type(arg)!r}")


def _stream_ptr(stream: Any | None, device: int) -> int:
    if stream is None:
        if _get_current_raw_stream is not None:
            return int(_get_current_raw_stream(device))
        return int(torch.cuda.current_stream(device).cuda_stream)

    if isinstance(stream, int):
        return stream

    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is not None:
        return int(cuda_stream)

    return int(stream)


@dataclass
class CudaKernel:
    image: bytes
    kernel_name: str
    _modules: dict[int, ctypes.c_void_p] = field(default_factory=dict)
    _functions: dict[int, ctypes.c_void_p] = field(default_factory=dict)
    _max_dynamic_smem: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _function_for_device(self, device: int) -> ctypes.c_void_p:
        with self._lock:
            if device in self._functions:
                return self._functions[device]

            lib = _load_driver()
            module = ctypes.c_void_p()
            image_buf = ctypes.create_string_buffer(self.image)
            _check_cuda(lib.cuModuleLoadData(ctypes.byref(module), image_buf))

            func = ctypes.c_void_p()
            _check_cuda(
                lib.cuModuleGetFunction(
                    ctypes.byref(func),
                    module,
                    self.kernel_name.encode("utf-8"),
                )
            )
            if self._max_dynamic_smem:
                _check_cuda(
                    lib.cuFuncSetAttribute(
                        func,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        int(self._max_dynamic_smem),
                    )
                )

            self._modules[device] = module
            self._functions[device] = func
            return func

    def set_shared_memory_config(self, shared_mem: int) -> None:
        self._max_dynamic_smem = max(self._max_dynamic_smem, int(shared_mem))
        lib = _load_driver()
        for func in self._functions.values():
            _check_cuda(
                lib.cuFuncSetAttribute(
                    func,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    int(self._max_dynamic_smem),
                )
            )

    def __call__(
        self,
        *launch_args: Any,
        grid: Any | None = None,
        block: Any | None = None,
        args: list[Any] | tuple[Any, ...] | None = None,
        shared_mem: int = 0,
        stream: Any | None = None,
    ) -> None:
        grid, block, args = _resolve_launch_args(
            launch_args,
            grid=grid,
            block=block,
            args=args,
        )

        # torch.cuda.nvtx.range_push("kernel launch overhead")
        grid3 = _as_3tuple(grid, "grid")
        block3 = _as_3tuple(block, "block")
        arg_list = list(args)
        device = _kernel_device(arg_list)

        torch.cuda._lazy_init()
        func = self._function_for_device(device)
        stream_ptr = _stream_ptr(stream, device)

        packed_args = _pack_kernel_args(arg_list, _kernel_param)

        lib = _load_driver()
        _check_cuda(
            lib.cuLaunchKernel(
                func,
                grid3[0],
                grid3[1],
                grid3[2],
                block3[0],
                block3[1],
                block3[2],
                int(shared_mem),
                ctypes.c_void_p(stream_ptr),
                packed_args.pointers,
                None,
            )
        )
        # torch.cuda.nvtx.range_pop()


def compile_cuda_kernel(
    kernel_source: str,
    *,
    kernel_name: str,
    header_code: str = "",
    compute_capability: Optional[str] = None,
    cuda_include_dirs: Optional[list[str]] = None,
    nvcc_options: Optional[list[str]] = None,
) -> CudaKernel:
    kernel_source = _prepend_header(kernel_source, header_code)

    arch = _normalize_arch(compute_capability)
    options = tuple(
        _nvrtc_options(
            arch=arch,
            cuda_include_dirs=cuda_include_dirs,
            nvcc_options=nvcc_options,
        )
    )
    image = _compile_nvrtc_image(
        kernel_source,
        kernel_name=kernel_name,
        arch=arch,
        options=options,
    )
    return CudaKernel(image=image, kernel_name=kernel_name)
