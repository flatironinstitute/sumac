from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from ._compile_utils_common import (
    _as_3tuple,
    _decode,
    _pack_kernel_args,
    _prepend_header,
    _resolve_launch_args,
    _run_rtc,
)


HIP_SUCCESS = 0
HIPRTC_SUCCESS = 0
HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE = 8
_get_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


class HipJitError(RuntimeError):
    pass


def _rocm_roots() -> list[Path]:
    roots: list[Path] = []
    for value in (
        os.environ.get("ROCM_HOME"),
        os.environ.get("ROCM_PATH"),
        os.environ.get("HIP_PATH"),
    ):
        if value:
            roots.append(Path(value))

    roots.append(Path("/opt/rocm"))
    return list(dict.fromkeys(roots))


def _torch_library_dir() -> Path | None:
    try:
        return Path(torch.__file__).resolve().parent / "lib"
    except Exception:
        return None


def _library_directories() -> list[Path]:
    directories: list[Path] = []

    torch_lib = _torch_library_dir()
    if torch_lib is not None:
        directories.append(torch_lib)

    for root in _rocm_roots():
        directories.append(root / "lib")
        directories.append(root / "lib64")
    return list(dict.fromkeys(directories))


def _library_candidates(filename: str) -> list[str]:
    return [str(directory / filename) for directory in _library_directories()]


def _best_library_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None

    unversioned = pattern[:-1] if pattern.endswith("*") else pattern
    if unversioned in matches:
        return unversioned

    def version_key(path: str) -> tuple[int, ...]:
        _, separator, version = path.partition(".so.")
        if not separator:
            return ()
        return tuple(
            int(component) if component.isdigit() else -1
            for component in version.split(".")
        )

    return max(matches, key=version_key)


@lru_cache(maxsize=1)
def _paired_hip_libraries() -> tuple[str, str] | None:
    """Prefer HIP runtime and HIPRTC libraries from one ROCm directory."""
    for directory in _library_directories():
        runtime = _best_library_match(str(directory / "libamdhip64.so*"))
        hiprtc = _best_library_match(str(directory / "libhiprtc.so*"))
        if runtime is not None and hiprtc is not None:
            return runtime, hiprtc
    return None


def _find_library(
    candidates: list[str],
    names: list[str],
    *,
    description: str,
) -> str:
    for pattern in candidates:
        match = _best_library_match(pattern)
        if match is not None:
            return match

    for name in names:
        found = ctypes.util.find_library(name)
        if found:
            return found

    raise HipJitError(
        f"Could not find {description}. Set ROCM_HOME/ROCM_PATH or "
        "LD_LIBRARY_PATH to a ROCm installation."
    )


def _torch_hip_library(loader_name: str) -> ctypes.CDLL | None:
    """Return the ROCm library selected by PyTorch, when its helper exists."""
    if getattr(torch.version, "hip", None) is None:
        return None

    try:
        from torch.cuda import _utils as torch_cuda_utils

        loader = getattr(torch_cuda_utils, loader_name, None)
        if loader is None:
            return None
        lib = loader()
    except (ImportError, AttributeError, IndexError, OSError):
        return None

    return lib if isinstance(lib, ctypes.CDLL) else None


@lru_cache(maxsize=1)
def _load_hip_runtime() -> ctypes.CDLL:
    lib = _torch_hip_library("_get_gpu_runtime_library")
    if lib is None:
        pair = _paired_hip_libraries()
        if pair is not None:
            path = pair[0]
        else:
            path = _find_library(
                _library_candidates("libamdhip64.so*"),
                ["amdhip64"],
                description="the HIP runtime library (libamdhip64)",
            )
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    lib.hipInit.argtypes = [ctypes.c_uint]
    lib.hipInit.restype = ctypes.c_int
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    lib.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.hipGetDevice.restype = ctypes.c_int
    lib.hipSetDevice.argtypes = [ctypes.c_int]
    lib.hipSetDevice.restype = ctypes.c_int
    lib.hipModuleLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib.hipModuleLoadData.restype = ctypes.c_int
    lib.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hipModuleGetFunction.restype = ctypes.c_int
    get_function_count = getattr(lib, "hipModuleGetFunctionCount", None)
    if get_function_count is not None:
        get_function_count.argtypes = [
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_void_p,
        ]
        get_function_count.restype = ctypes.c_int
    lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
    lib.hipModuleUnload.restype = ctypes.c_int
    lib.hipModuleLaunchKernel.argtypes = [
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
    lib.hipModuleLaunchKernel.restype = ctypes.c_int
    lib.hipFuncSetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.hipFuncSetAttribute.restype = ctypes.c_int

    _check_hip(lib.hipInit(0), lib)
    return lib


@lru_cache(maxsize=1)
def _load_hiprtc() -> ctypes.CDLL:
    lib = _torch_hip_library("_get_gpu_rtc_library")
    if lib is None:
        pair = _paired_hip_libraries()
        if pair is not None:
            path = pair[1]
        else:
            path = _find_library(
                _library_candidates("libhiprtc.so*"),
                ["hiprtc"],
                description="the HIP runtime compiler library (libhiprtc)",
            )
        lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    lib.hiprtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.hiprtcCreateProgram.restype = ctypes.c_int
    lib.hiprtcAddNameExpression.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hiprtcAddNameExpression.restype = ctypes.c_int
    lib.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.hiprtcDestroyProgram.restype = ctypes.c_int
    lib.hiprtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.hiprtcCompileProgram.restype = ctypes.c_int
    lib.hiprtcGetProgramLogSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.hiprtcGetProgramLogSize.restype = ctypes.c_int
    lib.hiprtcGetProgramLog.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hiprtcGetProgramLog.restype = ctypes.c_int
    lib.hiprtcGetCodeSize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.hiprtcGetCodeSize.restype = ctypes.c_int
    lib.hiprtcGetCode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hiprtcGetCode.restype = ctypes.c_int
    lib.hiprtcGetLoweredName.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.hiprtcGetLoweredName.restype = ctypes.c_int
    lib.hiprtcGetErrorString.argtypes = [ctypes.c_int]
    lib.hiprtcGetErrorString.restype = ctypes.c_char_p
    return lib


def _check_hip(code: int, lib: ctypes.CDLL | None = None) -> None:
    if code == HIP_SUCCESS:
        return
    if lib is None:
        lib = _load_hip_runtime()
    raise HipJitError(f"HIP runtime error {code}: {_decode(lib.hipGetErrorString(code))}")


def _check_hiprtc(code: int, lib: ctypes.CDLL | None = None) -> None:
    if code == HIPRTC_SUCCESS:
        return
    if lib is None:
        lib = _load_hiprtc()
    raise HipJitError(f"HIPRTC error {code}: {_decode(lib.hiprtcGetErrorString(code))}")


def _module_lookup_diagnostics(
    lib: ctypes.CDLL,
    module: ctypes.c_void_p,
    image: bytes,
    kernel_name: str,
    symbol_name: str,
) -> str:
    details = [
        f"HIP runtime library: {getattr(lib, '_name', '<unknown>')}",
        f"HIPRTC library: {getattr(_load_hiprtc(), '_name', '<unknown>')}",
        f"code object: {len(image)} bytes, magic={image[:4].hex()}",
        f"source symbol present in image: {kernel_name.encode('utf-8') in image}",
        f"lowered symbol present in image: {symbol_name.encode('utf-8') in image}",
    ]

    get_function_count = getattr(lib, "hipModuleGetFunctionCount", None)
    if get_function_count is None:
        details.append("module function count: unavailable in this HIP runtime")
    else:
        count = ctypes.c_uint()
        result = int(get_function_count(ctypes.byref(count), module))
        if result == HIP_SUCCESS:
            details.append(f"module function count: {count.value}")
        else:
            details.append(
                "module function count query failed: "
                f"HIP {result} ({_decode(lib.hipGetErrorString(result))})"
            )

    appended_options = os.environ.get("HIPRTC_COMPILE_OPTIONS_APPEND")
    if appended_options:
        details.append(f"HIPRTC_COMPILE_OPTIONS_APPEND is set to {appended_options!r}")
    return "\n".join(details)


def _device_index(device: Any | None = None) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return int(device)

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise HipJitError(f"HIP kernels require a CUDA/HIP device, got {resolved}")
    if resolved.index is None:
        return int(torch.cuda.current_device())
    return int(resolved.index)


def hip_device_arch(device: Any | None = None) -> str:
    """Return the full AMD target ID exposed by a ROCm build of PyTorch."""
    if getattr(torch.version, "hip", None) is None:
        raise HipJitError("HIP JIT compilation requires a ROCm build of PyTorch")

    torch.cuda._lazy_init()
    index = _device_index(device)
    properties = torch.cuda.get_device_properties(index)
    arch = str(getattr(properties, "gcnArchName", "")).strip()
    if not arch:
        raise HipJitError(
            "The active ROCm device does not expose gcnArchName; "
            "pass gpu_arch explicitly to compile_hip_kernel"
        )
    return arch


def _normalize_arch(gpu_arch: Optional[str], device: Any | None = None) -> str:
    if gpu_arch is None:
        return hip_device_arch(device)

    arch = str(gpu_arch).strip()
    for prefix in ("--gpu-architecture=", "--offload-arch="):
        if arch.startswith(prefix):
            arch = arch.removeprefix(prefix)
            break
    if not arch.startswith("gfx") or any(ch.isspace() for ch in arch):
        raise ValueError(f"Expected an AMD GPU target such as 'gfx942', got {gpu_arch!r}")
    return arch


def _has_arch_option(options: list[str]) -> bool:
    for option in options:
        if option in {"--gpu-architecture", "--offload-arch", "--amdgpu-target"}:
            return True
        if option.startswith(
            ("--gpu-architecture=", "--offload-arch=", "--amdgpu-target=")
        ):
            return True
    return False


def _hiprtc_options(
    *,
    arch: str,
    hip_include_dirs: Optional[list[str]],
    hip_options: Optional[list[str]],
) -> list[str]:
    options = list(hip_options or [])
    if _has_arch_option(options):
        raise ValueError("Pass the HIP target through gpu_arch, not hip_options")

    if not any(option.startswith(("--std=", "-std=")) for option in options):
        options.append("--std=c++17")
    if not any(option.startswith("-O") for option in options):
        options.append("-O3")
    options.append(f"--offload-arch={arch}")

    for include_dir in hip_include_dirs or []:
        options.append(f"-I{include_dir}")
    return options


def _extract_hiprtc_result(
    lib: ctypes.CDLL,
    program: ctypes.c_void_p,
    name_expression: bytes | None,
    *,
    kernel_name: str,
    arch: str,
) -> tuple[bytes, str]:
    if name_expression is None:
        raise RuntimeError("HIPRTC extraction requires a name expression")

    lowered_name = ctypes.c_char_p()
    _check_hiprtc(
        lib.hiprtcGetLoweredName(
            program,
            name_expression,
            ctypes.byref(lowered_name),
        ),
        lib,
    )
    if not lowered_name.value:
        raise HipJitError(
            f"HIPRTC returned an empty lowered name for {kernel_name}"
        )
    symbol_name = _decode(lowered_name.value)

    size = ctypes.c_size_t()
    _check_hiprtc(lib.hiprtcGetCodeSize(program, ctypes.byref(size)), lib)
    if size.value == 0:
        raise HipJitError(
            f"HIPRTC produced an empty code object for {kernel_name} ({arch})"
        )
    buffer = ctypes.create_string_buffer(size.value)
    _check_hiprtc(lib.hiprtcGetCode(program, buffer), lib)
    return bytes(buffer.raw), symbol_name


@lru_cache(maxsize=None)
def _compile_hiprtc_image(
    full_source: str,
    *,
    kernel_name: str,
    arch: str,
    options: tuple[str, ...],
) -> tuple[bytes, str]:
    lib = _load_hiprtc()
    return _run_rtc(
        source=full_source,
        program_name=f"{kernel_name}.cpp",
        options=options,
        success_code=HIPRTC_SUCCESS,
        create_program=lib.hiprtcCreateProgram,
        add_name_expression=lib.hiprtcAddNameExpression,
        compile_program=lib.hiprtcCompileProgram,
        get_program_log_size=lib.hiprtcGetProgramLogSize,
        get_program_log=lib.hiprtcGetProgramLog,
        destroy_program=lib.hiprtcDestroyProgram,
        check=lambda code: _check_hiprtc(code, lib),
        make_compile_error=lambda log: HipJitError(
            f"HIPRTC failed compiling {kernel_name} for {arch}\n"
            f"options: {' '.join(options)}\n{log}"
        ),
        extract_result=lambda program, expression: _extract_hiprtc_result(
            lib,
            program,
            expression,
            kernel_name=kernel_name,
            arch=arch,
        ),
        name_expression=kernel_name,
    )


def _kernel_device(args: list[Any]) -> int:
    devices: set[int] = set()
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            continue
        if not arg.is_cuda:
            raise TypeError("HIP kernel tensor arguments must reside on a CUDA/HIP device")
        devices.add(
            int(arg.device.index)
            if arg.device.index is not None
            else int(torch.cuda.current_device())
        )

    if len(devices) > 1:
        raise ValueError(f"HIP kernel arguments span multiple devices: {sorted(devices)}")
    if devices:
        return devices.pop()
    return int(torch.cuda.current_device())


def _stream_device(stream: Any | None) -> int | None:
    if stream is None or isinstance(stream, int):
        return None
    stream_device = getattr(stream, "device", None)
    if stream_device is None:
        return None
    return _device_index(stream_device)


def _launch_device(
    args: list[Any],
    *,
    stream: Any | None,
    requested_device: Any | None,
    default_device: int | None,
) -> int:
    tensor_device = _kernel_device(args) if any(
        isinstance(arg, torch.Tensor) for arg in args
    ) else None
    stream_device = _stream_device(stream)
    explicit_device = (
        _device_index(requested_device) if requested_device is not None else None
    )

    devices = {
        device
        for device in (tensor_device, stream_device, explicit_device)
        if device is not None
    }
    if len(devices) > 1:
        raise ValueError(
            "HIP launch tensors, stream, and explicit device must use the same device; "
            f"got {sorted(devices)}"
        )
    if devices:
        return devices.pop()
    if default_device is not None:
        return default_device
    return int(torch.cuda.current_device())


def _kernel_param(arg: Any) -> ctypes._SimpleCData:
    if isinstance(arg, torch.Tensor):
        return ctypes.c_void_p(arg.data_ptr())
    if arg is None:
        return ctypes.c_void_p(0)
    if isinstance(arg, bool):
        return ctypes.c_bool(arg)
    if isinstance(arg, int):
        if not -(2**31) <= arg < 2**31:
            raise OverflowError(
                "Python integer HIP arguments are int32; pass an explicit ctypes "
                "scalar such as ctypes.c_int64 for another kernel ABI type"
            )
        return ctypes.c_int(arg)
    if isinstance(arg, float):
        return ctypes.c_float(arg)
    if isinstance(arg, ctypes._SimpleCData):
        return arg
    if hasattr(arg, "__index__"):
        return _kernel_param(int(arg))
    raise TypeError(f"Unsupported HIP kernel argument type: {type(arg)!r}")


def _stream_ptr(stream: Any | None, device: int) -> int:
    if stream is None:
        if _get_current_raw_stream is not None:
            return int(_get_current_raw_stream(device))
        return int(torch.cuda.current_stream(device).cuda_stream)
    if isinstance(stream, int):
        stream_pointer = stream
    elif isinstance(stream, ctypes.c_void_p):
        stream_pointer = int(stream.value or 0)
    else:
        raw_stream = getattr(stream, "cuda_stream", None)
        stream_pointer = int(raw_stream) if raw_stream is not None else int(stream)

    if stream_pointer < 0:
        raise ValueError("HIP stream pointers cannot be negative")
    return stream_pointer


@contextmanager
def _hip_device_guard(device: int) -> Iterator[None]:
    lib = _load_hip_runtime()
    previous = ctypes.c_int()
    _check_hip(lib.hipGetDevice(ctypes.byref(previous)), lib)
    changed = previous.value != device
    if changed:
        _check_hip(lib.hipSetDevice(device), lib)
    try:
        yield
    finally:
        if changed:
            _check_hip(lib.hipSetDevice(previous.value), lib)


@dataclass
class HipKernel:
    image: bytes
    kernel_name: str
    gpu_arch: str
    default_device: int | None = None
    symbol_name: str | None = None
    _modules: dict[int, ctypes.c_void_p] = field(default_factory=dict)
    _functions: dict[int, ctypes.c_void_p] = field(default_factory=dict)
    _image_buffers: dict[int, Any] = field(default_factory=dict)
    _max_dynamic_smem: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _function_for_device(self, device: int) -> ctypes.c_void_p:
        with self._lock:
            if device in self._functions:
                return self._functions[device]

            lib = _load_hip_runtime()
            with _hip_device_guard(device):
                module = ctypes.c_void_p()
                image_buf = ctypes.create_string_buffer(self.image)
                _check_hip(lib.hipModuleLoadData(ctypes.byref(module), image_buf), lib)

                try:
                    function = ctypes.c_void_p()
                    symbol_name = self.symbol_name or self.kernel_name
                    try:
                        _check_hip(
                            lib.hipModuleGetFunction(
                                ctypes.byref(function),
                                module,
                                symbol_name.encode("utf-8"),
                            ),
                            lib,
                        )
                    except HipJitError as exc:
                        diagnostics = _module_lookup_diagnostics(
                            lib,
                            module,
                            self.image,
                            self.kernel_name,
                            symbol_name,
                        )
                        raise HipJitError(
                            f"Failed to resolve HIP kernel {self.kernel_name!r} "
                            f"as lowered symbol {symbol_name!r}: {exc}\n"
                            f"{diagnostics}"
                        ) from exc
                    if self._max_dynamic_smem:
                        _check_hip(
                            lib.hipFuncSetAttribute(
                                function,
                                HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                                int(self._max_dynamic_smem),
                            ),
                            lib,
                        )
                except Exception:
                    # Preserve the original lookup/attribute error. Unloading is
                    # best effort on this partial-construction path.
                    lib.hipModuleUnload(module)
                    raise

            self._image_buffers[device] = image_buf
            self._modules[device] = module
            self._functions[device] = function
            return function

    def close(self) -> None:
        """Unload owned modules after callers have finished all kernel work.

        The caller must ensure this does not race another launch and that any
        outstanding work using these modules is safe to unload.
        """
        with self._lock:
            lib = _load_hip_runtime()
            failures: list[Exception] = []
            for device, module in list(self._modules.items()):
                try:
                    with _hip_device_guard(device):
                        _check_hip(lib.hipModuleUnload(module), lib)
                except Exception as exc:
                    failures.append(exc)
                    continue

                self._modules.pop(device, None)
                self._functions.pop(device, None)
                self._image_buffers.pop(device, None)

            if failures:
                raise HipJitError(
                    f"Failed to unload {len(failures)} HIP module(s): {failures[0]}"
                ) from failures[0]

    def set_shared_memory_config(self, shared_mem: int) -> None:
        if not 0 <= shared_mem < 2**31:
            raise ValueError("shared_mem must fit in a non-negative C int")
        with self._lock:
            self._max_dynamic_smem = max(self._max_dynamic_smem, int(shared_mem))
            lib = _load_hip_runtime()
            for device, function in self._functions.items():
                with _hip_device_guard(device):
                    _check_hip(
                        lib.hipFuncSetAttribute(
                            function,
                            HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                            int(self._max_dynamic_smem),
                        ),
                        lib,
                    )

    def __call__(
        self,
        *launch_args: Any,
        grid: Any | None = None,
        block: Any | None = None,
        args: list[Any] | tuple[Any, ...] | None = None,
        shared_mem: int = 0,
        stream: Any | None = None,
        device: Any | None = None,
    ) -> None:
        """Launch the kernel.

        Plain Python integers and floats map to 32-bit C ``int`` and ``float``
        arguments. Pass an explicit ``ctypes`` scalar for any other kernel ABI
        type. Raw integer stream handles have no device metadata; pair them with
        tensor arguments, a compile-time default device, or ``device=``.
        """
        grid, block, args = _resolve_launch_args(
            launch_args,
            grid=grid,
            block=block,
            args=args,
        )

        grid3 = _as_3tuple(grid, "grid")
        block3 = _as_3tuple(block, "block")
        uint32_limit = 2**32
        for axis, (grid_dimension, block_dimension) in enumerate(
            zip(grid3, block3, strict=True)
        ):
            if grid_dimension >= uint32_limit or block_dimension >= uint32_limit:
                raise ValueError("HIP grid and block dimensions must fit in uint32")
            if grid_dimension * block_dimension >= uint32_limit:
                raise ValueError(
                    "HIP gridDim * blockDim must be less than 2**32 on each axis; "
                    f"axis {axis} is {grid_dimension} * {block_dimension}"
                )
        if not 0 <= shared_mem < uint32_limit:
            raise ValueError("shared_mem must fit in uint32")

        arg_list = list(args)
        torch.cuda._lazy_init()
        launch_device = _launch_device(
            arg_list,
            stream=stream,
            requested_device=device,
            default_device=self.default_device,
        )
        with _hip_device_guard(launch_device):
            function = self._function_for_device(launch_device)
            stream_ptr = _stream_ptr(stream, launch_device)
            packed_args = _pack_kernel_args(arg_list, _kernel_param)

            lib = _load_hip_runtime()
            _check_hip(
                lib.hipModuleLaunchKernel(
                    function,
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
                ),
                lib,
            )


def compile_hip_kernel(
    kernel_source: str,
    *,
    kernel_name: str,
    header_code: str = "",
    gpu_arch: Optional[str] = None,
    device: Any | None = None,
    hip_include_dirs: Optional[list[str]] = None,
    hip_options: Optional[list[str]] = None,
) -> HipKernel:
    kernel_source = _prepend_header(kernel_source, header_code)

    default_device = None
    if device is not None or gpu_arch is None:
        default_device = _device_index(device)

    arch = _normalize_arch(gpu_arch, default_device)
    options = tuple(
        _hiprtc_options(
            arch=arch,
            hip_include_dirs=hip_include_dirs,
            hip_options=hip_options,
        )
    )
    image, symbol_name = _compile_hiprtc_image(
        kernel_source,
        kernel_name=kernel_name,
        arch=arch,
        options=options,
    )
    return HipKernel(
        image=image,
        kernel_name=kernel_name,
        gpu_arch=arch,
        default_device=default_device,
        symbol_name=symbol_name,
    )


__all__ = [
    "HipJitError",
    "HipKernel",
    "compile_hip_kernel",
    "hip_device_arch",
]
