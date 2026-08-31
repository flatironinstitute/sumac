from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Protocol, TypeVar


_KernelArguments = list[Any] | tuple[Any, ...]
_ResultT = TypeVar("_ResultT")


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


def _run_rtc(
    *,
    source: str,
    program_name: str,
    options: tuple[str, ...],
    success_code: int,
    create_program: Callable[..., int],
    compile_program: Callable[..., int],
    get_program_log_size: Callable[..., int],
    get_program_log: Callable[..., int],
    destroy_program: Callable[..., int],
    check: Callable[[int], None],
    make_compile_error: Callable[[str], Exception],
    extract_result: Callable[[ctypes.c_void_p, bytes | None], _ResultT],
    name_expression: str | None = None,
    add_name_expression: Callable[..., int] | None = None,
) -> _ResultT:
    """Compile and extract one runtime-compiler program."""
    source_bytes = source.encode("utf-8")
    program_name_bytes = program_name.encode("utf-8")
    expression_bytes = (
        name_expression.encode("utf-8")
        if name_expression is not None
        else None
    )

    program = ctypes.c_void_p()
    check(
        create_program(
            ctypes.byref(program),
            source_bytes,
            program_name_bytes,
            0,
            None,
            None,
        )
    )

    primary_error: BaseException | None = None
    try:
        if expression_bytes is not None:
            if add_name_expression is None:
                raise RuntimeError(
                    "name_expression requires add_name_expression"
                )
            check(add_name_expression(program, expression_bytes))

        encoded_options = _encode_c_strings(options)
        compile_result = compile_program(
            program,
            encoded_options.count,
            encoded_options.pointers,
        )
        if compile_result != success_code:
            log_size = ctypes.c_size_t()
            check(get_program_log_size(program, ctypes.byref(log_size)))
            if log_size.value == 0:
                log = ""
            else:
                log_buffer = ctypes.create_string_buffer(log_size.value)
                check(get_program_log(program, log_buffer))
                log = _decode(log_buffer.value)
            raise make_compile_error(log)

        return extract_result(program, expression_bytes)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            check(destroy_program(ctypes.byref(program)))
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                "RTC program destruction also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


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


class _JitBackend(Protocol):
    keep_image_alive: bool

    def validate_launch(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_mem: Any,
    ) -> None: ...

    def validate_dynamic_smem(self, shared_mem: Any) -> None: ...

    def prepare_launch(
        self,
        arguments: list[Any],
        *,
        stream: Any | None,
        requested_device: Any | None,
    ) -> int: ...

    def device_guard(self, device: int) -> ContextManager[None]: ...

    def runtime(self) -> Any: ...

    def stream_ptr(self, stream: Any | None, device: int) -> int: ...

    def convert_argument(self, argument: Any) -> Any: ...

    def load_module(
        self,
        runtime: Any,
        module: ctypes.c_void_p,
        image_buffer: Any,
    ) -> None: ...

    def get_function(
        self,
        runtime: Any,
        function: ctypes.c_void_p,
        module: ctypes.c_void_p,
        *,
        image: bytes,
        kernel_name: str,
        symbol_name: str,
    ) -> None: ...

    def set_dynamic_smem(
        self,
        runtime: Any,
        function: ctypes.c_void_p,
        size: int,
    ) -> None: ...

    def unload_module_unchecked(
        self,
        runtime: Any,
        module: ctypes.c_void_p,
    ) -> None: ...

    def launch(
        self,
        runtime: Any,
        function: ctypes.c_void_p,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_mem: int,
        stream_ptr: int,
        parameter_ptrs: Any,
    ) -> None: ...


class JitKernel:
    """Backend-neutral state and orchestration for a runtime-compiled kernel."""

    image: bytes
    kernel_name: str
    _backend: _JitBackend
    _modules: dict[int, ctypes.c_void_p]
    _functions: dict[int, ctypes.c_void_p]
    _image_buffers: dict[int, Any]
    _max_dynamic_smem: int
    _lock: Any

    def _function_for_device(self, device: int) -> ctypes.c_void_p:
        with self._lock:
            if device in self._functions:
                return self._functions[device]

            runtime = self._backend.runtime()
            with self._backend.device_guard(device):
                module = ctypes.c_void_p()
                image_buffer = ctypes.create_string_buffer(self.image)
                self._backend.load_module(runtime, module, image_buffer)

                try:
                    function = ctypes.c_void_p()
                    symbol_name = (
                        getattr(self, "symbol_name", None) or self.kernel_name
                    )
                    self._backend.get_function(
                        runtime,
                        function,
                        module,
                        image=self.image,
                        kernel_name=self.kernel_name,
                        symbol_name=symbol_name,
                    )
                    if self._max_dynamic_smem:
                        self._backend.set_dynamic_smem(
                            runtime,
                            function,
                            int(self._max_dynamic_smem),
                        )
                except BaseException as primary_error:
                    try:
                        self._backend.unload_module_unchecked(runtime, module)
                    except Exception as cleanup_error:
                        primary_error.add_note(
                            "JIT module cleanup also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                    raise

            if self._backend.keep_image_alive:
                self._image_buffers[device] = image_buffer
            self._modules[device] = module
            self._functions[device] = function
            return function

    def set_shared_memory_config(self, shared_mem: int) -> None:
        self._backend.validate_dynamic_smem(shared_mem)
        with self._lock:
            self._max_dynamic_smem = max(
                self._max_dynamic_smem,
                int(shared_mem),
            )
            runtime = self._backend.runtime()
            for device, function in self._functions.items():
                with self._backend.device_guard(device):
                    self._backend.set_dynamic_smem(
                        runtime,
                        function,
                        int(self._max_dynamic_smem),
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
        grid, block, args = _resolve_launch_args(
            launch_args,
            grid=grid,
            block=block,
            args=args,
        )

        grid3 = _as_3tuple(grid, "grid")
        block3 = _as_3tuple(block, "block")
        self._backend.validate_launch(grid3, block3, shared_mem)

        argument_list = list(args)
        launch_device = self._backend.prepare_launch(
            argument_list,
            stream=stream,
            requested_device=device,
        )
        with self._backend.device_guard(launch_device):
            function = self._function_for_device(launch_device)
            stream_pointer = self._backend.stream_ptr(stream, launch_device)
            packed_args = _pack_kernel_args(
                argument_list,
                self._backend.convert_argument,
            )

            runtime = self._backend.runtime()
            self._backend.launch(
                runtime,
                function,
                grid=grid3,
                block=block3,
                shared_mem=int(shared_mem),
                stream_ptr=stream_pointer,
                parameter_ptrs=packed_args.pointers,
            )
