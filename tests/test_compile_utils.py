import pytest
import torch


IS_ROCM = getattr(torch.version, "hip", None) is not None

if IS_ROCM:
    from sumac.kernels._compile_utils_hip import (
        _compile_hiprtc_image as compile_image_cache,
    )
    from sumac.kernels.compile_utils import HipJitError as JitError
    from sumac.kernels.compile_utils import (
        compile_hip_kernel as compile_kernel,
    )

    RTC_NAME = "HIPRTC"
else:
    from sumac.kernels._compile_utils_cuda import (
        _compile_nvrtc_image as compile_image_cache,
    )
    from sumac.kernels.compile_utils import CudaJitError as JitError
    from sumac.kernels.compile_utils import (
        compile_cuda_kernel as compile_kernel,
    )

    RTC_NAME = "NVRTC"


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="compile-utils integration tests require a CUDA or ROCm accelerator",
)


VECTOR_ADD_SOURCE = r"""
extern "C" __global__ void sumac_test_vector_add(
    const float* a, const float* b, float* out, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}
"""


INVALID_SOURCE = r"""
extern "C" __global__ void sumac_test_compile_error(float* out) {
  out[0] = ;
}
"""


def test_vector_add_and_compile_caches():
    kernel_name = "sumac_test_vector_add"
    device = torch.cuda.current_device()

    kernel = compile_kernel(
        VECTOR_ADD_SOURCE,
        kernel_name=kernel_name,
        device=device,
    )
    cache_after_first_compile = compile_image_cache.cache_info()
    cached_kernel = compile_kernel(
        VECTOR_ADD_SOURCE,
        kernel_name=kernel_name,
        device=device,
    )
    cache_after_second_compile = compile_image_cache.cache_info()

    assert cache_after_second_compile.misses == cache_after_first_compile.misses
    assert cache_after_second_compile.hits == cache_after_first_compile.hits + 1
    assert kernel is not cached_kernel
    assert kernel.image == cached_kernel.image

    n = 257
    block_size = 128
    grid_size = (n + block_size - 1) // block_size
    a = torch.arange(n, device="cuda", dtype=torch.float32)
    b = torch.full((n,), 2.0, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)

    assert not kernel._modules
    assert not kernel._functions

    try:
        kernel(
            grid=(grid_size,),
            block=(block_size,),
            args=[a, b, out, n],
        )
        torch.cuda.synchronize()

        module_handle = kernel._modules[device].value
        function_handle = kernel._functions[device].value
        assert len(kernel._modules) == 1
        assert len(kernel._functions) == 1

        out.fill_(float("nan"))
        kernel(
            grid=(grid_size,),
            block=(block_size,),
            args=[a, b, out, n],
        )
        torch.cuda.synchronize()

        assert kernel._modules[device].value == module_handle
        assert kernel._functions[device].value == function_handle
        assert len(kernel._modules) == 1
        assert len(kernel._functions) == 1
        torch.testing.assert_close(out, a + b, rtol=0.0, atol=0.0)
    finally:
        torch.cuda.synchronize()


def test_syntax_error_includes_rtc_log():
    kernel_name = "sumac_test_compile_error"

    with pytest.raises(JitError) as exc_info:
        compile_kernel(
            INVALID_SOURCE,
            kernel_name=kernel_name,
            device=torch.cuda.current_device(),
        )

    message = str(exc_info.value)
    assert f"{RTC_NAME} failed compiling {kernel_name} for " in message
    assert "\noptions:" in message
    assert "error" in message.lower()
