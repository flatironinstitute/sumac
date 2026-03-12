from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
os.environ["TORCH_CUDA_ARCH_LIST"]="8.0;8.9;9.0"
setup(
    name="relu_fused_cuda",
    ext_modules=[
        CUDAExtension(
            name="relu_bat_a_fused_cuda",
            sources=["fused_bat_a.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo", "-std=c++17"
                ],
            },
        ),
        CUDAExtension(
            name="relu_bat_c_fused_cuda",
            sources=["fused_bat_c.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo", "-std=c++17"
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)