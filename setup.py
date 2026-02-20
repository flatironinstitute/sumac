from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="relu_bat_a_fused_cuda",
    ext_modules=[
        CUDAExtension(
            name="relu_bat_a_fused_cuda",
            sources=["fused_bat_a.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)