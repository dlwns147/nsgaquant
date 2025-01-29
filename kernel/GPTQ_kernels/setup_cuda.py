from setuptools import setup, Extension
from torch.utils import cpp_extension

extra_compile_args = {
    "cxx": [
        "-g", 
        "-O3", 
        "-fopenmp", 
        "-lgomp", 
        "-std=c++17",
    ],
    "nvcc": [
        "-O3", 
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8"
    ],
}

setup(
    name='custom_gptq',
    ext_modules=[cpp_extension.CUDAExtension(
        name = 'custom_gptq._C', 
        sources = ['custom_gptq_cuda_kernel.cu'],
        extra_compile_args=extra_compile_args,
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)