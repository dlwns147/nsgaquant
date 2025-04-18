from setuptools import setup, Extension
from torch.utils import cpp_extension

extra_compile_args = {
    "cxx": [
        "-g", 
        "-O3", 
        "-fopenmp", 
        "-lgomp", 
        "-std=c++17",
        "-DENABLE_BF16"
    ],
    "nvcc": [
        "-O3", 
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8"
    ],
}

setup(
    name='faster_transformer',
    ext_modules=[cpp_extension.CUDAExtension(
        name = 'faster_transformer', 
        sources = ['FT.cpp', 
                #    'dequantize.cu', 
                   "layernorm/layernorm.cu",
                   "attention/ft_attention.cpp",
                   "attention/decoder_masked_multihead_attention.cu",
                ],
        extra_compile_args=extra_compile_args,
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires = ["torch"],
)
