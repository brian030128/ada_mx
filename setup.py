from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


setup(
    name='fp8_kernel',
    ext_modules=[
        CUDAExtension('ada_mx', [

            ],
            extra_compile_args={'nvcc': [
                '-gencode=arch=compute_89,code=sm_89',
                '-O3', 
                '-std=c++17',
                '--use_fast_math', 
            ]},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })