from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='rabbit',
    ext_modules=[
        CppExtension(
            name='rabbit', 
            sources=[
            'reorder.cpp'
            ],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
            libraries=["numa", "tcmalloc_minimal"]
         ) 
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


