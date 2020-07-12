##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import io
import os
import subprocess
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)

cwd = os.path.dirname(os.path.abspath(__file__))

version = '1.2.1'
try:
    if not os.getenv('RELEASE'):
        from datetime import date
        today = date.today()
        day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'cvss', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is cvss version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

def make_cuda_ext(name, module, sources, sources_cuda=[]):

    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'portalocker',
    'torch>=1.4.0',
    'torchvision>=0.5.0',
    'Pillow',
    'scipy',
    'requests',
]

if __name__ == '__main__':
    create_version_file()
    setup(
        name="cvss",
        version=version,
        author="Zuoyu Zhou",
        author_email="zhuozuoyu0305@gmail.com",
        url="https://github.com/zhanghang1989/PyTorch-Encoding",
        description="Computer vision, Semantic segmentation",
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        license='MIT',
        install_requires=requirements,
        packages=find_packages(exclude=["tests", "experiments"]),
        package_data={ 'cvss.ops': ['*/*.so']},
        ext_modules=[
            make_cuda_ext(
                name='deform_conv_ext',
                module='cvss.ops.dcn',
                sources=['src/deform_conv_ext.cpp'],
                sources_cuda=[
                    'src/cuda/deform_conv_cuda.cpp',
                    'src/cuda/deform_conv_cuda_kernel.cu'
                ]),
            CUDAExtension(
                name='cvss.ops.encoding.encoding_ext', 
                sources=[
                    'cvss/ops/encoding/src/operator.cpp',
                    'cvss/ops/encoding/src/activation_kernel.cu',
                    'cvss/ops/encoding/src/encoding_kernel.cu',
                    'cvss/ops/encoding/src/syncbn_kernel.cu',
                    'cvss/ops/encoding/src/rectify_cuda.cu'], 
                define_macros=[('WITH_CUDA', None)],
                extra_compile_args={'cxx': [], 'nvcc': ['--expt-extended-lambda']})
        ],
        cmdclass={'build_ext': BuildExtension}
    )
