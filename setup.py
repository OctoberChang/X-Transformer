#!/usr/bin/env python

import sys, os
from os import path
from shutil import copytree, rmtree, ignore_patterns

try :
    from setuptools import setup, Extension
except :
    from distutils.core import setup, Extension

# a technique to build a shared library on windows
from distutils.command.build_ext import build_ext
build_ext.get_export_symbols = lambda x,y: []

def get_blas_link_args(blas='lapack_opt'):
    import numpy.distutils.system_info as info
    dirs = info.get_info(blas)['library_dirs']
    libs = info.get_info(blas)['libraries']
    libs_cmd = ['-l{}'.format(x) for x in libs]
    dirs_cmd = ['-L{}'.format(x) for x in dirs]
    rpath_cmd = ['-Wl,-rpath,{}'.format(':'.join(dirs))]
    blas_link_args = ['-fopenmp', '-Wl,--as-needed'] + rpath_cmd + libs_cmd + dirs_cmd + ['-liomp5']
    if sys.platform.lower() == 'darwin':
        blas_link_args = rpath_cmd + ['-framework Accelerate', '-liomp5']
    return blas_link_args

source_codes = ["xbert/corelib/rf_linear.cpp"]
headers = ["xbert/corelib/rf_matrix.h"]
include_dirs = ["trmf/corelib"]
libname = "xbert.corelib.rf_linear"
blas_link_args = get_blas_link_args()

if sys.platform == "win32":
    print('Not supported in Windows')
    sys.exit(-1)
    dynamic_lib = Extension('liblinear.liblinear_dynamic', source_codes,
            depends=headers,
            include_dirs=["src/"],
            define_macros=[("_WIN64",""), ("_CRT_SECURE_NO_DEPRECATE","")],
            language="c++",
            extra_link_args=["-DEF:src\linear.def"])
else :
    dynamic_lib_float32 = Extension('{}_float32'.format(libname),
                                    source_codes,
                                    depends=headers,
                                    include_dirs=include_dirs,
                                    define_macros=[("ValueType","float")],
                                    extra_compile_args=["-fopenmp", "-march=native", "-O3", "-std=c++11"],
                                    extra_link_args=blas_link_args,
                                    language="c++")

    dynamic_lib_float64 = Extension('{}_float64'.format(libname),
                                    source_codes,
                                    depends=headers,
                                    include_dirs=include_dirs,
                                    define_macros=[("ValueType","double")],
                                    extra_compile_args=["-fopenmp", "-march=native", "-O3", "-std=c++11"],
                                    extra_link_args=blas_link_args,
                                    language="c++")
setup(
    name='xbert',
    packages=["xbert"],
    version='0.1',
    description='Experimental Codes for X-BERT paper',
    author='Wei-Cheng Chang',
    author_email='peter78789@gmail.com',
    ext_modules=[dynamic_lib_float32, dynamic_lib_float64],
    package_data={"xbert":["corelib/*.cpp", "corelib/*.h"]},
    setup_requires=["mkl", "scipy", "numpy"],
    install_requires=["mkl", "scipy", "numpy"]
)
