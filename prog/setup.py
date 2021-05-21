from distutils.core import setup
from Cython.Build import cythonize
import numpy as np


setup(
    ext_modules=cythonize('src/model.pyx'),
    include_dirs=[np.get_include()]
)
#export CFLAGS="-I /usr/local/lib/python3.9/site-packages/numpy/core/include $CFLAGS"
#python3 setup.py build_ext --inplace
#CC=/usr/local/opt/llvm/bin/clang++ python setup.py build_ext --inplace