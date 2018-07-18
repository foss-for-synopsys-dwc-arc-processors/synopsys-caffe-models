# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()
    

ext_modules = [
    Extension("nms.cpu_nms", ["nms/cpu_nms.pyx"], include_dirs = [numpy_include]),
]

setup(
    name='fast_rcnn',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
