from setuptools import setup, find_packages
import versioneer
from Cython.Build import cythonize
import numpy as np


# ---- C/C++ EXTENSIONS ---- #
cython_modules = ['menpofit/dpm/utils.pyx']

cython_exts = cythonize(cython_modules, quiet=True)
include_dirs = [np.get_include()]


setup(
     name='menpofit',
     version=versioneer.get_version(),
     cmdclass=versioneer.get_cmdclass(),
     description="Menpo's image feature point localisation (AAMs, SDMs, "
                 "CLMs, ERT, ASMs, APS, DPM)",
     author='The Menpo Development Team',
     author_email='james.booth08@imperial.ac.uk',
     include_dirs=include_dirs,
     ext_modules=cython_exts,
     packages=find_packages(),
     install_requires=['menpo>=0.6,<0.7',
                       'scikit-learn>=0.17,<0.18',
                       'numpy>=1.10,<1.11',
                       'Cython>=0.23,<0.24'],
     package_data={'menpofit': ['dpm/cpp/*.h'],
                   '': ['*.pxd', '*.pyx']},
     tests_require=['nose', 'mock']
)
