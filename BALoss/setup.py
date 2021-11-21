import sys
import numpy as np
import setuptools
from distutils.core import setup
from distutils.extension import Extension

package_name = 'MBD'
module_name  = 'MBD'
version      = sys.version[0]
wrap_source  = './cpp/wrap_py{0:}.cpp'.format(version)
module1      = Extension(module_name,
                    include_dirs = [np.get_include(), './cpp'],
                    sources = ['./cpp/util.cpp', 
                               './cpp/MBD_distance_2d.cpp', 
                               './cpp/MBD_distance.cpp',
                               wrap_source])

# Get the summary
description = 'An open-source toolkit to calculate BAloss'


setup(
    name    = package_name,
    version = "0.1.7",
    author  ='Minh',
    author_email = 'vungocminh@gmail.com',
    description  = description,
    packages = setuptools.find_packages(),
    ext_modules = [module1],
    python_requires = '>=3.6',
)

# to build, run python setup.py build or python setup.py build_ext --inplace
# to install, run python setup.py install