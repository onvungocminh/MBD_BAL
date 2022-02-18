# to build, run python setup.py build or python setup.py build_ext --inplace
# to install, run python setup.py install

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

exts = [
    Pybind11Extension("BALoss.MBD",
                      ["BALoss/cpp/util.cpp", "BALoss/cpp/MBD_distance_2d.cpp", "BALoss/cpp/module.cpp"],
                      extra_link_args=["-static-libstdc++"]
    )
]

setup(
    name="BALoss",
    version="0.1.0",
    author="Minh",
    author_email="vungocminh@gmail.com",
    description="An open-source toolkit to calculate BAloss",
    packages=["BALoss"],
    package_dir={"BALoss": "BALoss"},
    cmdclass={"build_ext": build_ext},
    ext_modules=exts
)