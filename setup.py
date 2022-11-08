import os
from setuptools import setup, find_packages
from setuptools import Extension
from distutils.command.build import build as build_orig

with open("requirements.txt", 'r') as f:
    requirements = f.read()
    
exts = [
    Extension(
        name='spi3n.mc.ising.cy',
        sources=["spi3n/mc/ising/cy.pyx"],
        include_dirs=["spi3n/mc/ising"]),
    Extension(
        name='spi3n.mc.bw.cy',
        sources=["spi3n/mc/bw/cy.pyx"],
        include_dirs=["spi3n/mc/bw"]),
]

class build(build_orig):
    def finalize_options(self):
        super().finalize_options()
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())
        from Cython.Build import cythonize
        self.distribution.ext_modules = cythonize(
            self.distribution.ext_modules,
            language_level=3
        )


setup(
    name='spi3n',
    version='0.9',
    description='Spin NN module',
    author='Satankov',
    author_email='satankow@yandex.ru',
    packages=find_packages(),
    install_requires=requirements.split('\n'),
    ext_modules=exts,
    setup_requires=["cython", "numpy"],
    zip_safe=False,
    cmdclass={"build": build},
)
