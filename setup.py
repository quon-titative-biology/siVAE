#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from setuptools import find_packages

#### Snip from setup.py for tensorflow-forward-ad
## with workaround for cython/numpy
try:
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []
# include_dirs = [numpy.get_include()]
#
# if use_cython:
#   ext_modules += [
#       Extension(
#           "tensorflow_forward_ad.cbfs", ["tensorflow_forward_ad/cbfs.pyx"],
#           include_dirs=include_dirs),
#   ]
#   cmdclass.update({'build_ext': build_ext})
# else:
#   ext_modules += [
#       Extension(
#           "tensorflow_forward_ad.cbfs", ["tensorflow_forward_ad/cbfs.c"],
#           include_dirs=include_dirs),
#   ]

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)

if use_cython:
  ext_modules += [
      Extension("tensorflow_forward_ad.cbfs", ["tensorflow_forward_ad/cbfs.pyx"])
  ]
else:
  ext_modules += [
      Extension("tensorflow_forward_ad.cbfs", ["tensorflow_forward_ad/cbfs.c"])
  ]

cmdclass.update({'build_ext': CustomBuildExtCommand})

#### Run setup

setup(name='siVAE',
      version='1.0',
      description='scalable and interpretable Variational Autoencoder',
      url='https://github.com/quon-titative-biology/siVAE',
      author=['Yongin Choi', 'Gerald Quon'],
      author_email='yonchoi@ucdavis.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'pandas',
            'matplotlib',
            'scikit-learn',
            'seaborn',
            'tensorflow==1.15',
            "tensorflow-probability==0.8.0",
            'scipy',
            'scikit-image',
            'scanpy',
            'gseapy'],
      extras_requirements = [],
      cmdclass=cmdclass,
      ext_modules=ext_modules
      )
