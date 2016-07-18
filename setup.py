from pip.req import parse_requirements
from distutils.core import setup

setup(name='diktya',
      version='0.1.0',
      description='Extensions of keras',
      author='Leon Sixt',
      author_email='github@leon-sixt.de',
      install_requires=[
            "numpy>=1.9",
            "keras",
            "pytest>=2.7.2",
            "scikit-image>=0.11.3",
            "seya",
            "dotmap>=1.1.2",
            "h5py"
      ],
      packages=[
          'diktya',
          'diktya.layers',
          'diktya.plot',
          'diktya.numpy',
          'diktya.theano',
      ])
