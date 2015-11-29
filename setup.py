import uuid
from pip.req import parse_requirements
from distutils.core import setup

setup(name='beras',
      version='0.0.1',
      description='Extensions of keras',
      author='Leon Sixt',
      author_email='github@leon-sixt.de',
      install_requires=[
            "numpy>=1.9",
            "keras==0.2.0",
            "pytest>=2.7.2",
            "scikit-image>=0.11.3",
            "seya",
            "dotmap>=1.1.2",
            "h5py"
      ],
      packages=['beras',
                'beras.layers']
      )
