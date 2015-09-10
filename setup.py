import uuid
from pip.req import parse_requirements
from distutils.core import setup

setup(name='beras',
      version='0.0.1',
      description='Extensions of keras',
      author='Leon Sixt',
      author_email='github@leon-sixt.de',
      install_requires=[
            "numpy>=1.9,<1.10",
            "keras>=0.1.3",
            "pytest>=2.7.2",
            "seya"
      ],
      dependency_links=[
        "git+https://github.com/EderSantana/seya.git@53a6eb6c0f5e5c7036f95ca631a3adcb00b57dc8#egg=seya"
      ],
      packages=['beras',
                'beras.layers']
      )
