# -*- coding: utf-8 -*-
## @package setup
#
#  setup utility package.
#  @author      tody
#  @date        2015/08/14

from setuptools import setup, find_packages
from npr_sfs import __author__, __version__, __license__

setup(
        name = 'som_cm',
        version = __version__,
        description = 'Sample implementation of Inverse Toon Shading [Xu et al. 2015]',
        license = __license__,
        author = __author__,
        url = 'https://github.com/tody411/InverseToon.git',
        packages = find_packages(),
        install_requires = ['docopt'],
        )

