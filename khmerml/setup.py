import os
import warnings
from os.path import join
from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
  from numpy.distutils.misc_util import Configuration
  #from numpy.distutils.system_info import get_info, BlasNotFoundError
  #import numpy
  libraries = []
  if os.name == 'posix':
      libraries.append('m')
  config = Configuration('khmerml', parent_package, top_path)
  # submodules which do not have their own setup.py
  # we must manually add sub-submodules & tests
  # config.add_subpackage('examples')
  config.add_subpackage('algorithms')
  config.add_subpackage('utils')
  # config.add_subpackage('naive_bayes')
  config.add_subpackage('preprocessing')
  return config

if __name__ == '__main__':
  setup(**configuration(top_path='').todict())
