#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "KhmerML" <khmerml@aicambodia.com>

from setuptools import setup

default_setup = dict(
    name='KhmerML Dependenies',
    description='https://github.com/numpy/numpy/issues/2434',
    provides=['KhmerML Dependenies'],
    install_requires=['numpy==1.14.5', 'nltk>=3.2.4'],
    requires=['numpy', 'nltk'],
    license='GPLv3',
    author='KhmerML',
    author_email='khmerml@aicambodia.com',
)

setup(**default_setup)

# Install nltk data such as corpora, toy grammars, trained models, etc.
import subprocess
subprocess.Popen(['sh','nltk_data.sh'])
