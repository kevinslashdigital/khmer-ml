#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: "KhmerML" <khmerml@aicambodia.com>
#
from __future__ import division, print_function
from setuptools import setup, find_packages
DOCLINES = (__doc__ or '').split("\n")
import os
import sys
import subprocess

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")

if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU General Public License v3 (GPLv3)
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

DESCRIPTION = """khmerML: opensource module for machine learning"""

LONG_DESCRIPTION = """khmerML is an opensource Python module for machine learning that consists of machine learning algorithms building from scratch focus on solving complex problems in Cambodia society. It has developed by slash research team and it will be able to contribute by anybody that willing to share their research.

khmerML also is a way to encourage Cambodian tech engineers to start learning Machine Learning."""

MAJOR               = 0
MINOR               = 0
MICRO               = 9
ISRELEASED          = True
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

# This is a bit hackish: we are setting a global variable so that the main
# khmerml __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.  While ugly, it's
# a lot more robust than what was previously being used.
builtins.__KHMERML_SETUP__ = True


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of khmerml.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('khmerml/version.py'):
        # must be a source distribution, use existing version file
        try:
            from khmerml.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing " \
                              "khmerml/version.py and the build directory " \
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='khmerml/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM khmerml SETUP.PY
#
# To compare versions robustly, use `khmerml.lib.khmermlVersion`
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()



default_setup = dict(
    name='khmerml',
    version= get_version_info()[0],
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    provides=['khmerml'],
    install_requires=['numpy==1.14.5', 'nltk>=3.2.4'],
    requires=['numpy', 'nltk'],
    packages=find_packages(),
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    author='KhmerML',
    author_email='kevin@slash.co',
)

setup(**default_setup)

# Install nltk data such as corpora, toy grammars, trained models, etc.
# import subprocess
# subprocess.Popen(['sh','nltk_data.sh'])
