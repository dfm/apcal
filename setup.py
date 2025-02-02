#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
import numpy.distutils.misc_util

if True:
    extra_compile_args = []
    extra_link_args = []
else:
    extra_compile_args=['-fopenmp']
    extra_link_args=['-lgomp']

likelihood = Extension('apcal._likelihood',
                ['apcal/_likelihood.c'],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args)

setup(packages=['apcal'],
        ext_modules = [likelihood],
        include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
        )

