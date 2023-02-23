#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation routine
use
python setupy.py install 
to install library
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import Cython
import glob
import os
from os import path
import sys
from pathlib import Path
from numpy import get_include

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line for line in requirements_file.read().splitlines() if not line.startswith('#')
    ]

extension_addat = Extension(
    'rainforest.common.add_at', sources=['rainforest/common/add_at.pyx'],
    include_dirs = [get_include()])


s = setup(name        = "rainforest_mch",
    description = "RandomForest QPE python library",
    long_description = "A library to compute a gauge/radar database, train a RF QPE predictor and apply it to generate QPE fields",
    version     = "1.3.2",
    author = 'Rebecca Gugerli, Daniel Wolfensberger',
    author_email = ' rebecca.gugerli@epfl.ch, daniel.wolfensberger@meteoswiss.ch',
    license = 'GPL-3.0',
    packages = ['rainforest',
                'rainforest/common',
                'rainforest/qpe',
                'rainforest/ml',
                'rainforest/database'],
    include_package_data = True,
    install_requires=requirements,
    zip_safe=False,
    entry_points = {
        'console_scripts':['rainforest_interface =  rainforest.interface:main',
                           'qpe_compute = rainforest.qpe.qpe_compute:main',
                           'qpe_plot = rainforest.qpe.qpe_plot:main',
                           'qpe_evaluation = rainforest.qpe.qpe_evaluation:main',
                           'db_populate = rainforest.database.db_populate:main',
                           'rf_train = rainforest.ml.rf_train:main',
                           'interact_cloud = rainforest.common.interact_cloud:main']},
    ext_modules = cythonize(extension_addat)
        )
