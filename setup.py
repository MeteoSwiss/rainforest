#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation routine
use
python setupy.py install 
to install library
"""

from setuptools import setup
import glob
import os
from os import path
import sys
from pathlib import Path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line for line in requirements_file.read().splitlines() if not line.startswith('#')
    ]

s = setup(name        = "rainforest",
    description = "RandomForest QPE python library",
    version     = "1.1",
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
                           'rf_train = rainforest.ml.rf_train:main']},
        )
