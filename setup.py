#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation routine
use
python setupy.py install 
to install library
"""
from setuptools import setup

setup(name        = "rainforest",
        description = "QPE python library by wod",
        version     = "2.1",
        author = 'Daniel Wolfensberger - LTE EPFL/MeteoSwiss',
        author_email = 'daniel.wolfensberger@epfl.ch',
        license = 'GPL-3.0',
        packages = ['rainforest',
                  'rainforest/qpe',
                  'rainforest/common',
                  'rainforest/ml',
                  'rainforest/database'],
        include_package_data = True,
        install_requires=['numpy'],
        zip_safe=False,
        entry_points = {
            'console_scripts':['rainforest_interface =  rainforest.interface:main',
                               'qpe_compute = rainforest.qpe.qpe_compute:main',
                               'qpe_plot = rainforest.qpe.qpe_plot:main',
                               'qpe_evaluation = rainforest.qpe.qpe_evaluation:main',
                               'db_populate = rainforest.database.db_populate:main',
                               'rf_train = rainforest.ml.rf_train:main']}
        )

