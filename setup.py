#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation routine
use
python setupy.py install 
to install library
"""
from setuptools import setup
import boto3
import glob
import os
import sys
from pathlib import Path

def download_additional_data(installation_path, linode_obj_config):
    bpath_data = Path(installation_path, 'common')
    bpath_rfmodels = Path(installation_path, 'ml')
    path_data = Path(bpath_data, 'data')
    path_rfmodels = Path(bpath_rfmodels, 'rfmodels')
    
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    if not os.path.exists(path_rfmodels):
        os.makedirs(path_rfmodels)

    client = boto3.client("s3", **linode_obj_config)

    response = client.list_objects(Bucket='rainforest')
    for i, object in enumerate(response['Contents']):
        print("{:d}/{:d} files".format(i+1, len(response['Contents'])))
        key = object['Key']
        dirname = os.path.dirname(key)
        if 'rfmodels' in key:
            bpath = bpath_rfmodels
            if not os.path.exists(Path(bpath_rfmodels, dirname)):
                os.makedirs(Path(bpath_rfmodels, dirname))
        else:
            if not os.path.exists(Path(bpath_data, dirname)):
                os.makedirs(Path(bpath_data, dirname))
            bpath = bpath_data
        client.download_file(
            Bucket='rainforest',
            Key=object['Key'],
            Filename = str(Path(bpath, object['Key'])))


if __name__ == '__main__':

    s = setup(name        = "rainforest",
        description = "RandomForest QPE python library",
        version     = "1.0",
        author = 'Rebecca Gugerli, Daniel Wolfensberger',
        author_email = ' rebecca.gugerli@epfl.ch, daniel.wolfensberger@meteoswiss.ch',
        license = 'GPL-3.0',
        packages = ['rainforest',
                  'rainforest/common',
                  'rainforest/qpe',
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
 
    # Get setup.py installation dir (this is a mess...)
    bdir_install = s.command_obj['install'].install_lib
    install_dir = sorted(glob.glob(str(Path(bdir_install, 'rainforest*'))), 
         key=os.path.getmtime)[-1]
    rainforest_path = Path(install_dir, 'rainforest')
    sys.path.append('./rainforest/')
    from common.constants import linode_obj_config
    download_additional_data(rainforest_path, linode_obj_config)
