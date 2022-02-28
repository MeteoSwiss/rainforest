#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line script for the RandomForest QPE

see :ref:`qpe_compute` 
"""

# Global imports
import json
import os 
import datetime
from pathlib import Path
from optparse import OptionParser

# Local imports
from rainforest.qpe.qpe import QPEProcessor
from rainforest.ml.rfdefinitions import read_rf

# Suppress certain warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    parser = OptionParser()
    
    parser.add_option("-s", "--start", dest = "start", type = str,
                      help="Specify the start time in the format YYYYddmmHHMM",
                      metavar = "START")
    
    parser.add_option("-e", "--end", dest = "end", type = str,
                      help="Specify the end time in the format YYYYddmmHHMM",
                      metavar = "END")
    
    parser.add_option("-o", "--output", dest = "outputfolder", type = str,
                      help="Path of the output folder, default is current folder",  default = './',
                      metavar="OUTPUT")
    
    parser.add_option("-c", "--config", dest = "config", type = str,
                      default = None, help="Path of the config file, the default will be default_config.yml in the qpe module", 
                      metavar="CONFIG")
    
    parser.add_option("-m", "--models", dest = "models", type = str,
                      default = '{"RF_dualpol":"RF_dualpol_BETA_-0.5_BC_spline.p"}',
                      help='Specify which models you want to use in the form of a json line' +
                      ', the models must be in the folder /ml/rf_models/, for example \'{"RF_dualpol":"RF_dualpol_BETA_-0.5_BC_spline.p}\'' +
                      ', please note the double and single quotes, which are required',
                      metavar="MODELS")
    
    parser.add_option("-p", "--modelpath", dest = "modelpath", type=str,
                      default = None,
                      help ='Specify where the models are stored in case they are not saved under /ml/rf_models/',
                      metavar='MODELPATHS')
    
    
    (options, args) = parser.parse_args()
    
    if options.config == None:
        script_path = os.path.dirname(os.path.realpath(__file__)) 
        options.config = str(Path(script_path, 'default_config.yml'))
        
    options.models = json.loads(options.models)
    for k in options.models.keys():
        if options.modelpath == None:
            options.models[k] = read_rf(options.models[k])
        else:
            options.models[k] = read_rf(options.models[k], filepath=options.modelpath)
    
    if not os.path.exists(options.outputfolder):
        os.makedirs(options.outputfolder)
        
    options.start = datetime.datetime.strptime(options.start, '%Y%m%d%H%M')
    options.end = datetime.datetime.strptime(options.end, '%Y%m%d%H%M')
    print(options.config, options.models)
    print(options.start, options.end)    
    qpe = QPEProcessor(options.config, options.models)
    qpe.compute(options.outputfolder, options.start, options.end)
