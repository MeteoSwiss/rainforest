#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line script to prepare input features and train RF models

see :ref:`rf_train`
"""

# Global imports
import sys
import os 
import datetime
import logging
import json
from pathlib import Path
logging.basicConfig(level=logging.INFO)
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)
from optparse import OptionParser

# Local imports
from rainforest.ml.rf import RFTraining
from rainforest.common import constants

def main(): 
    parser = OptionParser()
    

    parser.add_option("-o", "--outputfolder", dest = "outputfolder", type = str,
                      default = None,
                      help="Path of the output folder, default is the ml/rf_models folder in the current library", 
                      metavar="OUTPUT")

    parser.add_option("-d", "--dbfolder", dest = "dbfolder", type = str,
                      default = constants.FOLDER_DATABASE,
                      help="Path of the database main folder, default is {:s}".format(constants.FOLDER_DATABASE), 
                      metavar="DBFOLDER")
    
    parser.add_option("-i", "--inputfolder", dest = "inputfolder", type = str,
                      default = None,
                      help="Path where the homogeneized input files for the RF algorithm are stored, default is the subfolder 'rf_input_data' within the database folder", 
                      metavar="DBFOLDER")

    parser.add_option("-s", "--start", dest = "start", type = str,
                      help="Specify the start time in the format YYYYddmmHHMM, if not provided the first timestamp in the database will be used",
                      metavar = "START", default = None)
    
    parser.add_option("-e", "--end", dest = "end", type = str,
                      help="Specify the end time in the format YYYYddmmHHMM, if not provided the last timestamp in the database will be used",
                      metavar = "END", default = None)

    parser.add_option("-c", "--config", dest = "config", type = str,
                      default = None, help="Path of the config file, the default will be default_config.yml in the database module", 
                      metavar="CONFIG")
    
    parser.add_option("-m", "--models", dest = "models", type = str,
                      default = None,
                      help='Specify which models you want to use in the form of a json line of a dict, the keys are names you give to the models, the values the input features they require' +
                      ', for example \'{"RF_dualpol": ["RADAR", "zh_VISIB_mean", "zv_VISIB_mean","KDP_mean","RHOHV_mean","T", "HEIGHT","VISIB_mean"]}\'' +
                      ', please note the double and single quotes, which are required' + 
                      'IMPORTANT: if no model is provided only the ml input data will be recomputed from the database, but no model will be computed'+
                      'To simplify three aliases are proposed: ' +
                      '"dualpol_default" = \'{"RF_dualpol": ["RADAR", "zh_VISIB_mean", "zv_VISIB_mean","KDP_mean","RHOHV_mean","SW_mean", "T", "HEIGHT","VISIB_mean"]}\'' +
                      '"vpol_default" = \'{"RF_vpol": ["RADAR", "zv_VISIB_mean","SW_mean", "T", "HEIGHT","VISIB_mean"]}\'' +
                      '"hpol_default" = \'{"RF_hpol": ["RADAR", "zh_VISIB_mean","SW_mean", "T", "HEIGHT","VISIB_mean"]}\'' +
                      'You can combine them for example "vpol_default, hpol_default, dualpol_default, will compute all three"',
                      metavar="MODELS")

    parser.add_option("-g", "--generate_inputs", dest = "generate_inputs", 
                      type = int,
                      default = 1,
                      help="If set to 1 (default), the input parquet files (homogeneized tables) for the ml routines will be recomputed from the current database rows"+
                      "This takes a bit of time but is needed if you updated the database and want to use the new data in the training", 
                      metavar="MODELS")
    
    (options, args) = parser.parse_args()
    

    if options.start != None:
        options.start = datetime.datetime.strptime(options.start, '%Y%m%d%H%M')
    if options.end != None:
        options.end = datetime.datetime.strptime(options.end, '%Y%m%d%H%M')  
   
    if options.outputfolder == None:
        options.outputfolder = str(Path(script_path, 'rf_models' ))
                
    if not os.path.exists(options.outputfolder):
        os.makedirs(options.outputfolder)
        
    logging.info('Output folder will be {:s}'.format(options.outputfolder))
    
    if options.inputfolder == None:
        options.inputfolder = str(Path(options.dbfolder, 'rf_input_data'))

    logging.info('Folder with RF input data will be {:s}'.format(options.inputfolder))
    
    dic_models = {}
    
    only_regenerate = False # Whether or not to only generate new input data (no training)
    if options.models == None:
        only_regenerate = True
    else:
        if 'default' in options.models:
            logging.info('Found at least one "default" alias in models {:s}, assuming they are all aliases'.format(options.models))
            options.models = options.models.split(',')
            for opt in options.models: # Add aliases here if needed
                opt = opt.strip()
                if opt == 'dualpol_default':
                    dic_models['RF_dualpol'] =  ["RADAR", "zh_VISIB_mean",
                                                 "zv_VISIB_mean","KDP_mean",
                                                 "RHOHV_mean","SW_mean","T", 
                                                 "HEIGHT","VISIB_mean"]
                elif opt == 'hpol_default':
                    dic_models['RF_hpol'] =  ["RADAR", "zh_VISIB_mean","SW_mean","T",
                                              "HEIGHT","VISIB_mean"]
                elif opt == 'vpol_default':
                    dic_models['RF_hpol'] =  ["RADAR", "zv_VISIB_mean","SW_mean","T",
                                              "HEIGHT","VISIB_mean"]
        else:
            dic_models = json.loads(options.models)
            
    options.models = dic_models
    
    if only_regenerate:
        logging.info('No model was given, only the input parquet files will be regenerated')
        

    if options.config == None:
        default_config_path = str(Path(script_path, 'default_config.yml'))
        options.config = default_config_path
    
    logging.info('Starting randomForest training, leave the script running!')
    
    input_location = str(Path(options.dbfolder, 'rf_input_data'))
    logging.info('Assuming that input data for RF training is in folder {:s}'.format(input_location))
    logging.info('If not available in this folder, they will be computed and stored there')
    
    rf = RFTraining(options.dbfolder, options.inputfolder,
                    options.generate_inputs)
    
    if not only_regenerate:
        rf.fit_models(options.config, options.models, options.start,
                      options.end)
    
