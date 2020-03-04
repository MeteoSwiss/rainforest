#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line script to add new data to the database

see :doc:`db_cmd`
"""

# Global imports
import sys
import os 
import datetime
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from optparse import OptionParser

# Local imports
from rainforest.database.database import Database
from rainforest.common import constants

def main(): 
    parser = OptionParser()
    
    parser.add_option("-t", "--type", dest = "type", type = str,
                      help="Type of table to populate, either 'gauge', 'reference' or 'radar'", 
                      metavar="TYPE")
    
    parser.add_option("-o", "--outputfolder", dest = "outputfolder", type = str,
                      default = None,
                      help="Path of the output folder, default is /store/msrad/radar/radar_database/<type>", 
                      metavar="OUTPUT")
    
    parser.add_option("-s", "--start", dest = "start", type = str,
                      help="Specify the start time in the format YYYYddmmHHMM, it is mandatory only if type == 'gauge', otherwise if not provided, will be inferred from gauge data",
                      metavar = "START", default = None)
    
    parser.add_option("-e", "--end", dest = "end", type = str,
                      help="Specify the end time in the format YYYYddmmHHMM, it is mandatory only if type == 'gauge', otherwise if not provided, will be inferred from gauge data",
                      metavar = "END", default = None)
    
    parser.add_option("-c", "--config", dest = "config", type = str,
                      default = None, help="Path of the config file, the default will be default_config.yml in the database module", 
                      metavar="CONFIG")
    
    parser.add_option("-g", "--gauge", dest = "gauge", type = str,
                      default = '/store/msrad/radar/radar_database/gauge/*.csv.gz',
                      help="Needed only if type == reference or radar, path pattern (with wildcards) of the gauge data (from database) to be used, " +
                          "default = '/store/msrad/radar/radar_database/gauge/*.csv.gz', IMPORTANT you have to put this statement into quotes (due to wildcard)!")
   
    (options, args) = parser.parse_args()
    
    if options.type not in ['gauge','radar','reference']:
        raise ValueError("Type (-t) must be either 'radar', 'gauge' or 'reference'")
    if options.type == 'gauge' and (options.end == None or options.start == None):
        raise ValueError("Please enter start and time when type == 'gauge'")
        
        
    if options.start != None:
        options.start = datetime.datetime.strptime(options.start, '%Y%m%d%H%M')
    if options.end != None:
        options.end = datetime.datetime.strptime(options.end, '%Y%m%d%H%M')  
   
    if options.outputfolder == None:
        options.outputfolder = str(Path(constants.FOLDER_DATABASE, options.type))
        
    if not os.path.exists(options.outputfolder):
        os.makedirs(options.outputfolder)
        
        
    if options.config == None:
        script_path = os.path.dirname(os.path.realpath(__file__)) 
        default_config_path = str(Path(script_path, 'default_config.yml'))
        options.config = default_config_path
        
    dbase = Database(config_file = options.config)
    
    if options.type != 'gauge':
        logging.info('Trying to read gauge table...')
        try:
            dbase.add_tables({'gauge':options.gauge})
        except:
            logging.error('Could not read gauge table, please check -g pattern')
            raise
    
    logging.info('Starting database update, leave the script running!')
    if options.type == 'gauge':
        dbase.update_station_data(options.start, options.end, 
                                  options.outputfolder)
    elif options.type == 'reference':
        dbase.update_reference_data('gauge', options.outputfolder,
                                    options.start, options.end)
    elif options.type  == 'radar':
        dbase.update_radar_data('gauge', options.outputfolder,
                                    options.start, options.end)
    
