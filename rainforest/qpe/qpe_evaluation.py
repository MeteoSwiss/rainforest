#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line script to evaluate a set of QPE realizations using gauge data
as reference

see :ref:`qpe_evaluation` 
"""

import os 
import datetime

from optparse import OptionParser
from rainforest.qpe.evaluation import evaluation

def main():
    parser = OptionParser()
    
    parser.add_option("-q", "--qpefolder", dest = "qpefolder", type = str,
                      help="Path of the folder where QPE data is stored", 
                      metavar="QPEFOLDER")
    
    parser.add_option("-g", "--gaugepattern", dest = "gaugepattern", type = str,
                      help="Path pattern (with wildcards) of the gauge data (from database) to be used, " +
                          "default = '/store/msrad/radar/radar_database/gauge/*.csv.gz', IMPORTANT you have to put this statement into quotes (due to wildcard)!", 
                      metavar="GAUGEFOLDER",
                      default = "/store/msrad/radar/radar_database/gauge/*.csv.gz")
     
    parser.add_option("-o", "--output", dest = "outputfolder", type = str,
                      help="Path of the output folder", 
                      metavar="OUTPUT")
    
    parser.add_option("-s", "--start", dest = "start", type = str,
                      help="Specify the start time in the format YYYYddmmHHMM, optional: " + 
                      "if not provided entire timerange in qpe folder will be used",
                      metavar = "START", default = None)
    
    parser.add_option("-e", "--end", dest = "end", type = str,
                      help="Specify the end time in the format YYYYddmmHHMM, optional: " + 
                      "if not provided entire timerange in qpe folder will be used",         
                      metavar = "END", default = None)
    
    parser.add_option("-b", "--b10", dest = "b10", type = str,
                      help="Specify which precipitation ranges you want to use at the 10 min resolution, " + \
                      " as a comma separated string, e.g. 0,1,10,200 will separate the results, in the ranges [0,1),[1,10),[10,200(",
                      metavar = "b10", default = '0,2,10,200')
    
    parser.add_option("-B", "--b60", dest = "b60", type = str,
                      help="Specify which precipitation ranges you want to use at the 60 min resolution, " + \
                      " as a comma separated string, e.g. 0,1,10,200 will separate the results, in the ranges [0,2),[2,10),[10,200(",
                      metavar = "END", default = '0,2,10,200')        

    parser.add_option("-m", "--models", dest = "models", type = str,
                      help="Specify which models (i.e. subfolders in the qpefolder you want to use, default is to use all available, must be comma separated and put into quotes, e.g. 'dualpol,hpol,RZC'",
                      metavar = "MODELS", default = None)     
    
    (options, args) = parser.parse_args()

    options.b10 = [int(u) for u in options.b10.split(',')]
    options.b60 = [int(u) for u in options.b60.split(',')]
    
    if not os.path.exists(options.outputfolder):
        os.makedirs(options.outputfolder)
        
    if options.models != None:
        options.models = options.models.split(',')
        options.models = [m.strip() for m in options.models]
    
    if options.outputfolder[-1] != '/':
        options.outputfolder += '/'
    
    if options.start != None:
        options.start = datetime.datetime.strptime(options.start, '%Y%m%d%H%M')
    if options.end != None:
        options.end = datetime.datetime.strptime(options.end, '%Y%m%d%H%M')
        
    evaluation(options.qpefolder, options.gaugepattern, list_models = options.models,
               outputfolder = options.outputfolder, t0 = options.start,
               t1 = options.end, bounds10 = options.b10, 
               bounds60 = options.b60)
    
