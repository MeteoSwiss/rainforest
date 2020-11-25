#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line script to display QPE realisazions

see :ref:`qpe_plot` 
"""

# Global imports
import sys
import os 
import datetime
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from optparse import OptionParser

# Local imports
from rainforest.common.graphics import qpe_plot
from rainforest.common.io_data import read_cart
from rainforest.common.utils import get_qpe_files

def main(): 
    parser = OptionParser()
    
    parser.add_option("-i", "--inputfolder", dest = "inputfolder", type = str,
                      help="Path of the input folder", 
                      metavar="INPUT")
    
    parser.add_option("-o", "--outputfolder", dest = "outputfolder", type = str,
                      help="Path of the output folder", 
                      metavar="OUTPUT")
    
    parser.add_option("-s", "--start", dest = "start", type = str,
                      help="Specify the start time in the format YYYYddmmHHMM, optional: " + 
                      "if not provided entire timerange in input folder will be plotted",
                      metavar = "START", default = None)
    
    parser.add_option("-e", "--end", dest = "end", type = str,
                      help="Specify the end time in the format YYYYddmmHHMM, optional: " + 
                      "if not provided entire timerange in input folder will be plotted",
                      metavar = "END", default = None)
    
    parser.add_option("-S", "--shapefile", dest = "shapefile", type = int,
                      help="Whether or not to overlay the shapefile of swiss borders, default is True", 
                      metavar="SHAPEFILE", default = 1)
    
    parser.add_option("-f", "--figsize", dest = "figsize", type = str,
                      help="Size of figure width,height in inches, e.g. 5,10, default is automatically chosen depending on how many QPE fields are to be plotted", 
                      metavar="FIGSIZE", default = None)

    parser.add_option("-x", "--xlim", dest = "xlim", type = str,
                      help="Limits in the west-east direction, in Swiss coordinates, e.g. 100,300, default is 400,900", 
                      metavar="XLIM", default = '400,900')
    
    parser.add_option("-c", "--cbar", dest = "cbar", type = str,
                    help="Orientation of the colorbar, either 'vertical' or 'horizontal', default is 'horizontal'", 
                      metavar="CBAR", default = 'vertical')
    
    parser.add_option("-y", "--ylim", dest = "ylim", type = str,
                    help="Limits in the south-north direction, in Swiss coordinates, e.g. 500,700, default is 0,350", 
                      metavar="YLIM", default = '0,350')
    
    
    parser.add_option("-d", "--display", dest = "display", type = str,
                      help="Specify how you want to display the QPE subplots as a comma separated string, e.g 2,1 will put them on 2 rows, one column. Default is to put them in one row", 
                      metavar="DISPLAY", default = None)
    
    parser.add_option("-t", "--transition", dest = "transition", type = float,
                      help="Precipitation intensity at which the colormap changes (purple for lower intensities, rainbow for larger intensities), default is 10", 
                      metavar="TRANSITION", default = 3)    

    parser.add_option("-v", "--vmin", dest = "vmin", type = float,
                      help="Minimum precip. intensity to display, default = 0.04", 
                      metavar="VMIN", default = 0.04)
    
    parser.add_option("-V", "--vmax", dest = "vmax", type = float,
                      help="Maximum precip. intensity to display, default = 120", 
                      metavar="VMAX", default = 120)

    parser.add_option("-m", "--models", dest = "models", type = str,
                      help="Specify which models (i.e. subfolders in the qpefolder you want to use, default is to use all available, must be comma separated and put into quotes, e.g. 'dualpol,hpol,RZC'",
                      metavar = "MODELS", default = None)     
    
                          
    (options, args) = parser.parse_args()
    
    if options.start != None:
        options.start = datetime.datetime.strptime(options.start, '%Y%m%d%H%M')
    if options.end != None:
        options.end = datetime.datetime.strptime(options.end, '%Y%m%d%H%M')  
    if options.figsize != None:
        options.figsize = options.figsize.split(',')
        options.figsize = [float(v) for v in options.figsize]
    if options.xlim != None:
        options.xlim = options.xlim.split(',')
        options.xlim = [float(v) for v in options.xlim]
    if options.ylim != None:
        options.ylim = options.ylim.split(',')
        options.ylim = [float(v) for v in options.ylim]
    if options.display != None:
        options.display = options.display.split(',')
        options.display = [int(v) for v in options.display]
    if not os.path.exists(options.outputfolder):
        os.makedirs(options.outputfolder)
    if options.models != None:
        options.models = options.models.split(',')
        options.models = [m.strip() for m in options.models]
            
    all_files = get_qpe_files(options.inputfolder, options.start, options.end,
                              list_models = options.models)
    
    for t in sorted(all_files.keys()): # Loop on timesteps
        logging.info('Processing timestep : '+str(t) )
        labels = list(all_files[t].keys())
        fields = []
        for fname in all_files[t].values():
            fields.append(read_cart(fname[0]))
        
        fig,ax = qpe_plot(fields,
                               figsize = options.figsize,
                               transition = options.transition,
                               vmax = options.vmax,
                               vmin = options.vmin,
                               subplots = options.display,
                               cbar_orientation = options.cbar,
                               xlim = options.xlim,
                               ylim = options.ylim)
        for i in range(len(labels)):
            ax[i].set_title(labels[i])
           
        fig.suptitle(str(t), y = max([1 - 0.05 * len(fields), 0.9]),
                     fontsize = 14, fontweight="bold")
        tstr = datetime.datetime.strftime(t, '%Y%j%H%M')
        plt.savefig(options.outputfolder + '/qpeplot' + tstr.zfill(3)+'.png',
                    bbox_inches = 'tight',
                    dpi = 300)
        plt.close('all')
    
     
    