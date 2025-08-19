#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    
    Functions to plot the performance scores on a Swiss map
    Rebecca Gugerli, EPFL-MeteoSwiss, March 2023 
"""
#%%

import os, sys
from time import time
from typing import Dict
dir_path = os.path.dirname(os.path.realpath(__file__))

import pandas as pd
import numpy as np
import datetime

from pathlib import Path

from ..common import constants
from ..common.utils import envyaml
from ..common.io_data import read_cart

from .eval_get_estimates import get_QPE_filelist

import logging
logging.getLogger().setLevel(logging.INFO)

# Settings for plots
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import from_levels_and_colors

from PIL import Image

size=12
params = {'legend.fontsize': size,
          'legend.title_fontsize': size,
          'figure.figsize': (8,8),
          'axes.labelsize': size*0.8,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': 20}
plt.rcParams.update(params)


# For scatter-bias plots
def label_axes(ax):
    ax.set_ylabel('Scatter [dB]')
    ax.set_xlabel('Bias [dB]')
    ax.grid(True)
    ax.set_xlim([-3,3])
    ax.set_ylim([1.,4.5])
    return ax

def get_colors_sums():
        lcmap = plt.get_cmap("tab20c")
        bounds = np.arange(500,4600,250)
        norm = colors.BoundaryNorm(bounds, lcmap.N)
        return norm

def get_colors_altitude(cmap="tab20c"):
        lcmap = plt.get_cmap(cmap)
        bounds = np.arange(240,2500,120)
        norm = colors.BoundaryNorm(bounds, lcmap.N)
        return norm

def axes_setup_map(ax):
        x0 = 490000-5000; x1 = 835000+5000 
        y0 = 75000-5000; y1 = 320000+5000
        ax.set_xlim([x0,x1])
        ax.set_ylim([y0,y1])
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(x0, x1, 100000))
        ax.set_yticks(np.arange(y0, y1, 100000))
        return ax

def add_swiss_contours(ax):     
    for shape in constants.BORDER_SHP.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        ax.plot(x,y,'grey',linewidth=1)
    return ax

def add_swiss_contours_qpe_map(ax):     
    for shape in constants.BORDER_SHP.shapeRecords():
        x = [i[0]/1000. for i in shape.shape.points[:]]
        y = [i[1]/1000. for i in shape.shape.points[:]]
        ax.plot(x,y,'k',linewidth=1.2)

        ax.set_ylabel('Swiss (CH1903) S-N coordinate [km]')
        ax.set_xlabel('Swiss (CH1903) W-E cooridnate [km]')

        props = {"rotation" : 90, 
                "verticalalignment": 'center'}
        plt.setp(ax.get_yticklabels(), **props)

    return ax


def add_map_to_plot(ax, y, x, c, scoreDic):

        map = ax.scatter(y,x,c=c,cmap=scoreDic['colormap'],
                        marker='o',label=scoreDic['colorbarlabel'], s=40, 
                        norm=scoreDic['norm'], edgecolor='grey')
        fig = plt.gcf()
        cbar = fig.colorbar(map, ax=ax, orientation='horizontal', extend='both')
        cbar.set_label(scoreDic['colorbarlabel'])

        props = {"rotation" : 90, 
                "verticalalignment": 'center'}
        plt.setp(ax.get_yticklabels(), **props)

        return ax, map

def getFigShapeLocIdx(n_subplots : int):

        if n_subplots == 1:
                figSize=(8,8)
                figShape = (1,1)

        elif n_subplots == 2 :
                figSize=(11.7,8)
                figShape = (1,2)

        elif n_subplots == 3 :
                figSize=(11.7*1.3,8)
                figShape = (1,3)

        elif n_subplots == 4 :
                figSize=(11.7,11.7)
                figShape = (2,2)

        else:
                figSize=(16.5,11.7)
                figShape = (2,3)

        return figSize, figShape

def compileSubplotSetup(modellist, modelfullnames) :
        """

        Gets setup for plotting figures either on a 2x2 graph or a 2x3 graph

        Args:
            modellist (list): Models which to use
            modelnames (dict, optional): Dictionnary which assings a 
                                full name to each model.
                                If none, it uses a given dictionnary Defaults to None.

        Returns:
            dict: Setup for plotting, incl. location, title, model, etc.
        """
        # if modelnames == None:
        #         modelnames = getFullNamesSampleModels()

        if len(modellist) == 1:
                setup={}
                setup[(0,0)] = {'lab': ''}

        elif len(modellist) == 3: 
                setup = {}
                setup[(0,0)] = {'lab' : '(a)'}
                setup[(0,1)] = {'lab' : '(b)'}
                setup[(0,2)] = {'lab' : '(c)'}

        elif len(modellist) == 4 :
                setup = {}
                setup[(0,0)] = {'lab' : '(a)'}
                setup[(0,1)] = {'lab' : '(b)'}
                setup[(1,0)] = {'lab' : '(c)'}
                setup[(1,1)] = {'lab' : '(d)'}

        else :
                setup = {}
                setup[(0,0)] = {'lab' : '(a)'}
                setup[(0,1)] = {'lab' : '(b)'}
                setup[(0,2)] = {'lab' : '(c)'}
                setup[(1,0)] = {'lab' : '(d)'}
                setup[(1,1)] = {'lab' : '(e)'}
                setup[(1,2)] = {'lab' : '(f)'}

        for im, model in enumerate(modellist) :
                setup[list(setup.keys())[im]]['model'] = model
                if model in modelfullnames.keys() :
                        setup[list(setup.keys())[im]]['tit'] = modelfullnames[model]
                else: 
                        setup[list(setup.keys())[im]]['tit'] = model
        return setup

def plotModelMapsSubplots(perfscores, modellist, score='BIAS', config_figures=None, 
                        output_file=True, filename=None):
        """
        The model performances are assembled on a map showing the defined score.

        Args:
            perfscores (DataFrame object): Contains a DataFrame with stations as index, 
                                        and columnnames as perfscores
            modellist (list): list with all models, they need to be within perfscores.keys()
            score (str, optional): Score to evaluate (BIAS, SCATTER, CORR). Defaults to 'BIAS'.
            modelnames (dict, optional): Assigns to each model a name that appears in the 
                                legend title of the figure. Defaults to None.

        Returns:
            _type_: Figure handle to further process or save
        """

        if (config_figures == None) or not os.path.exists(config_figures) :
                logging.info('No figure configuration file is given, using default.')
                dir_path = os.path.dirname(os.path.realpath(__file__))
                config_figures = dir_path+'/config_figures.yml'
        
        setup = envyaml(config_figures)

        # Check that path is given
        if output_file and filename == None:
                logging.error('An output file was wished, but no name is given. Cannot save figure.')
                output_file = False

        # Get full names of models
        figsetup = compileSubplotSetup(modellist, setup['MODELFULLNAMES'])

        # Get score setup
        scoreSetup = setup['SCORESETTING'][score]
        lcmap = plt.get_cmap(scoreSetup['colormap'])
        bounds = np.arange(scoreSetup['bounds'][0], scoreSetup['bounds'][1], scoreSetup['bounds'][2])
        scoreSetup['norm'] = colors.BoundaryNorm(bounds, lcmap.N)
        

        # Check what is used for x- and y-coords
        if 'X-coord' not in perfscores[modellist[0]].columns:
                xcolname = 'X'
                ycolname = 'Y'
        else:
                xcolname = 'X-coord'
                ycolname = 'Y-coord'

        figSize, figShape = getFigShapeLocIdx(len(modellist))
        plt.figure(figsize=figSize)

        for loc in figsetup:
                if loc == (0,0):
                        ax1 = plt.subplot2grid(figShape, loc)
                else: 
                        ax1 =  plt.subplot2grid(figShape, loc, sharey=ax0, sharex=ax0)

                if 'model' not in figsetup[loc].keys():
                        plt.axis('off')
                        continue
                else:
                        model = figsetup[loc]['model']

                ax1 = axes_setup_map(ax1)
                ax1 = add_swiss_contours(ax1)
                leg1 = ax1.legend([],[], framealpha=0, loc='upper left', 
                        title='{} {}'.format(figsetup[loc]['lab'], figsetup[loc]['tit']))
                ax1.add_artist(leg1)

                ax1, _ = add_map_to_plot(ax1,
                                perfscores[model][ycolname],
                                perfscores[model][xcolname],
                                perfscores[model][scoreSetup['colname']],
                                scoreSetup)
                if loc == (0,0):
                        ax0 = ax1
        plt.tight_layout()

        if output_file :
                logging.info('Saving file : {}'.format(filename))
                plt.savefig(filename, transparent=False, dpi=300)
                
        return plt.gcf()

class plotQPEMaps(object):

        def __init__(self, config_file, config_figures=None):
                """ Initiate the class evaluation

                Args:
                        config_file (str): Path to a configuration file
                """
                dir_path = os.path.dirname(__file__)

                try:
                        config = envyaml(config_file)
                        self.configfile = config_file
                except:
                        logging.warning('Using default config as no valid config file was provided')
                        config = envyaml(dir_path + '/default_config.yml')
                        self.configfile = dir_path + '/default_config.yml'

                # Check if setup is ok, and get data where necessary
                self._check_elements(config)

                if (config_figures == None) or not os.path.exists(config_figures) :
                        logging.info('No figure configuration file is given, using default.')
                        dir_path = os.path.dirname(os.path.realpath(__file__))
                        config_figures = dir_path+'/config_figures.yml'
        
                self.setup = envyaml(config_figures)


        def _check_elements(self, config):
                """
                In this function, we check that all necessary elements 
                of the configuration files are there

                Args:
                        config (dic): Dictionnary with the setup for the evaluation
                """

                if ('MAINFOLDER' not in config['PATHS'].keys()) or (not os.path.exists(config['PATHS']['MAINFOLDER'])):
                        logging.error('No existing output folder was defined, please check your config file.')
                        return
                else:
                        self.mainfolder = config['PATHS']['MAINFOLDER']  
                        
                if os.path.exists(config['PATHS']['QPEFOLDER']) :
                        self.qpefolder = config['PATHS']['QPEFOLDER']
                elif os.path.exists(config['PATHS']['MAINFOLDER'] + '/data/') :
                        self.qpefolder = config['PATHS']['MAINFOLDER'] + '/data/'
                else:
                        logging.error('Given mainfolder does not include a subfolder /data/, please check.')
                        return

                if not os.path.exists(config['PATHS']['MAINFOLDER'] + '/qpe_maps/') :
                        os.mkdir(config['PATHS']['MAINFOLDER'] + '/qpe_maps/')
                        logging.info('Creating subfolder /results/ to save the plots and performance scores.')
                self.qpemapfolder = config['PATHS']['MAINFOLDER'] + '/qpe_maps/'

                if ('TIME_START' not in config.keys()):
                        logging.error('No starting time was defined, please check.')
                else:
                        try:
                                self.tstart = datetime.datetime.strptime(config['TIME_START'],
                                        '%Y%m%d%H%M').replace(tzinfo=datetime.timezone.utc)
                        except:
                                logging.error('Starting time is not valid, please check that the format corresponds to YYYYMMDDHHMM.')

                if ('TIME_END' not in config.keys()):
                        logging.error('No ending time was defined, please check.')
                else:
                        try:
                                self.tend = datetime.datetime.strptime(config['TIME_END'],
                                        '%Y%m%d%H%M').replace(tzinfo=datetime.timezone.utc)
                        except:
                                logging.error('Ending time is not valid, please check that the format corresponds to YYYYMMDDHHMM.')

                if ('RF_MODELS' in config.keys()):
                        self.modellist = config['RF_MODELS']
                else:
                        logging.error('Please define a model to evaluate :)')
                        sys.exit()

                if ('REFERENCES' in config.keys()):
                        self.references = list(config['REFERENCES'])
                else:
                        self.references = None

                return

        def get_colors_bounds(self):
                # MeteoSwiss map coloring
                bounds = [0.00, 0.01, 0.16, 0.25, 0.40, 
                        0.63, 1.00, 1.60, 2.50, 4.00,
                        6.30, 10.00, 16.00, 25.00, 40.00,
                        63.00, 100.]  
                colors = ['#FFFFFF', '#C0C0C0', '#660066', '#CC00CC', '#FF33FF',
                        '#001190', '#001CF2', '#066000', '#1AA90F', '#11FF00', 
                        '#ABFF00', '#D1FF00', '#FFFF00', '#FFC000', '#FFA200',
                        '#FF8000', '#FF0000']
                cmap, norm = from_levels_and_colors(bounds,colors,extend='max')
                        
                cmap.set_bad('white')

                collabel = 'precipitation intensity [mm/h]'

                colticklabs = ['0', '0.01', '0.16', '0.25', '0.40', '0.63',
                                '1.0', '1.6','2.5', '4.0', '6.3', '10',
                                '16','25','40', '63', '100']

                return bounds, cmap, norm, collabel, colticklabs

        def plot_qpe(self, ax, pData):
        
                # Get colormaps
                bounds, cmap, norm, collabel, colticklab = self.get_colors_bounds()
                
                # Setup grid
                x = 0.5 * (constants.X_QPE[1:] + constants.X_QPE[0:-1])
                y = 0.5 * (constants.Y_QPE[1:] + constants.Y_QPE[0:-1])
                extent = [np.min(y),np.max(y),
                                np.min(x),np.max(x)]

                # Plot data
                pData = np.ma.masked_invalid(pData)
                im = ax.imshow(pData, norm=norm, cmap=cmap, extent=extent)
                ax.set_aspect('equal', adjustable='box')
                # Add Swiss contours
                ax = add_swiss_contours_qpe_map(ax)
                # ax = axes_setup_map(ax)
                plt.grid(False)

                # # CBAR
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("bottom", size="5%", pad=0.45)
                cb = plt.colorbar(im, cax=cax, orientation='horizontal')
                cb.set_label(collabel)
                cb.ax.tick_params(labelsize=8)

                cb.set_ticks(bounds)
                cb.set_ticklabels(colticklab, rotation=270)

                return ax

        def plotQPEMapsSubplots(self, modellist, time_aggregation='10min', t0=None, t1=None, 
                                output_file=True, filename=None, movie=False,
                                zoom=False, zoom_extent=None,
                                plot_points=False, plot_points_dic=None):
                """

                zoom_extent = list[xlim1, xlim2, ylim1, ylim2] ; CH1903/LV03 coordinates
                plot_points_dic = Dict{'station1': [x,y],
                                        'station2': [x,y]}

                """

                # Check that path is given
                if output_file and filename == None:
                        logging.error('An output file was wished, but no name is given. Cannot save figure.')
                        output_file = False

                # Check if zoom_extent is given when zoom is defined
                if zoom and zoom_extent == None:
                        logging.error('A zoom was defined, but no extent given. Not zooming')
                        zoom = False

                if plot_points and plot_points_dic == None:
                        logging.error('The option plot_points given, but no coordinate dictionary defined')
                        plot_points = False                                

                # Get date-range
                if t0 == None:
                        t0 = self.tstart
                if t1 == None:
                        t1 = self.tend

                if t0.tzinfo == None:
                        t0.replace(tzinfo=datetime.timezone.utc)
                if t1.tzinfo == None:
                        t1.replace(tzinfo=datetime.timezone.utc)


                if time_aggregation  == '5min':
                        time_res= {'time_agg':5,'file_tol':1,'out':'05min'}
                elif time_aggregation == '10min':
                        time_res = {'time_agg':10,'file_tol':2,'out':'10min'}
                elif time_aggregation == '60min':
                        time_res = {'time_agg':60,'file_tol':4,'out':'60min'}
                else:
                        logging.info('Time aggregation is not defined, please check!')


                time_res['tstamps'] = pd.date_range(start=t0, end=t1, 
                                freq=time_aggregation, tz='UTC')

                # Get file list
                qpe_files10_filt = get_QPE_filelist(self.qpefolder, time_res,
                                         modellist)

                # Setup figure
                #---------------
                # Get full names of models
                figsetup = compileSubplotSetup(modellist, self.setup['MODELFULLNAMES'])
                figSize, figShape = getFigShapeLocIdx(len(modellist))

                if movie == True:
                        framenames = []

                # Plotting all realizations for each timestamp
                #----------------------------------------------
                logging.info('All images are saved here: {}'.format(self.qpemapfolder))
                for i, tstep in enumerate(time_res['tstamps'][1::]):

                        plt.figure(figsize=figSize)

                        for im, model in enumerate(modellist):
                                
                                # Initiate figure
                                loc = list(figsetup.keys())[im]

                                if loc == (0,0):
                                        ax1 = plt.subplot2grid(figShape, loc)
                                else: 
                                        ax1 =  plt.subplot2grid(figShape, loc, sharey=ax0, sharex=ax0)

                                # if 'model' not in figsetup[loc].keys():
                                #         plt.axis('off')
                                #         continue
                                # else:
                                #         model = figsetup[loc]['model']

                                # Read model
                                if len(qpe_files10_filt[tstep][model]) == 1:
                                        data = read_cart(qpe_files10_filt[tstep][model][0])
                                elif len(qpe_files10_filt[tstep][model]) > 1:
                                        for f in qpe_files10_filt[tstep][model]:
                                                if f  ==  qpe_files10_filt[tstep][model][0]:
                                                        data = read_cart(f).copy()
                                                else:
                                                        data += read_cart(f).copy()
                                        data = data / len(qpe_files10_filt[tstep][model])
                                else:
                                        logging.info('No timestep for this file')
                                        plt.axis('off')
                                        continue

                                ax1 = self.plot_qpe(ax1, data)
                                lab, tit = figsetup[loc]['lab'], figsetup[loc]['tit']
                                leg1 = ax1.legend([],[], loc='upper left', 
                                        title=f'{lab} {tit}',alignment='center', borderpad=0.1, framealpha=0.8,
                                        labelspacing=0, handleheight=0.)
                                ax1.add_artist(leg1)
                                ax1.set_title(datetime.datetime.strftime(tstep,'%d %b %Y %H:%M UTC'))

                                if zoom :
                                        ax1.set_xlim([zoom_extent[0], zoom_extent[1]])
                                        ax1.set_ylim([zoom_extent[2], zoom_extent[3]])

                                if plot_points :
                                        for station in plot_points_dic.keys():
                                                ax1.plot(plot_points_dic[station][0], plot_points_dic[station][1],
                                                        marker='o', color='black')
                                                ax1.annotate(station, 
                                                        xy=(plot_points_dic[station][0], plot_points_dic[station][1]),
                                                        xytext=(plot_points_dic[station][0], plot_points_dic[station][1]))
                                
                                if loc == (0,0):
                                        ax0 = ax1

                        plt.tight_layout()

                        modellabel = [m + '_' for m in modellist]

                        if zoom :
                                fname = 'QPEmap_zoom_{}{}.png'.format(''.join(modellabel),
                                        datetime.datetime.strftime(tstep,'%Y%m%d%H%M'))
                        else:
                                fname = 'QPEmap_{}{}.png'.format(''.join(modellabel),
                                        datetime.datetime.strftime(tstep,'%Y%m%d%H%M'))

                        logging.info('Saving image to {}'.format(fname))
                        plt.savefig(self.qpemapfolder+fname, bbox_inches='tight')
                        plt.close('all')

                        if movie :
                                framenames.append(fname)


                if movie:
                        frames = []
                        for i in framenames:
                                new_frame = Image.open(self.qpemapfolder+i)
                                frames.append(new_frame)
                        # Save into a GIF file that loops forever
                        timestring = '{}_{}'.format(time_res['tstamps'][0].strftime('%Y%m%d%H'),time_res['tstamps'][-1].strftime('%Y%m%d%H'))
                        if zoom : 
                                fname = 'QPEmap_zoom_{}.gif'.format(timestring)
                        else:
                                fname = 'QPEmap_{}.gif'.format(timestring)
                                
                        frames[0].save(self.qpemapfolder+fname,
                                format='GIF',
                                append_images=frames[1:],
                                save_all=True,
                                duration=300, loop=0)