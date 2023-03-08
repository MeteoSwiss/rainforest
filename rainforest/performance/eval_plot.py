#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Rebecca Gugerli, EPFL-MeteoSwiss, March 2022
    
    Script to plot the performance scores
    To calculate these, please run the scirpt 03_perfscores_calc.py first
    
"""
#%%
import sys, os

import pandas as pd
import numpy as np

from pathlib import Path

from ..common import constants
from ..common.utils import envyaml

import logging
logging.getLogger().setLevel(logging.INFO)

# Settings for plots
import matplotlib.pyplot as plt
from matplotlib import colors

size=12
params = {'legend.fontsize': size,
          'legend.title_fontsize': size,
          'figure.figsize': (8,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.8,
          'ytick.labelsize': size*0.8,
          'axes.titlepad': 25}
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

        if len(modellist) <= 4 :
                setup = {}
                setup[(0,0)] = {'lab' : '(a)'}
                setup[(0,1)] = {'lab' : '(b)'}
                setup[(1,0)] = {'lab' : '(c)'}
                setup[(1,1)] = {'lab' : '(d)'}
                for im, model in enumerate(modellist) :
                        setup[list(setup.keys())[im]]['model'] = model
                        if model in modelfullnames.keys() :
                                setup[list(setup.keys())[im]]['tit'] = modelfullnames[model]
                        else: 
                                setup[list(setup.keys())[im]]['tit'] = model

        if len(modellist) > 4 :
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

        # Set figure sizes
        if len(modellist) <= 4 :
                plt.figure(figsize=(11.7,11.7))
                figShape = (2,2)
        else:
                plt.figure(figsize=(16.5,11.7))
                figShape = (2,3)

        for loc in figsetup:
                if loc == (0,0):
                        ax1 = plt.subplot2grid(figShape, loc)
                else: 
                        ax1 =  plt.subplot2grid(figShape, loc, sharey=ax0, sharex=ax0)

                if 'model' not in figsetup[loc].keys():
                        continue
                else:
                        model = figsetup[loc]['model']

                ax1 = axes_setup_map(ax1)
                ax1 = add_swiss_contours(ax1)
                leg1 = ax1.legend([],[], framealpha=0, loc='upper left', 
                        title='{} {}'.format(figsetup[loc]['lab'], figsetup[loc]['tit']))
                ax1.add_artist(leg1)

                ax1, map1 = add_map_to_plot(ax1,
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