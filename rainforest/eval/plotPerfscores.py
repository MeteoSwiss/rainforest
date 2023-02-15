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
from rainforest.common import constants
from rainforest.common.utils import envyaml

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


# ACTUAL SCRIPT
#------------------------------------------------
if __name__ == '__main__':
        #-----------------------------------------------------------
        # Paths
        #-----------------------------------------------------------
        # pathIn = '/scratch/rgugerli/analysis/eval_bias_aug2022/analyses/'
        # timeperiod = '202208040000_202208072350'
        # modellist =  ['RFQ','RFO_standard','RFO_no_disag','RFO_no_pp','RZC', 'CPCH', 'RFO_no_OR', 'RFO_no_Gauss']
        
        timeperiod = '202110312000_202111012000'
        pathIn = '/scratch/rgugerli/analysis/eval_bias_nov2021/results/'
        pathOut = '/scratch/rgugerli/analysis/eval_bias_nov2021/results/plots_final/'
        
        modellist = ['RFQ','RFQ_v1','RFO','RFO_2016_19','RZC', 'CPCH', 'RFO_DB',
                     'RFO_AC', 'RFO_T0O0G0', 'RFO_T1O0G0', 'RFO_T1O1G0', 'RFO_T1O0G1',
                     'RFO_RADPROP', 'RFO_UNBIAS', 'RFO_UNBIAS_DW', 'RFO_FINAL', 'RFO_FINAL_DISAG',
                     'RFO_FINAL_T0O0G0']
        savefigformat = 'png'

        #-----------------------------------------------------------
        # Read results of Wolfensberger et al.
        #-----------------------------------------------------------
        pathIn_comp = '/scratch/rgugerli/analysis/'
        dw = pd.read_csv(pathIn_comp+'wolfensberger_scatter_bias.csv', sep=';', index_col=0)

        #-----------------------------------------------------------
        # GET PERFSCORES
        #-----------------------------------------------------------
        perfscores={}
        for agg in ['10min', '60min']:
                perfscores[agg] = {}
                for ith in [0.1, 0.6]:# ,1.0]:
                        perfscores[agg][ith] = {}
                        for m in modellist:
                                perfscores[agg][ith][m] = pd.read_csv(pathIn+'perfscores_{}_{}_{}.csv'.format(m,
                                                        agg, str(ith).replace('.','_')), index_col=0)
        
        # # Not needed anymore, as DB is included in the previous script
        # modellist.append('RFO_DB')                
        # pathin_db = '/scratch/rgugerli/analysis/eval_bias_nov2021/data/RFO_database/'
        # data = pickle.load(open(pathin_db+'all_station_scores.p', 'rb'))
        # perfscores_10min[ith]['RFO_DB'] = data['10min']['RFO']['all'].T
        # perfscores_10min[ith]['RFO_DB'] = data['60min']['RFO']['all'].T 
        # perfscores_10min[ith]['RFO_DB']['Y-coord'] = perfscores_10min[ith]['RFO']['Y-coord'] 
        # perfscores_10min[ith]['RFO_DB']['X-coord'] = perfscores_10min[ith]['RFO']['X-coord']         
                        
        #%%
        
        dic = {}
        dic['gauge'] = {'label' : 'Gauges (10min)', 'color' : 'black'}
        dic['RFO_standard'] = {'label' : 'RFO (10min)', 'color' : '#33a02c'}
        dic['RZC'] = {'label': 'PRECIP (10min)', 'color' : 'grey'}
        dic['RFQ'] = {'label': 'RFQ (10min)', 'color' : '#1f78b4'}
        dic['RFO_no_pp'] = {'label' : 'RFO without postprocessing', 'color' : '#66c2a4'}
        dic['RFO_no_disag'] = {'label' : 'RFO without disag-factor', 'color' : '#ccece6'}

        dic['RFO'] = {'label' : 'RFO (10min)', 'color' : '#33a02c'}
        dic['RFO_2016_19'] = {'label' : 'RFO (2016-19, 10min)', 'color' : '#b2df8a'}
        dic['CPCH'] =  {'label' : 'CPCH (10min)', 'color' : '#fb9a99'}
        dic['RFQ_v1'] = {'label': 'RFQ (old, 10min)', 'color' : '#a6cee3'}
        dic['RFO_DB'] = {'label' : 'RFO (DB, 10min)', 'color' : '#b3de69'}
        dic['RFO_AC'] = {'label': 'RFO (AC, 10min)', 'color' : '#ffff99'}
        dic['RFO_T0O0G0'] =  {'label' : 'RFO (NO TIMEDISAG, NO POSTPROC)', 'color' : '#fdbf6f'}
        dic['RFO_T1O0G0'] = {'label' : 'RFO (NO POSTPROC)', 'color' : '#ff7f00'}
        dic['RFO_T1O0G1'] = {'label' : 'RFO (GaussSmooth)', 'color' : '#cab2d6'}
        dic['RFO_T1O1G0']  = {'label' : 'RFO (OutRem)', 'color' : '#6a3d9a'}

        dic['RFO_UNBIAS'] = {'label': 'RFO (2016-2021, unbiased)', 'color' : '#a6cee3'}
        dic['RFO_UNBIAS_DW'] = {'label': 'RFO (2016-2021, unbiased new)', 'color' : '#a6cee3'}
        dic['RFO_FINAL'] = {'label': 'RFO (2016-2021, final version)', 'color' : '#a6cee3'}
        dic['RFO_FINAL_DISAG'] = {'label': 'RFO (2016-2021, new temp. disag.)', 'color' : '#a6cee3'}

        dic['RFO_FINAL_T0O0G0'] = {'label' : 'RFO (2016-2021, final without temp. disag.)', 'color' : '#a6cee3'}

        sys.exit()
        # ---------------------------------------------------------------------
        # PLOTS
        # ---------------------------------------------------------------------
        figsetup = {}
        figsetup[(0,0)] = {'model': 'RFO_FINAL', 'tit' : '(a) RainForest RFO (ISO0_height)'}
        figsetup[(0,1)] = {'model': 'RFQ', 'tit' :  '(b) RainForest RFQ (COSMO-1 Tair)'}
        figsetup[(1,0)] = {'model': 'RZC', 'tit' :  '(c) PRECIP (RZC)'}
        figsetup[(1,1)] = {'model': 'CPCH', 'tit' :  '(d) CombiPrecip (CPCH)'}
                
        # thresh = 0.6
        # tagg = '10min'
        
        for tagg in ['10min', '60min']:
                for ith in [0.1, 0.6]:
                        fig = plot_with_figsetup_2x2(figsetup, time_agg=tagg, thresh=ith, metric_in='BIAS')
                        fig.savefig(pathOut+'maps_logbias_diff_prod_db_{}_{}_{}.png'.format(str(ith).replace('.', '_'), tagg, timeperiod))

                        fig = plot_with_figsetup_2x2(figsetup,time_agg=tagg, thresh=ith, metric_in='SCATTER')
                        fig.savefig(pathOut+'maps_scatter_diff_prod_db_{}_{}_{}.png'.format(str(ith).replace('.', '_'), tagg, timeperiod))

        # ---------------------------------------------------------------------
        figsetup = {}
        figsetup[(0,0)] = {'model': 'RFO', 'tit' :  '(a) RainForest RFO (2016-2021, operational)'}
        figsetup[(0,1)] = {'model': 'RFO_DB', 'tit' :  '(b) RainForest RFO (2016-2021, database)'}
        figsetup[(0,2)] = {'model': 'RFO_2016_19', 'tit' : '(c) RainForest RFO (2016-2019)'}
        figsetup[(1,0)] = {'model': 'RFO_FINAL_T0O0G0','tit' : '(d) RainForest RFO (final, no temp. disag.)'}
        figsetup[(1,1)] = {'model': 'RFO_FINAL','tit' : '(e) RainForest RFO (final version)'}
        # figsetup[(1,0)] = {'model': 'RFO_AC','tit' : '(d) RainForest RFO (advection correction)'}
        figsetup[(1,2)] = {'model': 'RFO_FINAL_DISAG', 'tit' :  '(f) RainForest RFO (new temp. disag.)'}        
        # figsetup[(1,2)] = {'model': 'RFO_T1O0G0', 'tit' :  '(f) RainForest RFO (no postproc)'}
        
        for tagg in ['10min', '60min']:
                for ith in [0.1, 0.6]:
                        fig = plot_with_figsetup_2x3(figsetup, time_agg=tagg, thresh=ith,metric_in='BIAS')
                        fig.savefig(pathOut+'maps_logbias_RFO_db_{}_{}_{}.png'.format(str(ith).replace('.', '_'), tagg, timeperiod))

                        fig = plot_with_figsetup_2x3(figsetup, time_agg=tagg, thresh=ith, metric_in='SCATTER')
                        fig.savefig(pathOut+'maps_scatter_RFO_db_{}_{}_{}.png'.format(str(ith).replace('.', '_'), tagg, timeperiod))

        # ---------------------------------------------------------------------
        figsetup = {}
        figsetup[(0,0)] = {'model': 'RFO', 'tit' :  '(a) RainForest RFO (2016-2021, operational)'}
        figsetup[(0,1)] = {'model': 'RFO_DB', 'tit' :  '(b) RainForest RFO (2016-2021, database)'}
        # figsetup[(1,1)] = {'model': 'RFO_UNBIAS_DW','tit' : '(c) RainForest RFO (unbiased)'}
        figsetup[(1,0)] = {'model': 'RFO_FINAL','tit' : '(c) RainForest RFO (final version)'}
        figsetup[(1,1)] = {'model': 'RFO_FINAL_DISAG','tit' : '(d) RainForest RFO (new temp. disag.)'}

        for tagg in ['10min', '60min']:
                for ith in [0.1, 0.6]:
                        fig = plot_with_figsetup_2x2(figsetup, time_agg=tagg, thresh=ith,metric_in='BIAS')
                        fig.savefig(pathOut+'maps_logbias_RFO_db_{}_{}_{}.png'.format(str(ith).replace('.', '_'), tagg, timeperiod))

                        fig = plot_with_figsetup_2x2(figsetup, time_agg=tagg, thresh=ith, metric_in='SCATTER')
                        fig.savefig(pathOut+'maps_scatter_RFO_db_{}_{}_{}.png'.format(str(ith).replace('.', '_'), tagg, timeperiod))


        # plt.figure()
        # ax = plt.subplot2grid((1,1),(0,0))
        # ax.plot(perfscores['RFO']['logBias'], perfscores['RFQ']['logBias'], marker='o', lw=0, label='RFQ')
        # ax.plot(perfscores['RFO']['logBias'], perfscores['RZC']['logBias'], marker='o', lw=0, label='RZC')
        # ax.set_ylabel('logBias [dB]')
        # ax.set_xlabel('logBias of RFO (standard) [dB]')
        # ax.legend(loc='upper left', framealpha=0)
        # plt.savefig(pathOut+'scatterplot_logbias_diff_prod_{}_{}.png'.format(temp_agg, timeperiod))

        # plt.figure()
        # ax = plt.subplot2grid((1,1),(0,0))
        # ax.plot(perfscores['RFO_standard']['logBias'], perfscores['RFO_no_disag']['logBias'], marker='o', lw=0, label='no temporal disaggregation')
        # ax.plot(perfscores['RFO_standard']['logBias'], perfscores['RFO_no_Gauss']['logBias'], marker='o', lw=0, label='no gaussian smoothing')
        # ax.plot(perfscores['RFO_standard']['logBias'], perfscores['RFO_no_OR']['logBias'], marker='o', lw=0, label='no outlier removal')
        # ax.set_ylabel('logBias [dB]')
        # ax.set_xlabel('logBias of RFO (standard) [dB]')
        # ax.legend(loc='upper left', framealpha=0)
        # plt.savefig(pathOut+'scatterplot_logbias_same_product_{}_{}.png'.format(temp_agg, timeperiod))

        # Plot scatterplot of scatter between two versions
        tagg = '10min'; ith = 0.6
        mx = 'RFO_DB'
        my1 = 'RFO_FINAL'
        my2 = 'RFO_FINAL_T0O0G0'
        plt.figure()
        plt.plot([0,5], [0,5], color='grey')
        plt.plot(perfscores[tagg][ith][mx]['scatter'], 
                perfscores[tagg][ith][my1]['scatter'], 
                color='blue', lw=0, marker='o', label=my1)
        plt.plot(perfscores[tagg][ith][mx]['scatter'],
                perfscores[tagg][ith][my2]['scatter'], color='green', lw=0, marker='o', label=my2)
        plt.ylabel('qpe-RFO, scatter [dB]'); plt.xlabel('database-RFO, scatter [dB]')
        plt.legend(loc='upper left')
        plt.savefig(pathOut+'scatterplot_scatter_{}_{}_{}_{}_{}_{}.png'.format(mx,my1,my2,tagg,str(ith).replace('.','_'),timeperiod))

        plt.figure()
        plt.plot([-4,2],[-4,2], color='grey')
        plt.plot(perfscores[tagg][ith][mx]['logBias'],
                perfscores[tagg][ith][my1]['logBias'], color='blue', lw=0, marker='o', label=my1)
        plt.plot(perfscores[tagg][ith][mx]['logBias'], 
                perfscores[tagg][ith][my2]['logBias'], color='green', lw=0, marker='o', label=my2)
        plt.ylabel('qpe-RFO, 10 log.Bias [dB]'); plt.xlabel('database-RFO, 10 log. Bias [dB]')
        plt.legend(loc='upper left')
        plt.legend(loc='upper left')
        plt.savefig(pathOut+'scatterplot_logBias_{}_{}_{}_{}_{}_{}.png'.format(mx,my1,my2,tagg,str(ith).replace('.','_'),timeperiod))