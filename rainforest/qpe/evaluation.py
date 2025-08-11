#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function to evaluate QPE runs with gauge data

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019
"""

# global imports
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import copy
import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Local imports
from ..common.utils import read_df, get_qpe_files, get_qpe_files_multiple_dirs, perfscores
from ..common.utils import timestamp_from_datetime, nearest_time
from ..common.lookup import get_lookup
from ..common.io_data import read_cart
from ..common.graphics import score_plot, qpe_scatterplot
from ..common.retrieve_data import retrieve_CPCCV

def evaluation(qpefolder, gaugepattern, list_models = None, 
               outputfolder = './', t0 = None, t1 = None,
               bounds10 = [0,2,10,100], bounds60 = [0,1,10,100]):
    
    """
        PErforms an evaluation of QPE products with reference gauge data
        
        Parameters
        ----------
        qpefolder : str
            Main directory where the QPE data is stored, each model corresponding
            to a subfolder, as given by qpe_compute.py
        gaugepattern : str
            The pattern of gauge files that contain the gauge data.
            on CSCS: '/store/msrad/radar/radar_database/gauge/*.csv.gz'
        t0: datetime.datetime instance
            starting time of the time range, default is first timestep available
        t1 : datetime.datetime instance
            end time of the time range, default is last timestep available
        bounds10 : list of float
            list of precipitation bounds for which to compute scores separately
            at 10 min time resolution
            [0,2,10,100] will give scores in range [0-2], [2-10] and [10-100]
        bounds60 : list of float
            list of precipitation bounds for which to compute scores separately
            at hourly time resolution
            [0,1,10,100] will give scores in range [0-1], [1-10] and [10-100]
        list_models : list of str
            list of models to use in the evaluation, default is to use all
            subfolders (models) available in qpefolder
                        
     
    """
    
    if type(qpefolder) == list:
        logging.info('Getting all files from multiple qpe folders')
        tmp = get_qpe_files_multiple_dirs(qpefolder, time_agg = 10, list_models = list_models)
    else:
        logging.info('Getting all files from qpe folder {:s}'.format(qpefolder))
        tmp = get_qpe_files(qpefolder, time_agg = 10, list_models = list_models)
        
    # Get only timesteps where at least 2 files are available during 10 min period
    qpe_files10 = copy.deepcopy(tmp)
    for k in tmp.keys():
        for m in tmp[k].keys():
            if len(tmp[k][m]) < 2:
                del qpe_files10[k][m]
                
    # number of models by timestep
    nmodels = np.array([len(d) for d in qpe_files10.values()])
   
        
    # Get only timesteps where all qpe models are available
    qpe_files10_filt = {}
    for i, k in enumerate(qpe_files10.keys()):
        if nmodels[i] == max(nmodels):
            qpe_files10_filt[k] = qpe_files10[k]
            
    models = list(list(qpe_files10_filt.values())[0].keys())
    if list_models == None:
        list_models = nmodels
        
    tsteps = sorted(list(qpe_files10_filt.keys()))
    
    logging.info('Reading gauge data from pattern {:s}'.format(gaugepattern))
    df = read_df(gaugepattern)
    
    
    logging.info('Converting to pandas dataframe...')
    t0 = timestamp_from_datetime(tsteps[0])
    t1 = timestamp_from_datetime(tsteps[-1])
    df = df[(df['TIMESTAMP'] >= t0) & (df['TIMESTAMP'] <= t1)].compute()
    stations = np.unique(df['STATION'])
    
    logging.info('Getting lookup table')
    lut = get_lookup('station_to_qpegrid')
    
    # Initialize matrices of precip at stations
    precip_qpe = {}
    for m in models:
        precip_qpe[m] = np.zeros((len(tsteps), len(stations))) 
    precip_ref = np.zeros((len(tsteps), len(stations)))
   
    for i, tstep in enumerate(tsteps): # Loop on timesteps
        logging.info('Reading timestep {:d}/{:d}'.format(i+1, len(tsteps)))
        #  Get reference precip
        tstamp = timestamp_from_datetime(tstep)
        measures_10 = df[df['TIMESTAMP'] == tstamp]
        idx = np.searchsorted(stations, measures_10['STATION']) # idx of stations for this timestep
        precip_ref[i, idx] = measures_10['RRE150Z0'] * 6
      
        # Get QPE precip
        for m in models:
            for f in qpe_files10_filt[tstep][m]:
                data = read_cart(f)
                for j,s in enumerate(stations):
                    precip_qpe[m][i,j] += data[lut[s]['00'][0], lut[s]['00'][1]]
                    
            precip_qpe[m][i] /= len(qpe_files10_filt[tstep][m])
   
    #Get avg over 10min peri
    scores10 = {}
    # COmpute 10min scores
    valid_ref = np.isfinite(precip_ref.ravel())
    for m in models:
        scores10[m] = perfscores(precip_qpe[m].ravel()[valid_ref],
                precip_ref.ravel()[valid_ref],
                bounds10)
    
    # Hourly resolution
    # get hour of tsteps
    hours = np.array([nearest_time(t, 60) for t in tsteps])
    hours_u,cnt = np.unique(hours, return_counts = True)
    precip_qpe60 = {}

    for m in models:
        data = precip_qpe[m]
        precip_qpe60[m] = np.array([np.nanmean(data[hours == h], axis = 0) 
            for h in hours_u[cnt == 6]] )
    if 'CPC.CV' in list_models:
        logging.info('Retrieving CPC.CV data. at hourly resolution...')
        
        precip_qpe60['CPC.CV'] = []
        for h in hours_u:
            precip_qpe60['CPC.CV'] .append(retrieve_CPCCV(h, stations))
        precip_qpe60['CPC.CV']  = np.array(precip_qpe60['CPC.CV'])
        models.append('CPC.CV')
        
    precip_ref60 = np.array([np.nanmean(precip_ref[hours == h], axis = 0) 
            for h in hours_u[cnt == 6]] )
    
    scores60 = {}
    # COmpute 60min scores
    valid_ref = np.isfinite(precip_ref60.ravel())
    for m in models:
        scores60[m] = perfscores(precip_qpe60[m].ravel()[valid_ref],
                precip_ref60.ravel()[valid_ref],
                bounds60)
        
    # Make score plots
    timerange = datetime.datetime.strftime(tsteps[0], '%Y%m%d%H%M') + '_' +\
        datetime.datetime.strftime(tsteps[-1], '%Y%m%d%H%M')
        
    # Save the data as parquet
    for m in models:
        df_precip = pd.DataFrame(precip_qpe[m],columns=stations, index=tsteps) 
        df_precip.to_csv(outputfolder+'/'+str(m)+'_qpe10min_'+timerange+'.csv', float_format='%.3f')
    
    title = datetime.datetime.strftime(tsteps[0], '%d %b %Y %H:%M')
    title += ' - ' +  datetime.datetime.strftime(tsteps[-1], '%d %b %Y %H:%M')
    score_plot(scores10, title + ', agg = 10 min ', figsize = (13,8))
    plt.savefig(outputfolder + 'scores_agg10_' + timerange+ '.png', 
                bbox_inches='tight',
                dpi = 300)
    
    score_plot(scores60, title + ', agg = 60 min ', figsize = (13,8))
    plt.savefig(outputfolder + 'scores_agg60_' + timerange+ '.png', 
                bbox_inches='tight',
                dpi = 300)
    
    # Make scatterplots
    qpe_scatterplot(precip_qpe,  precip_ref, figsize = (10,8.5),
                    title_prefix = title + ', agg = 10 min ')
    plt.savefig(outputfolder + 'scatterplots10_' + timerange+ '.png', 
                bbox_inches='tight',
                dpi = 300)
    qpe_scatterplot(precip_qpe60, precip_ref60, figsize = (10,8.5),
                    title_prefix = title + ', agg = 60 min ')
    plt.savefig(outputfolder + 'scatterplots60_' + timerange+ '.png', 
                bbox_inches='tight',
                dpi = 300)

   
