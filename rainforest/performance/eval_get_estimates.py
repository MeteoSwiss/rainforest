#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Class to extract the station data from the QPE maps and 
assemble one DataFrame with 10min estimates of gauges and RFmodels

"""

import numpy as np
import pandas as pd
from pathlib import Path
import dask.dataframe as dd
import copy
import pickle
import datetime
import yaml

import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))

import logging
logging.getLogger().setLevel(logging.INFO)

from ..common.retrieve_data import retrieve_prod
from ..common.utils import envyaml, get_qpe_files_multiple_dirs
from ..common.lookup import get_lookup
from ..common.io_data import read_cart


def getGaugeObservations(gaugefolder, t0=None, t1=None, slf_stations=False, 
                        missing2nan=False):
    """_summary_

    Args:
        gaugefolder (str) : Folder of database where all gauge data is stored
        t0 (datetime object) : Start time of gauge observations. Default is None
        t1 (datetime object) : End time of gauge observations. Default is None
        slf_stations (bool, optional): If slf stations are included. Defaults to False.

    Returns:
        Object : DataFrame object with all gauge observations in mm/h
    """

    #----------------------------------------------------------------------------
    # Read gauge data
    #----------------------------------------------------------------------------
    gauge_all = dd.read_csv(str(Path(gaugefolder, '*.csv.gz')), 
                        compression='gzip', 
                        assume_missing=True,
                        dtype = {'TIMESTAMP':int,  'STATION': str})

    gauge_all = gauge_all.compute().drop_duplicates()

    if missing2nan:
        gauge_all = gauge_all.replace(-9999,np.nan)

    # Assure that datetime object is in UTC
    if t0 != None :
        gauge_all = gauge_all.loc[(gauge_all['TIMESTAMP']>= t0.replace(tzinfo=datetime.timezone.utc).timestamp()),
                            ['STATION','TIMESTAMP','RRE150Z0']]
    if t1 != None :
        gauge_all = gauge_all.loc[(gauge_all['TIMESTAMP']<= t1.replace(tzinfo=datetime.timezone.utc).timestamp()),
                    ['STATION','TIMESTAMP','RRE150Z0']]

    # Remove SLF stations
    if slf_stations == False:
        list_slf = []
        for ss in gauge_all['STATION'].unique():
            if ss.startswith('SLF'):
                list_slf.append(ss)
        gauge_all = gauge_all.loc[~gauge_all['STATION'].isin(list_slf)]

    # Get datetime object for index
    gauge_all['TIME'] = [datetime.datetime.utcfromtimestamp(ti).replace(tzinfo=datetime.timezone.utc) 
                                for ti in gauge_all['TIMESTAMP']]
    # Convert gauge accumulation to mm/h
    gauge_all['RRE150Z0'] = gauge_all['RRE150Z0'] * 6

    return gauge_all

def get_QPE_filelist(qpefolder, time_conditions, modellist):

    # Get all qpe files
    tmp = get_qpe_files_multiple_dirs(qpefolder, 
                        time_agg=time_conditions['time_agg'],
                        list_models = modellist)

    # Get only timesteps where at least 2 files are available during 10 min period
    qpe_files10 = copy.deepcopy(tmp)
    for k in tmp.keys():
        for m in tmp[k].keys():
            if 'T0' not in m: # if a RF model without temporal aggregation is included
                if len(tmp[k][m]) < time_conditions['file_tol']:
                    del qpe_files10[k][m]
            else:
                for it, time in enumerate(tmp[k][m]):
                    if time.split('/')[-1][11:12] != '0':
                        del qpe_files10[k][m][it]
                
    nmodels = np.array([len(d) for d in qpe_files10.values()])

    # Get only timesteps where all qpe models are available
    qpe_files10_filt = {}
    for i, k in enumerate(qpe_files10.keys()):
        if nmodels[i] == max(nmodels):
            qpe_files10_filt[k] = qpe_files10[k]

    return qpe_files10_filt

class compileMapEstimates(object):

    def __init__(self,config_file, overwrite=False, tagg_hourly = True, 
                slf_stations=False, save_output = True):
        """ Initiate the class evaluation

        Args:
            config_file (str): Path to a configuration file
        """

        try:
            config = envyaml(config_file)
            self.configfile = config_file
        except:
            logging.warning('Using default config as no valid config file was provided')
            config = envyaml(dir_path + '/default_config.yml')
            self.configfile = dir_path + '/default_config.yml'

        # Check if there are already some data compiled:
        if 'FILE_10MIN' in config.keys():
            if os.path.exists(config['FILE_10MIN']) :
                if overwrite :
                    logging.info('Overwriting existing compilation of model estimates')
                else:
                    logging.info('Files already compiled, set overwrite=True if you want a new compilation')
                    return

        # Check if setup is ok, and get data where necessary
        self._check_elements(config)
        self._get_data()

        # Run main part
        self.extractEstimatesFromMaps(slf_stations=slf_stations, tagg_hourly=tagg_hourly, 
                            save_output=save_output)

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

            if not os.path.exists(config['PATHS']['MAINFOLDER'] + '/results/') :
                os.mkdir(config['PATHS']['MAINFOLDER'] + '/results/')
                logging.info('Creating subfolder /results/ to save the plots and performance scores.')
              

        if ('GAUGEFOLDER' not in config['PATHS'].keys()) or (not os.path.exists(config['PATHS']['GAUGEFOLDER'])):
            logging.error('No existing gauge folder was defined, defining default folder')
            self.gaugefolder = '/store/msrad/radar/radar_database_v2/gauge/'
        else:
            self.gaugefolder = config['PATHS']['GAUGEFOLDER']

        if ('TIME_START' not in config.keys()):
            logging.error('No starting time was defined, please check.')
        else:
            try:
                self.tstart = datetime.datetime.strptime(config['TIME_START'], '%Y%m%d%H%M').replace(tzinfo=datetime.timezone.utc)
            except:
                 logging.error('Starting time is not valid, please check that the format corresponds to YYYYMMDDHHMM.')

        if ('TIME_END' not in config.keys()):
            logging.error('No ending time was defined, please check.')
        else:
            try:
                self.tend = datetime.datetime.strptime(config['TIME_END'], '%Y%m%d%H%M').replace(tzinfo=datetime.timezone.utc)
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

    def _get_data(self):
        """
            Checks if data is already there and gets it from the archive if not
        """

        # Check if the RF-models are there
        for model in self.modellist:
            self.model_files = {}
            path = self.qpefolder+'{}'.format(model)
            if not os.path.exists(path) or (len(os.listdir(path)) == 0):
                logging.info('Extracting {} files from archive'.format(model))
                try:
                    path = self.qpefolder+'{}'.format(model)
                    self.model_files[model] = retrieve_prod(path + '/', self.tstart, 
                                                                    self.tend, model)
                    logging.info('Model data: {} taken from file archive!'.format(model))
                except:
                    logging.error('No QPE maps available for {}, please check path or produce QPE maps'.format(model))

        # Check for reference data
        if self.references:
            self.ref_files = {}
            for ref in self.references:
                path = self.qpefolder+'{}'.format(ref)
                if not os.path.exists(path) or (len(os.listdir(path)) == 0):
                    logging.info('Extracting {} files from archive'.format(ref))
                    if (ref == 'CPC') or (ref == 'CPCH'):
                        path = self.qpefolder
                        self.ref_files[ref] = retrieve_prod(path, self.tstart, 
                                                                    self.tend, ref, 
                                                                    pattern = '*5.801.gif')
                    elif (ref == 'CPC60'):
                        path = self.qpefolder
                        self.ref_files[ref] = retrieve_prod(path, self.tstart, 
                                                                    self.tend, ref, 
                                                                    pattern = '*60.801.gif')                        
                    else:
                        path = self.qpefolder+'{}'.format(ref)
                        self.ref_files[ref] = retrieve_prod(path + '/', self.tstart, 
                                                                    self.tend, ref)
                self.modellist.append(ref)
        
        return


    def extractEstimatesFromMaps(self, slf_stations=False, tagg_hourly=True, save_output=True):
        """
        Extracts all estimates from the QPE maps and compiles one pickle file with a DataFrame
        including all the model estimates for 10min and hourly aggregation, incl. gauge observations

        Args:
            slf_stations (bool, optional): if SLF stations should be included. Defaults to False.
            tagg_hourly (bool, optional): if hourly estimates are also included. Defaults to True.
            save_output (bool, optional): Saves the generated DataFrame to a pickle file. Defaults to True.

        Returns:
            _type_: DataFrame with all timesteps,
                    if tagg_hourly=True, a second DataFrame with hourly time steps is included
                    The config file is also updated with a the path where the output is saved
        """

        # Get full time span
        time_res = {}
        tstamps_10min = pd.date_range(start=self.tstart, end=self.tend, 
                                    freq='10min',tz=datetime.timezone.utc)
        time_res['10min'] = {'time_agg':10,'file_tol':2,'out':'10min'}

        if 'CPC60' in self.modellist:
            cpc60 = True
            self.modellist.remove('CPC60')
            tagg_hourly = True
        else:
            cpc60 = False

        if tagg_hourly:
            # Assure that hourly starts with 0 minutes
            tstart = datetime.datetime(self.tstart.year, self.tstart.month, 
                        self.tstart.day, self.tstart.hour,0).replace(tzinfo=datetime.timezone.utc)
            tstamps_60min = pd.date_range(start=tstart, end=self.tend, 
                                    freq='H', tz=datetime.timezone.utc)
            time_res['60min'] = {'time_agg':60,'file_tol':4,'out':'60min'}

        # Lookup table for coordinates between stations and Cartesian grid
        #-----------------------------------------------------------
        lut = get_lookup('station_to_qpegrid')

        # Get ground reference (gauge data)
        #-----------------------------------
        logging.info('Get gauge observations')
        gauge_all = getGaugeObservations(gaugefolder=self.gaugefolder, 
                        t0 = self.tstart, t1= self.tend, slf_stations = False,
                        missing2nan=False)
        stations = np.unique(gauge_all['STATION'])

        # Get all model files
        #----------------------
        list_models = self.modellist
        if 'RFO_DB' in list_models:
            list_models.remove('RFO_DB')

        # # Get all qpe files
        # tmp = get_qpe_files_multiple_dirs(self.qpefolder, 
        #                     time_agg=time_res['10min']['time_agg'],
        #                     list_models = list_models)

        # # Get only timesteps where at least 2 files are available during 10 min period
        # qpe_files10 = copy.deepcopy(tmp)
        # for k in tmp.keys():
        #     for m in tmp[k].keys():
        #         if 'T0' not in m: # if a RF model without temporal aggregation is included
        #             if len(tmp[k][m]) < time_res['10min']['file_tol']:
        #                 del qpe_files10[k][m]
        #         else:
        #             for it, time in enumerate(tmp[k][m]):
        #                 if time.split('/')[-1][11:12] != '0':
        #                     del qpe_files10[k][m][it]
                    
        # nmodels = np.array([len(d) for d in qpe_files10.values()])

        # # Get only timesteps where all qpe models are available
        # qpe_files10_filt = {}
        # for i, k in enumerate(qpe_files10.keys()):
        #     if nmodels[i] == max(nmodels):
        #         qpe_files10_filt[k] = qpe_files10[k]

        qpe_files10_filt = get_QPE_filelist(self.qpefolder, time_res['10min'],
                                         list_models)

        precip_qpe = {}
        for m in list_models:
            precip_qpe[m] = np.zeros((len(tstamps_10min), len(stations)))
        
        logging.info('Get QPE model estimations for each station at 10min')
        for i, tstep in enumerate(tstamps_10min): # Loop on timesteps    
            # Get QPE precip
            for m in list_models:
                if (i == 0) or (i == len(tstamps_10min)-1):
                    logging.info('Processing model {} and timestep {}'.format(m, tstep))
                if tstep in qpe_files10_filt.keys():
                    for f in qpe_files10_filt[tstep][m]:
                        data = read_cart(f)
                        # if len(np.shape(data)) == 3:
                        #     data = np.flipud(data[0,:,:])
                        for j,s in enumerate(stations):
                            try:
                                precip_qpe[m][i,j] += data[lut[s]['00'][0], lut[s]['00'][1]]
                            except:
                                logging.info('Could not extract value for model {}, timestep {} and station {}'.format(m, tstep, s))
                                continue    
                    precip_qpe[m][i] /= len(qpe_files10_filt[tstep][m])
                else:
                    precip_qpe[m][i] = np.nan

        # Add GAUGE DATA
        precip_qpe['GAUGE'] = pd.DataFrame(columns=stations, index=tstamps_10min)
        # Compile data into frame with timestamp as index and stations as columns
        for ss in stations:
            col = gauge_all.loc[gauge_all['STATION'] == ss, 'RRE150Z0'].to_frame(name=ss)
            col.set_index(gauge_all.loc[(gauge_all['STATION'] == ss),'TIME'], inplace=True)
            precip_qpe['GAUGE'][ss] = col
            precip_qpe['GAUGE'][ss].fillna(0, inplace=True)
            precip_qpe['GAUGE'][ss].loc[precip_qpe['GAUGE'][ss] < 0] = np.nan

        for m in list_models:
            precip_qpe[m] = pd.DataFrame(precip_qpe[m], index=tstamps_10min, columns=precip_qpe['GAUGE'].columns)

        if 'RFO_DB' in self.modellist:
            # Read data
            path = self.mainfolder+'/data/RFO_DB'
            rfo_db = dd.read_parquet(path+'/reference_RFO.parquet').compute()
            rfo_db_models = [m for m in rfo_db.columns if m.startswith('RFO')]
            rfo_db['TIME'] = [datetime.datetime.utcfromtimestamp(ti) for ti in rfo_db['TIMESTAMP']]
            for m in rfo_db_models:
                precip_qpe[m+'_DB'] = pd.DataFrame(index = tstamps_10min)
                for ss in stations:
                    col = rfo_db.loc[rfo_db['STATION'] == ss, m].to_frame(name=ss)
                    col.index = rfo_db.loc[rfo_db['STATION'] == ss, 'TIME']
                    precip_qpe[m+'_DB'] = precip_qpe[m+'_DB'].join(col)

        # Save output
        #--------------
        if save_output :
            pathOut = self.qpefolder
            save_file = pathOut+'all_data_10min_{}_{}.p'.format(datetime.datetime.strftime(self.tstart, '%Y%m%d%H%M'),
                                             datetime.datetime.strftime(self.tend, '%Y%m%d%H%M'))
            logging.info('Saving {}'.format(save_file))
            pickle.dump(precip_qpe, open(save_file, 'wb'))

            # Add path to config file
            with open(self.configfile,'a') as yamlfile:
                yaml.safe_dump({'FILE_10MIN' : save_file}, yamlfile)


        # Aggregate to 60min
        #-----------------------------------------------------------
        if tagg_hourly :
            precip_60min = {}

            # Convert to hourly estimates
            logging.info('Converting 10min estimates to hourly estimates')
            for model in precip_qpe.keys():
                precip_60min[model] = pd.DataFrame(columns = precip_qpe[model].columns, index=tstamps_60min)
                for it in tstamps_60min:
                    if it > precip_qpe[model].index[0]:
                        tstamps = pd.date_range(start=it-datetime.timedelta(minutes=50), end=it, 
                                        freq='10min', tz=datetime.timezone.utc)
                        for ss in precip_qpe[model].columns:
                            # This results in the precipitation sum during one hour, which is given in mm/h
                            if precip_qpe[model][ss][tstamps].count() > time_res['60min']['file_tol']:
                                precip_60min[model][ss][it] = precip_qpe[model][ss][tstamps].mean()

            if cpc60 : 
                pass
                # TODO 
                precip_60min['CPC60'] = self._get_cpc60(tstamps_60min)

            # Save output
            #-----------------------------------------------------------
            if save_output :
                pathOut = self.qpefolder
                save_file_60 = pathOut+'all_data_60min_{}_{}.p'.format(datetime.datetime.strftime(self.tstart, '%Y%m%d%H%M'),
                                                        datetime.datetime.strftime(self.tend, '%Y%m%d%H%M'))
                logging.info('Saving {}'.format(save_file_60))
                pickle.dump(precip_60min, open(save_file_60, 'wb'))

                # Update config file
                with open(self.configfile,'a') as yamlfile:
                    yaml.safe_dump({'FILE_60MIN' : save_file_60}, yamlfile)

        return