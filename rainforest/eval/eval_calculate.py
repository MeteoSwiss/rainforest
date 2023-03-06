#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""   
    Class to calculate performance scores of different
    QPE products

    Rebecca Gugerli, EPFL-MeteoSwiss, February 2023

"""

import pandas as pd
import numpy as np
import pickle

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import logging
logging.getLogger().setLevel(logging.INFO)

from pysteps.verification.detcontscores import *

from ..common.utils import envyaml
from ..common.constants import METSTATIONS

#----------------------------------------------------------
# PERFORMANCE SCORES
#-----------------------------------------------------------
def calcScoresStations(precip_ref, precip_est, threshold=[0.1,0.1]):
    """
    Computes the following scores for a given reference and estimation
        Contingency table: true_pos,'true_neg','false_pos','false_neg'
        Root mean square error [mm/h]: RMSE
        Pearson correlation [-]: corr_p
        Scatter (in dB, see Germann et al. (2006)): scatter
        Bias (in dB, see Germann et al. (2006)logBias
        Number of observations: n_values
        Number of observations with double conditional bias: n_events_db
        Total sum of precipitation at reference sum_gauge_db
        Total sum of precipitation at estimate: sum_qpe_db

        A double conditional bias affects RMSE, Corr_p, scatter, logbias, 
                                            n_events_db, sum_qpe_db, sum_gauge_db

    Args:
        precip_ref (_type_): DataFrame containing timesteps as index, and stations as columns
        precip_est (_type_): same as precip_ref
        threshold (list, optional): Thresholds to set for double conditional scores. Defaults to [0.1,0.1].

    Returns:
        DataFrame object: Dataframe with stations as indices and scores as columns
    """

    if type(threshold) != list:
        th_ref = threshold
        th_est = th_ref
    else:
        th_ref = threshold[0]
        th_est = threshold[1]

    perfscores = pd.DataFrame(columns=['true_pos','true_neg','false_pos','false_neg',
                                    'RMSE','corr_p','scatter','logBias', 'n_values', 'n_events_db','sum_gauge_db','sum_qpe_db'], 
                            index=precip_ref.columns.unique())

    stations = METSTATIONS.copy()
    stations.index = stations['Abbrev']
    perfscores['X'] = stations['X']
    perfscores['Y'] = stations['Y']
    perfscores['Z'] = stations['Z']

    for ss in precip_ref.columns.unique():
        perfscores.loc[perfscores.index == ss, 'n_values'] = precip_ref[ss].count()
        perfscores.loc[perfscores.index == ss,'true_pos'] = precip_ref.loc[(precip_ref[ss] >= th_ref) & (precip_est[ss] >= th_est), ss].count()
        perfscores.loc[perfscores.index == ss,'true_neg'] = precip_ref.loc[(precip_ref[ss] < th_ref) & (precip_est[ss] < th_est), ss].count()
        perfscores.loc[perfscores.index == ss,'false_pos'] = precip_ref.loc[(precip_ref[ss] < th_ref) & (precip_est[ss] >= th_est), ss].count()
        perfscores.loc[perfscores.index == ss,'false_neg'] = precip_ref.loc[(precip_ref[ss] >= th_ref) & (precip_est[ss] < th_est), ss].count()

        doublecond = precip_ref.loc[(precip_ref[ss] >= th_ref) & (precip_est[ss] >= th_est)].index
        perfscores.loc[perfscores.index == ss,'n_events_db'] = len(doublecond)
        perfscores.loc[perfscores.index == ss,'sum_gauge_db'] = precip_ref[ss][doublecond].sum()
        perfscores.loc[perfscores.index == ss,'sum_qpe_db'] = precip_est[ss][doublecond].sum()

        if (precip_est[ss][doublecond].sum() > 0) & (precip_ref[ss][doublecond].sum() > 0):
            perfscores.loc[perfscores.index == ss,'logBias'] = np.round(10*np.log10(precip_est[ss][doublecond].sum()/precip_ref[ss][doublecond].sum()),decimals=4)
        else:
            logging.info('No measurements for station {}'.format(ss))
            continue
        try:
            scores = det_cont_fct(precip_est[ss][doublecond].to_numpy(dtype=float), precip_ref[ss][doublecond].to_numpy(dtype=float),
                                    scores=['RMSE','corr_p','scatter'])
            perfscores.loc[perfscores.index == ss,'RMSE'] = np.round(scores['RMSE'],decimals=4)
            perfscores.loc[perfscores.index == ss,'corr_p'] = np.round(scores['corr_p'],decimals=4)
            perfscores.loc[perfscores.index == ss,'scatter']= np.round(scores['scatter'],decimals=4)
        except:
            logging.info('Could not calculate scores for station {}'.format(ss))

    return perfscores

def calcScoresSwitzerland(precip_ref, precip_pred, threshold=[0.1,0.1]):
    
    if type(threshold) != list:
        th_ref = threshold
        th_est = th_ref
    else:
        th_ref = threshold[0]
        th_est = threshold[1]

    perfscores = pd.Series(index=['true_pos','true_neg','false_pos','false_neg',
                                'RMSE','corr_p','scatter','logBias', 'n_true', 'n_events_db'])
    perfscores['n_true'] = len(precip_ref[~np.isnan(precip_ref)])
    perfscores['true_pos'] = len(np.where((precip_ref >= th_ref) & (precip_pred >= th_est))[0])
    perfscores['true_neg'] = len(np.where((precip_ref < th_ref) & (precip_pred < th_est))[0])
    perfscores['false_pos'] = len(np.where((precip_ref < th_ref) & (precip_pred >= th_est))[0])
    perfscores['false_neg'] = len(np.where((precip_ref >= th_ref) & (precip_pred < th_est))[0])

    doublecond = (precip_ref>= th_ref) & (precip_pred>= th_est)
    perfscores['n_events_db'] = len(np.where(doublecond)[0])
    scores = det_cont_fct(precip_pred[doublecond], precip_ref[doublecond],
                        scores=['RMSE','corr_p','scatter'])

    perfscores['RMSE'] = np.round(scores['RMSE'],decimals=4)
    perfscores['corr_p'] = np.round(scores['corr_p'],decimals=4)
    perfscores['scatter'] = np.round(scores['scatter'],decimals=4)
    perfscores['logBias'] = np.round(10*np.log10(precip_pred[doublecond].sum()/precip_ref[doublecond].sum()),decimals=4)

    return perfscores


#----------------------------------------------------------
# APPLICATION ON RF-DATA: PERFORMANCE SCORES
#-----------------------------------------------------------

class calcPerfscores(object) :

    def __init__(self, configfile=None, read_only=False) :
        """
        Initiates the class to compute performance scores for RainForest models
        and other Cartesian QPE products of MeteoSwiss

        Args:
            configfile (str): Path to a file with general setup for 
                        evaluation based on case studies

        Returns:
            _type_: Returns a dictionnary only if read_only is True
        """

        if (configfile == None) or (not os.path.exists(configfile)):
            configfile = dir_path + '/default_config.yml'
            logging.info('Using default config as no valid config file was provided')

        config = envyaml(configfile)

        if 'MAINFOLDER' in config['PATHS'].keys():
            self.mainfolder = config['PATHS']['MAINFOLDER']
            if not os.path.exists(self.mainfolder+'/results/'):
                os.mkdir(self.mainfolder+'/results/')
        else:
            logging.info('No output folder is given, please check your config file.')

        # try:
        self.datestring = '{}_{}'.format(config['TIME_START'], config['TIME_END'])
        # PERFORMANCE ESTIMATES
        self.timeagg = config['PERFORMANCE']['TIMEAGGREGATION']
        self.doublecond = config['PERFORMANCE']['DOUBLECONDITIONAL']
        self.reference = config['PERFORMANCE']['REF_ESTIMATE']
        # Assemble all models
        self.modellist = config['RF_MODELS']
        self.modellist.extend(config['REFERENCES'])
        # except:
        #     logging.error('Config file is missing information, please check the default.')

        if not read_only :
            self.precip = {}
            if 'FILE_10MIN' in config.keys():
                self.precip['10min'] = pickle.load(open(config['FILE_10MIN'],'rb'))
            else:
                logging.info('Filepath to 10min QPE compilation is missing')

            if 'FILE_60MIN' in config.keys():
                self.precip['60min'] = pickle.load(open(config['FILE_60MIN'],'rb'))
            else: 
                logging.info('Filepath to hourly QPE compilation is missing')

            if len(self.precip) == 0:
                logging.error('No filenames are given... ')
                return

            logging.info('Calculating performance scores')
            self.scores = self.calcScores(self.modellist, reference=self.reference,
                    timeagg=self.timeagg, doublecond=self.doublecond)

        else:
            logging.info('Getting previously calculated performance scores')
            self.scores = self.readScores(self.modellist, reference=self.reference,
                                    timeagg=self.timeagg, doublecond=self.doublecond)


    def readScores(self, modellist, reference='GAUGE',
                    timeagg=['10min', '60min'], 
                    doublecond=[0.1, 0.6]):
        """
        Reads files and sorts them into a dictionnary with the following setup:
        scores[timeaggregation][threshold][model] = DataFrame(cols=score, index=stations)

        Args:
            modellist (list): list with all model names
            reference (str, optional): The estimates that are used as references. Defaults to 'GAUGE'.
            timeaggregation (list, optional): Time aggregations to analyse. Defaults to ['10min', '60min'].
            thresholds (list, optional): Double-conditional thresholds. Defaults to [0.1, 0.6].

        Returns:
           Dictionary: DataFrame (see above)
        """

        scores = {}
        for tagg in timeagg :
            scores[tagg] = {}
            for ith in doublecond :
                scores[tagg][ith] = {}
                for model in modellist :
                    filename = 'perfscores_{}_{}_doublecond_{}_{}_{}.csv'.format(self.datestring, tagg,
                                     str(ith).replace('.','_'), reference, model)
                    try:
                        scores[tagg][ith][model] = pd.read_csv(
                                            self.mainfolder+'/results/{}'.format(filename), 
                                            index_col=0)
                    except:
                        logging.warning('The following performance scores are not available {}'.format(filename))

        return scores

    def calcScores(self, modellist, reference='GAUGE',
                    timeagg=['10min', '60min'], 
                    doublecond=[0.1, 0.6]):
        """
        Core of this class, it manages reference and estimates and calls
        the routine self.calcScoresStations

        Args:
            reference (str, optional): Defines which precipitation source should 
                                        be used as reference. Defaults to 'GAUGE'.
            estimate (str, optional): Defines which precipitation source should be evaluated.
                                      Defaults to 'RFO'.
            tagg (str, optional): Defines time aggregation to evaluate. Defaults to '10min'.
            threshold (list, optional): Defines thresholds to be applied. Defaults to [0.1,0.6].

        Returns:
            DataFrame object: Dataframe with stations as indices and scores as columns
        """

        scores = {}
        for tagg in timeagg :
            scores[tagg] = {}
            for ith in doublecond :
                scores[tagg][ith] = {}
                for model in modellist :
                    filename = 'perfscores_{}_{}_doublecond_{}_{}_{}.csv'.format(self.datestring, tagg,
                                     str(ith).replace('.','_'), reference, model)
                    # Calculate score
                    logging.info('Calculating scores for {}'.format(filename))
                    scores[tagg][ith][model] = calcScoresStations(self.precip[tagg][reference], 
                                        self.precip[tagg][model], threshold=ith)
                    # Save file
                    scores[tagg][ith][model].to_csv(self.mainfolder+'/results/{}'.format(filename))

        return scores


if __name__ == '__main__':

    #-----------------------------------------------------------
    # PATHS
    #-----------------------------------------------------------
    pathIn = '/scratch/rgugerli/analysis/eval_bias_nov2021/data/'
    pathOut = '/scratch/rgugerli/analysis/eval_bias_nov2021/results/'

    #-----------------------------------------------------------
    # READ DATA - Gauges
    #-----------------------------------------------------------
    # Met stations for analysis
    pathIn_gauges = '/scratch/rgugerli/analysis/'
    info_gauges = pd.read_csv(pathIn_gauges+'data_stations.csv',header=0,
                            index_col='Abbrev',sep=';',encoding='latin-1')

    #-----------------------------------------------------------
    # READ DATA - QPE products
    #-----------------------------------------------------------
    datestr = '202110312000_202111012000'
    precip10min = pickle.load(open(pathIn+'all_data_10min_{}.p'.format(datestr),'rb'))
    precip60min = pickle.load(open(pathIn+'all_data_60min_{}.p'.format(datestr),'rb'))

    ref = 'gauge'
    models = list(precip10min.keys())
    models.remove(ref)
    if 'DISAG' in models:
        models.remove('DISAG')


    perfscores_10min={}
    perfscores_60min={}
    perfscores_switzerland_10min={}
    perfscores_switzerland_60min={}
    # for ith in [0.3, 0.6, 1.0]:
    # for ith in [0.1, 0.6, 1.0]:
    for ith in [0.1, 0.6]:
        perfscores_10min[ith] = {}
        perfscores_60min[ith] = {}
        perfscores_switzerland_10min[ith]={}
        perfscores_switzerland_60min[ith]={}
        for m in models:
            perfscores_10min[ith][m] = calc_perfscores(precip10min[ref],precip10min[m], th=ith)
            perfscores_10min[ith][m].to_csv(pathOut+'perfscores_{}_10min_{}.csv'.format(m,
                                    str(ith).replace('.','_')))
            perfscores_60min[ith][m] = calc_perfscores(precip60min[ref],precip60min[m], th=ith)
            perfscores_60min[ith][m].to_csv(pathOut+'perfscores_{}_60min_{}.csv'.format(m,
                                            str(ith).replace('.','_')))

            # precipref_10min_array = precip10min[ref].to_numpy().ravel()
            # precipqpe_10min_array = precip10min[m].to_numpy().ravel()
            # valid = (~np.isnan(precipqpe_10min_array)) & (~np.isnan(precipref_10min_array))
            # perfscores_switzerland_10min[ith][m] = calc_perfscores_CH(precipref_10min_array[valid], precipqpe_10min_array[valid],th=ith)
            # perfscores_switzerland_10min[ith][m].to_csv(pathOut+'perfscoresSwitzerland_{}_10min_{}.csv'.format(m,
            #                                       str(ith).replace('.','_')))

            # precipref_60min_array = precip60min[ref].to_numpy().ravel()
            # precipqpe_60min_array = precip60min[m].to_numpy().ravel()
            # valid = (~np.isnan(precipqpe_60min_array)) & (~np.isnan(precipref_60min_array))
            # perfscores_switzerland_60min[ith][m] = calc_perfscores_CH(precipref_60min_array[valid], precipqpe_60min_array[valid],th=ith)
            # perfscores_switzerland_60min[ith][m].to_csv(pathOut+'perfscoresSwitzerland_{}_60min_{}.csv'.format(m,
            #                                       str(ith).replace('.','_')))