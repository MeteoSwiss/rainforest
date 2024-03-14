#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module to 
"""

# Global imports
from audioop import cross
import logging
logging.getLogger().setLevel(logging.INFO)
import os
import pickle
import glob
import dask.dataframe as dd
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from scipy.stats import rankdata

# Local imports
from ..common import constants
from .utils import vert_aggregation, split_event, split_years
from .rfdefinitions import RandomForestRegressorBC
from ..common.utils import perfscores, envyaml
from ..common.graphics import plot_crossval_stats

dir_path = os.path.dirname(os.path.realpath(__file__))
FOLDER_MODELS = Path(os.environ['RAINFOREST_DATAPATH'], 'rf_models')

class RFTraining(object):
    '''
    This is the main class that allows to preparate data for random forest
    training, train random forests and perform cross-validation of trained models
    '''
    def __init__(self, db_location, input_location=None,
                 force_regenerate_input = False):
        """
        Initializes the class and if needed prepare input data for the training
        
        Note that when calling this constructor the input data is only 
        generated for the central pixel (NX = NY = 0 = loc of gauge), if you
        want to regenerate the inputs for all neighbour pixels, please 
        call the function self.prepare_input(only_center_pixel = False)
        
        Parameters
        ----------
        db_location : str
            Location of the main directory of the database (with subfolders
            'reference', 'gauge' and 'radar' on the filesystem)
        input_location : str
            Location of the prepared input data, if this data cannot be found
            in this folder, it will be computed here, default is a subfolder
            called rf_input_data within db_location
        force_regenerate_input : bool
            if True the input parquet files will always be regenerated from
            the database even if already present in the input_location folder
        """
        
        if input_location == None:
            input_location = str(Path(db_location, 'rf_input_data'))
            
        # Check if at least gauge.parquet, refer_x0y0.parquet and radar_x0y0.parquet
        # are present
        valid = True
        if not os.path.exists(input_location):
            valid = False
            os.makedirs(input_location)
        files = glob.glob(str(Path(input_location, '*')))
        files = [os.path.basename(f) for f in files]
        if ('gauge.parquet' not in files or 'reference_x0y0.parquet' not in files
            or 'radar_x0y0.parquet' not in files):
            valid = False
        
        self.input_location = input_location
        self.db_location = db_location
        
        if not valid :
            logging.info('Could not find valid input data from the folder {:s}'.format(input_location))
        # if force_regenerate_input or not valid:
        #     logging.info('The program will now compute this input data from the database, this takes quite some time')
        #     self.prepare_input()
    
    def prepare_input(self, only_center=True, foldername_radar='radar'):
        """
        Reads the data from the database  in db_location and processes it to 
        create easy to use parquet input files for the ML training and stores 
        them in the input_location, the processing steps involve
        
        For every neighbour of the station (i.e. from -1-1 to +1+1):
        
        -   Replace missing flags by nans
        -   Filter out timesteps which are not present in the three tables 
            (gauge, reference and radar)
        -   Filter out incomplete hours (i.e. where less than 6 10 min timesteps
            are available)
        -   Add height above ground and height of iso0 to radar data
        -   Save a separate parquet file for radar, gauge and reference data
        -   Save a grouping_idx pickle file containing *grp_vertical*
            index (groups all radar rows with same timestep and station),
            *grp_hourly* (groups all timesteps with same hours) and *tstamp_unique*
            (list of all unique timestamps)
        
        Parameters
        ----------
        only_center : bool
            If set to True only the input data for the central neighbour
            i.e. NX = NY = 0 (the location of the gauge) will be recomputed
            this takes much less time and is the default option since until
            now the neighbour values are not used in the training of the RF
            QPE
        foldername_radar: str
            Name of the folder to use for the radar data. Default name is 'radar'            
        """
        
        if not os.path.exists(Path(self.db_location, foldername_radar)):
            logging.error('Invalid foldername for radar data, please check')

        if only_center:
            nx = [0]
            ny = [0]
        else:
            nx = [0,1,-1]
            ny = [0,1,-1]
        gauge = dd.read_csv(str(Path(self.db_location, 'gauge', '*.csv.gz')), 
                            compression='gzip', 
                            assume_missing=True,
                            dtype = {'TIMESTAMP':int,  'STATION': str})
        
        gauge = gauge.compute().drop_duplicates()
        gauge = gauge.replace(-9999,np.nan)
        for x in nx:
            for y in ny:
                logging.info('Processing neighbour {:d}{:d}'.format(x, y))
                radar = dd.read_parquet(str(Path(self.db_location, foldername_radar,
                                                  '*.parquet')))
                refer = dd.read_parquet(str(Path(self.db_location, 'reference', 
                                                 '*.parquet')))
                        
                # Select only required pixel
                radar = radar.loc[np.logical_and(radar['NX'] == x, 
                                                  radar['NY'] == y)]
                refer = refer.loc[np.logical_and(refer['NX'] == x, 
                                                 refer['NY'] == y)]
                
                # Convert to pandas and remove duplicates 
                radar = radar.compute().drop_duplicates(subset = ['TIMESTAMP',
                                                                   'STATION',
                                                                   'RADAR',
                                                                   'NX','NY',
                                                                   'SWEEP'])
                
                refer = refer.compute().drop_duplicates(subset = ['TIMESTAMP',
                                                                  'STATION'])
                
                # Replace missing flags with nan
                radar = radar.replace(-9999, np.nan)
                refer = refer.replace(-9999, np.nan)

                # Sort values
                radar = radar.sort_values(by = ['TIMESTAMP','STATION','SWEEP'])
                refer = refer.sort_values(by = ['TIMESTAMP','STATION'])
                gauge = gauge.sort_values(by = ['TIMESTAMP','STATION'])

                # Get only valid precip data
                gauge = gauge[np.isfinite(gauge['RRE150Z0'])]
                
                # Create individual 10 min - station stamps
                gauge['s-tstamp'] = np.array(gauge['STATION'] + 
                                           gauge['TIMESTAMP'].astype(str)).astype(str)
                radar['s-tstamp'] = np.array(radar['STATION'] + 
                                            radar['TIMESTAMP'].astype(str)).astype(str)
                refer['s-tstamp'] = np.array(refer['STATION'] + 
                                           refer['TIMESTAMP'].astype(str)).astype(str)
                
                # Get gauge and reference only when radar data available
        
                # Find timestamps that are in the three datasets
                ststamp_common = np.array(pd.Series(list(set(gauge['s-tstamp'])
                                    .intersection(set(refer['s-tstamp'])))))
                ststamp_common = np.array(pd.Series(list(set(radar['s-tstamp'])
                                     .intersection(set(ststamp_common)))))
                radar = radar.loc[radar['s-tstamp'].isin(ststamp_common)]
                gauge = gauge.loc[gauge['s-tstamp'].isin(ststamp_common)]
                refer = refer.loc[refer['s-tstamp'].isin(ststamp_common)]
        
                # Filter incomplete hours
                stahour = np.array(gauge['STATION'] + 
                       ((gauge['TIMESTAMP'] - 600 ) - 
                         (gauge['TIMESTAMP'] - 600 ) % 3600).astype(str)).astype(str)
                  
                full_hours = np.array(gauge.groupby(stahour)['STATION']
                                        .transform('count') == 6)
               
                refer = refer[full_hours]
                gauge = gauge[full_hours]    
                radar = radar[radar['s-tstamp'].
                                isin(np.array(gauge['s-tstamp']))]
                
                stahour = stahour[full_hours]
                
                # Creating vertical grouping index
                
                _, idx, grp_vertical = np.unique(radar['s-tstamp'],
                                                 return_inverse = True,
                                                 return_index = True)
                # Get original order
                sta_tstamp_unique = radar['s-tstamp'].index[np.sort(idx)]
                # Preserves order and avoids sorting radar_statstamp
                grp_vertical = idx[grp_vertical]
                # However one issue is that the indexes are not starting from zero with increment
                # of one, though they are sorted, they are like 0,7,7,7,15,15,23,23
                # We want them starting from zero with step of one
                grp_vertical = rankdata(grp_vertical,method='dense') - 1
                
                # Repeat operation with gauge hours
                sta_hourly_unique, idx, grp_hourly = np.unique(stahour, 
                                                           return_inverse = True,
                                                           return_index = True)
                grp_hourly = idx[grp_hourly]
                
                # Add derived variables  height iso0 (HISO) and height above ground (HAG)
                # Radar
                stations = constants.METSTATIONS
                cols = list(stations.columns)
                cols[1] = 'STATION'
                stations.columns = cols
                radar = pd.merge(radar,stations, how = 'left', on = 'STATION',
                                 sort = False)
                
                if 'T' in radar.columns:
                    radar['HISO'] = -radar['T'] / constants.LAPSE_RATE * 100
                    radar['HAG'] = radar['HEIGHT'] - radar['Z']
                    radar['HAG'][radar['HAG'] < 0] = 0
        
                # Gauge
                gauge['minutes'] = (gauge['TIMESTAMP'] % 3600)/60
                
                # Save all to file
                # Save all to file
                logging.info('Saving files to {}'.format(self.input_location))
                refer.to_parquet(str(Path(self.input_location, 
                                          'reference_x{:d}y{:d}.parquet'.format(x,y))),
                                 compression = 'gzip', index = False)
                
                radar.to_parquet(str(Path(self.input_location, 
                                          'radar_x{:d}y{:d}.parquet'.format(x,y))),
                                 compression = 'gzip', index = False)
                
                grp_idx = {}
                grp_idx['grp_vertical'] = grp_vertical
                grp_idx['grp_hourly'] = grp_hourly
                grp_idx['tstamp_unique'] = sta_tstamp_unique
                
                pickle.dump(grp_idx, 
                    open(str(Path(self.input_location, 
                                  'grouping_idx_x{:d}y{:d}.p'.format(x,y))),'wb'))
                
                if x == 0 and y == 0:
                    # Save only gauge for center pixel since it's available only there
                    gauge.to_parquet(str(Path(self.input_location, 'gauge.parquet')),
                                 compression = 'gzip', index = False)
        
            
    def fit_models(self, config_file, features_dic, tstart = None, tend = None,
                   output_folder = None):
        """
        Fits a new RF model that can be used to compute QPE realizations and
        saves them to disk in pickle format
        
        Parameters
        ----------
        config_file : str
            Location of the RF training configuration file, if not provided 
            the default one in the ml submodule will be used       
        features_dic : dict
            A dictionary whose keys are the names of the models you want to
            create (a string) and the values are lists of features you want to
            use. For example {'RF_dualpol':['RADAR', 'zh_VISIB_mean',
            'zv_VISIB_mean','KDP_mean','RHOHV_mean','T', 'HEIGHT','VISIB_mean']}
            will train a model with all these features that will then be stored
            under the name RF_dualpol_BC_<type of BC>.p in the ml/rf_models dir
        tstart : datetime
            the starting time of the training time interval, default is to start
            at the beginning of the time interval covered by the database
        tend : datetime
            the end time of the training time interval, default is to end
            at the end of the time interval covered by the database   
        output_folder : str
            Location where to store the trained models in pickle format,
            if not provided it will store them in the standard location 
            <library_path>/ml/rf_models
        """
        
        if output_folder == None:
            output_folder =  str(Path(FOLDER_MODELS, 'rf_models'))
            
        try:
            config = envyaml(config_file)
        except:
            logging.warning('Using default config as no valid config file was provided')
            config_file = dir_path + '/default_config.yml'
            
        config = envyaml(config_file)
  
        #######################################################################
        # Read data
        #######################################################################
        
        logging.info('Loading input data')
        radartab = pd.read_parquet(str(Path(self.input_location, 'radar_x0y0.parquet')))
        gaugetab = pd.read_parquet(str(Path(self.input_location, 'gauge.parquet')))
        grp = pickle.load(open(str(Path(self.input_location, 'grouping_idx_x0y0.p')),'rb'))
        grp_vertical = grp['grp_vertical']
        
        ###############################################################################
        # Compute additional data if needed
        ###############################################################################
    
        # currently the only supported additional features is zh (refl in linear units)
        # and DIST_TO_RAD{A-D-L-W-P} (dist to individual radars)
        # Get list of unique features names
        features = np.unique([item for sub in list(features_dic.values())
                            for item in sub])

        for f in features:
            if 'zh' in f:
                logging.info('Computing derived variable {:s}'.format(f))
                radartab[f] = 10**(0.1 * radartab[f.replace('zh','ZH')])
            elif 'zv' in f:
                logging.info('Computing derived variable {:s}'.format(f))
                radartab[f] = 10**(0.1 * radartab[f.replace('zv','ZV')])        
            if 'DIST_TO_RAD' in f:
                info_radar = constants.RADARS
                vals = np.unique(radartab['RADAR'])
                for val in vals:
                    dist = np.sqrt((radartab['X'] - info_radar['X'][val])**2+
                           (radartab['Y'] - info_radar['Y'][val])**2) / 1000.
                    radartab['DIST_TO_RAD' + str(val)] = dist
                    
        ###############################################################################
        # Compute data filter for each model
        ###############################################################################

        for model in features_dic.keys():

            vweights = 10**(config[model]['VERT_AGG']['BETA'] * (radartab['HEIGHT']/1000.)) # vert. weights

            filterconf = config[model]['FILTERING'].copy()
            logging.info('Computing data filter')
            logging.info('List of stations to ignore {:s}'.format(','.join(filterconf['STA_TO_REMOVE'])))
            logging.info('Start time {:s}'.format(str(tstart)))
            logging.info('End time {:s}'.format(str(tend)))           
            logging.info('ZH must be > {:f} if R <= {:f}'.format(filterconf['CONSTRAINT_MIN_ZH'][1],
                                                filterconf['CONSTRAINT_MIN_ZH'][0]))   
            logging.info('ZH must be < {:f} if R <= {:f}'.format(filterconf['CONSTRAINT_MAX_ZH'][1],
                                                filterconf['CONSTRAINT_MAX_ZH'][0]))    

            ZH_agg = vert_aggregation(pd.DataFrame(radartab['ZH_mean']),
                                        vweights,
                                        grp_vertical,
                                        True, radartab['VISIB_mean'])
            cond1 = np.array(np.isin(gaugetab['STATION'], filterconf['STA_TO_REMOVE']))
            cond2 = np.logical_and(ZH_agg['ZH_mean'] < filterconf['CONSTRAINT_MIN_ZH'][1],
                6 * gaugetab['RRE150Z0'].values >= filterconf['CONSTRAINT_MIN_ZH'][0])
            cond3 = np.logical_and(ZH_agg['ZH_mean'] >  filterconf['CONSTRAINT_MAX_ZH'][1],
                6 * gaugetab['RRE150Z0'].values <=  filterconf['CONSTRAINT_MIN_ZH'][0])
            
            invalid = np.logical_or(cond1,cond2)
            invalid = np.logical_or(invalid,cond3)
            invalid = np.logical_or(invalid,cond3)
            invalid = np.array(invalid)

            if tend != None:
                tend_unix = (tend - datetime.datetime(1970,1,1) ).total_seconds()
                invalid[gaugetab['TIMESTAMP'] > tend_unix] = 1
            if tstart != None:
                tstart_unix = (tstart - datetime.datetime(1970,1,1) ).total_seconds()
                invalid[gaugetab['TIMESTAMP'] < tstart_unix] = 1
            invalid[np.isnan(gaugetab['RRE150Z0'])] = 1

            ###############################################################################
            # Prepare training dataset
            ###############################################################################
        
            gaugetab_train = gaugetab[~invalid].copy()
        
            logging.info('Performing vertical aggregation of input features for model {:s}'.format(model))                
            features_VERT_AGG = vert_aggregation(radartab[features_dic[model]], 
                                 vweights, grp_vertical,
                                 config[model]['VERT_AGG']['VISIB_WEIGHTING'],
                                 radartab['VISIB_mean'])
            features_VERT_AGG = features_VERT_AGG[~invalid]
                        
            ###############################################################################
            # Fit
            ###############################################################################
            # create name of variables used in the model
            features = []
            for f in features_VERT_AGG.columns:
                if '_max' in f:
                    f = f.replace('_max','')
                elif '_min' in f:
                    f = f.replace('_min','')
                elif '_mean' in f:
                    f = f.replace('_mean','')
                features.append(f)
            
            Y = np.array(gaugetab_train['RRE150Z0'] * 6)
            valid = np.all(np.isfinite(features_VERT_AGG),axis=1)

            # Add some metadata
            config[model]['FILTERING']['N_datapoints'] = len(Y[valid])
            config[model]['FILTERING']['GAUGE_min_10min_mm_h'] = np.nanmin(Y[valid])
            config[model]['FILTERING']['GAUGE_max_10min_mm_h'] = np.nanmax(Y[valid])
            config[model]['FILTERING']['GAUGE_median_10min_mm_h'] = np.nanmedian(Y[valid])

            config[model]['FILTERING']['TIME_START'] = np.nanmin(gaugetab['TIMESTAMP'][~invalid])
            config[model]['FILTERING']['TIME_END'] = np.nanmax(gaugetab['TIMESTAMP'][~invalid])

            config[model]['FILTERING']['STA_INCLUDED'] = gaugetab['STATION'][~invalid].unique()
            config[model]['FILTERING']['CREATED'] = datetime.datetime.utcnow().strftime('%d %b %Y %H:%M UTC')

            logging.info('')
            logging.info('Training model on gauge data')

            reg = RandomForestRegressorBC(degree = 1, 
                          bctype = config[model]['BIAS_CORR'],
                          visib_weighting = config[model]['VERT_AGG']['VISIB_WEIGHTING'],
                          variables = features,
                          beta = config[model]['VERT_AGG']['BETA'],
                          metadata = config[model]['FILTERING'],
                          **config[model]['RANDOMFOREST_REGRESSOR'])

            reg.fit(features_VERT_AGG[valid], Y[valid])
            
            out_name = str(Path(output_folder, '{:s}_BETA_{:2.1f}_BC_{:s}.p'.format(model, 
                                                  config[model]['VERT_AGG']['BETA'],
                                                  config[model]['BIAS_CORR'])))
            logging.info('Saving model to {:s}'.format(out_name))
            
            pickle.dump(reg, open(out_name, 'wb'))

    def feature_selection(self, features_dic, featuresel_configfile, 
                        output_folder, K=5, tstart=None, tend=None):
        """
        The relative importance of all available input vairables aggregated to 
        to the ground and to choose the most important ones, an approach 
        from Han et al. (2016) was adpated to for regression.
        See Wolfensberger et al. (2021) for further information.

        Parameters
        -----------
        features : dic
            A dictionnary with all eligible features to test
        feature_sel_config : str
            yaml file with setup
        output_folder : str
            Path to where to store the scores
        tstart: str (YYYYMMDDHHMM)
            A date to define a starting time for the input data
        tend: str (YYYYMMDDHHMM)
            A date to define the end of the input data
        K : int or None
            Number of splits in iterations do perform in the K fold cross-val
        """

        config = envyaml(featuresel_configfile)
        modelnames = list(features_dic.keys())

        #######################################################################
        # Read data
        #######################################################################
        logging.info('Reading input data from {}'.format(self.input_location))
        radartab = pd.read_parquet(str(Path(self.input_location, 'radar_x0y0.parquet')))
        gaugetab = pd.read_parquet(str(Path(self.input_location, 'gauge.parquet')))
        grp = pickle.load(open(str(Path(self.input_location, 'grouping_idx_x0y0.p')),'rb'))
        grp_vertical = grp['grp_vertical']
        grp_hourly = grp['grp_hourly']

        #######################################################################
        # Filter time
        #######################################################################
        if tstart != None:
            try:
                tstart = datetime.datetime.strptime(tstart,
                        '%Y%m%d%H%M').replace(tzinfo=datetime.timezone.utc).timestamp()
            except:
                tstart = gaugetab['TIMESTAMP'].min()
                logging.info('The format of tstart was wrong, taking the earliest date')
        if tend != None:
            try:
                tend = datetime.datetime.strptime(tend,
                        '%Y%m%d%H%M').replace(tzinfo=datetime.timezone.utc).timestamp()
            except:
                tend = gaugetab['TIMESTAMP'].max()
                logging.info('The format of tend was wrong, taking the earliest date')

        timevalid = gaugetab['TIMESTAMP'].copy().astype(bool)
        vertvalid = radartab['TIMESTAMP'].copy().astype(bool)

        if (tstart != None):
            timevalid[(gaugetab['TIMESTAMP'] < tstart)] = False
            vertvalid[(radartab['TIMESTAMP'] < tstart)] = False
        if (tend != None):
            timevalid[(gaugetab['TIMESTAMP'] > tend)] = False
            vertvalid[(radartab['TIMESTAMP'] > tend)] = False

        gaugetab = gaugetab[timevalid]
        grp_hourly = grp_hourly[timevalid]
        radartab = radartab[vertvalid]
        grp_vertical = grp_vertical[vertvalid]

        ###############################################################################
        # Compute additional data if needed
        ###############################################################################
        # currently the only supported additional features is zh (refl in linear units)
        # and DIST_TO_RAD{A-D-L-W-P} (dist to individual radars)
        # Get list of unique features names
        features = np.unique([item for sub in list(features_dic.values())
                            for item in sub])
        for f in features:
            if 'zh' in f:
                logging.info('Computing derived variable {:s}'.format(f))
                radartab[f] = 10**(0.1 * radartab[f.replace('zh','ZH')])
            elif 'zv' in f:
                logging.info('Computing derived variable {:s}'.format(f))
                radartab[f] = 10**(0.1 * radartab[f.replace('zv','ZV')])        
            if 'DIST_TO_RAD' in f:
                info_radar = constants.RADARS
                vals = np.unique(radartab['RADAR'])
                for val in vals:
                    dist = np.sqrt((radartab['X'] - info_radar['X'][val])**2+
                           (radartab['Y'] - info_radar['Y'][val])**2) / 1000.
                    radartab['DIST_TO_RAD' + str(val)] = dist

        ###############################################################################
        # Compute vertical aggregation
        ###############################################################################
        features_VERT_AGG = {}
        regressors = {}
        for im, model in enumerate(modelnames):
            logging.info('Performing vertical aggregation of input features for model {:s}'.format(model))            
          
            if (im > 0) and (config[model]['VERT_AGG']['BETA'] == config[modelnames[im-1]]['VERT_AGG']['BETA']) \
                    and (config[model]['VERT_AGG']['VISIB_WEIGHTING'] == config[modelnames[im-1]]['VERT_AGG']['VISIB_WEIGHTING']):
                logging.info('Model {} has same vertical aggregation settings as {}, hence just copy aggregated 2D fields'.format(model, modelnames[im-1]))
                features_VERT_AGG[model] = features_VERT_AGG[modelnames[im-1]].copy()
            else:
                vweights = 10**(config[model]['VERT_AGG']['BETA'] *
                                    (radartab['HEIGHT']/1000.)) # vert. weights
                features_VERT_AGG[model] = vert_aggregation(radartab[features_dic[model]], 
                                    vweights, grp_vertical,
                                    config[model]['VERT_AGG']['VISIB_WEIGHTING'],
                                    radartab['VISIB_mean'])

            regressors[model] = RandomForestRegressorBC(degree = 1, 
                        bctype = config[model]['BIAS_CORR'],
                        beta = config[model]['VERT_AGG']['BETA'],
                        variables = features_dic[model],
                        visib_weighting=config[model]['VERT_AGG']['VISIB_WEIGHTING'],
                        **config[model]['RANDOMFOREST_REGRESSOR'])
        
        # remove nans
        valid = np.all(np.isfinite(features_VERT_AGG[modelnames[0]]),
                       axis = 1)

        # if (tstart != None):
        #     valid[(gaugetab['TIMESTAMP'] < tstart)] = False
        # if (tend != None):
        #     valid[(gaugetab['TIMESTAMP'] > tend)] = False

        test_not_ok = False
        for iv, val in enumerate(radartab['s-tstamp'].groupby(grp_vertical).first()):
            if gaugetab['s-tstamp'][iv] != val:
                test_not_ok = True
                print(gaugetab['s-tstamp'][iv])
        if test_not_ok:
            logging.error('Time cut went wrong!!')

        for model in modelnames:
            features_VERT_AGG[model] = features_VERT_AGG[model][valid]
        
        gaugetab = gaugetab[valid]
        grp_hourly = grp_hourly[valid]
        
        # Get R, T and idx test/train
        R = np.array(gaugetab['RRE150Z0'] * 6) # Reference precip in mm/h
        R[np.isnan(R)] = 0

        ###############################################################################
        # Randomly split test/ train dataset
        ###############################################################################
        if (K != None):
            K = list(range(K))
        elif (K == None):
            logging.info('Cross validation with random events defined but not specified, applying 5-fold CV')
            K = list(range(5))

        idx_testtrain = split_event(gaugetab['TIMESTAMP'].values, len(K))

        ###############################################################################
        # Prepare score dictionnary
        ###############################################################################
        scores = {}
        for tagg in ['10min', '60min']:
            scores[tagg] = {}
            for model in modelnames:
                scores[tagg][model] = {}
                for feat in features_VERT_AGG[model].keys():
                    scores[tagg][model][feat] = []

        for k in K:
            logging.info('Run {:d}/{:d}-{:d} of cross-validation'.format(k,np.nanmin(K),np.nanmax(K)))

            test = idx_testtrain == k
            train = idx_testtrain != k

            for model in modelnames:
                # Model fit always based on 10min values
                regressors[model].fit(features_VERT_AGG[model][train],R[train])
                R_pred_10 = regressors[model].predict(features_VERT_AGG[model][test])

                # Get reference RMSE
                rmse_ref = perfscores(R_pred_10, R[test], bounds=None)['all']['RMSE']

                # At hourly values
                R_test_60 = np.squeeze(np.array(pd.DataFrame(R[test])
                            .groupby(grp_hourly[test]).mean()))
                R_pred_60 = np.squeeze(np.array(pd.DataFrame(R_pred_10)
                            .groupby(grp_hourly[test]).mean()))
                rmse_ref_60 = perfscores(R_pred_60, R_test_60, bounds=None)['all']['RMSE']

                for feat in features_VERT_AGG[model].keys():
                    logging.info('Shuffling feature: {}'.format(feat))
                    # Shuffle input feature on test fraction, keep others untouched
                    x_test = features_VERT_AGG[model][test].copy()
                    x_test[feat] = np.random.permutation(x_test[feat].values)

                    # Calculate estimates and shuffled RMSE
                    R_pred_shuffled = regressors[model].predict(x_test)

                    #Compute increase in RMSE score at 10min
                    rmse_shuff = perfscores(R_pred_shuffled, R[test], bounds=None)['all']['RMSE']
                    scores['10min'][model][feat].append((rmse_shuff - rmse_ref) / rmse_ref)
                    
                    #Compute increase in RMSE score at 60min
                    R_pred_shuffled_60 = np.squeeze(np.array(pd.DataFrame(R_pred_shuffled)
                                .groupby(grp_hourly[test]).mean()))
                    rmse_shuff_60 = perfscores(R_pred_shuffled_60, R_test_60, bounds=None)['all']['RMSE']
                    scores['60min'][model][feat].append((rmse_shuff_60- rmse_ref_60) / rmse_ref_60)

        # Save all output
        name_file = str(Path(output_folder, 'feature_selection_scores.p'))
        pickle.dump(scores, open(name_file, 'wb'))
             
    def model_intercomparison(self, features_dic, intercomparison_configfile, 
                              output_folder, reference_products = ['CPCH','RZC'],
                              bounds10 = [0,2,10,100], bounds60 = [0,2,10,100],
                              cross_val_type='years', K=5, years=None,
                              tstart=None, tend=None, station_scores=False,
                              save_model=False):
        """
        Does an intercomparison (cross-validation) of different RF models and
        reference products (RZC, CPC, ...) and plots the performance plots
        
        Parameters
        ----------
        features_dic : dict
            A dictionary whose keys are the names of the models you want to
            compare (a string) and the values are lists of features you want to
            use. For example {'RF_dualpol':['RADAR', 'zh_VISIB_mean',
            'zv_VISIB_mean','KDP_mean','RHOHV_mean','T', 'HEIGHT','VISIB_mean'],
            'RF_hpol':['RADAR', 'zh_VISIB_mean','T', 'HEIGHT','VISIB_mean']}
            will compare a model of RF with polarimetric info to a model
            with only horizontal polarization
        output_folder : str
            Location where to store the output plots
        intercomparison_config : str
            Location of the intercomparison configuration file, which
            is a yaml file that gives for every model key of features_dic which
            parameters of the training you want to use (see the file 
            intercomparison_config_example.yml in this module for an example)
        reference_products : list of str
            Name of the reference products to which the RF will be compared
            they need to be in the reference table of the database
        bounds10 : list of float
            list of precipitation bounds for which to compute scores separately
            at 10 min time resolution
            [0,2,10,100] will give scores in range [0-2], [2-10] and [10-100]
        bounds60 : list of float
            list of precipitation bounds for which to compute scores separately
            at hourly time resolution
            [0,1,10,100] will give scores in range [0-1], [1-10] and [10-100]
        cross_val_type: str
            Define how the split of events is done. Options are "random events", 
            "years" and "seasons" (TODO)
        K : int or None
            Number of splits in iterations do perform in the K fold cross-val
        years : list or None
            List with the years that should be used in cross validation
            Default is [2016,2017,2018,2019,2020,2021]
        tstart: str (YYYYMMDDHHMM)
            A date to define a starting time for the input data
        tend: str (YYYYMMDDHHMM)
            A date to define the end of the input data
        station_scores: True or False (Boolean)
            If True, performance scores for all stations will be calculated
            If False, only the scores across Switzerland are calculated
        save_model: True or False (Boolean)
            If True, all models of the cross-validation are saved into a pickle file
            This is useful for reproducibility
        """
        
        # dict of statistics to compute for every score over the K-fold crossval,
        stats =  {'mean': np.nanmean, 'std': np.nanstd, 'min': np.nanmin, 'max': np.nanmax}
        
        config = envyaml(intercomparison_configfile)
        
        modelnames = list(features_dic.keys())
        keysconfig = list(config.keys())
        
        if not all([m in keysconfig for m in modelnames]):
            raise ValueError('Keys in features_dic are not all present in intercomparison config file!')
  
        if (cross_val_type == 'years') and (years == None):
            logging.info('Cross validation years defined, but not specified, years from 2016-2021 used')
            K = list(range(2016,2022,1))
        elif (cross_val_type == 'years') and (years != None):
            K = years

        if (cross_val_type == 'random events') and (K != None):
            K = list(range(K))
        elif (cross_val_type == 'random events') and (K == None):
            logging.info('Cross validation with random events defined but not specified, applying 5-fold CV')
            K = list(range(5))

        #######################################################################
        # Read data
        #######################################################################
        logging.info('Reading input data from {}'.format(self.input_location))
        radartab = pd.read_parquet(str(Path(self.input_location, 'radar_x0y0.parquet')))
        refertab = pd.read_parquet(str(Path(self.input_location, 'reference_x0y0.parquet')))
        gaugetab = pd.read_parquet(str(Path(self.input_location, 'gauge.parquet')))
        grp = pickle.load(open(str(Path(self.input_location, 'grouping_idx_x0y0.p')),'rb'))
        grp_vertical = grp['grp_vertical']
        grp_hourly = grp['grp_hourly']

        #######################################################################
        # Filter time
        #######################################################################
        if tstart != None:
            try:
                tstart = datetime.datetime.strptime(tstart,
                        '%Y%m%d%H%M').replace(tzinfo=datetime.timezone.utc).timestamp()
            except:
                tstart = gaugetab['TIMESTAMP'].min()
                logging.info('The format of tstart was wrong, taking the earliest date')
        if tend != None:
            try:
                tend = datetime.datetime.strptime(tend,
                        '%Y%m%d%H%M').replace(tzinfo=datetime.timezone.utc).timestamp()
            except:
                tend = gaugetab['TIMESTAMP'].max()
                logging.info('The format of tend was wrong, taking the earliest date')

        ###############################################################################
        # Compute additional data if needed
        ###############################################################################
        # currently the only supported additional features is zh (refl in linear units)
        # and DIST_TO_RAD{A-D-L-W-P} (dist to individual radars)
        # Get list of unique features names
        features = np.unique([item for sub in list(features_dic.values())
                            for item in sub])
        for f in features:
            if 'zh' in f:
                logging.info('Computing derived variable {:s}'.format(f))
                radartab[f] = 10**(0.1 * radartab[f.replace('zh','ZH')])
            elif 'zv' in f:
                logging.info('Computing derived variable {:s}'.format(f))
                radartab[f] = 10**(0.1 * radartab[f.replace('zv','ZV')])        
            if 'DIST_TO_RAD' in f:
                info_radar = constants.RADARS
                vals = np.unique(radartab['RADAR'])
                for val in vals:
                    dist = np.sqrt((radartab['X'] - info_radar['X'][val])**2+
                           (radartab['Y'] - info_radar['Y'][val])**2) / 1000.
                    radartab['DIST_TO_RAD' + str(val)] = dist
                    
        ###############################################################################
        # Compute vertical aggregation
        ###############################################################################
        features_VERT_AGG = {}
        regressors = {}
        for model in modelnames:
            logging.info('Performing vertical aggregation of input features for model {:s}'.format(model))            
          
            vweights = 10**(config[model]['VERT_AGG']['BETA'] *
                                (radartab['HEIGHT']/1000.)) # vert. weights
            features_VERT_AGG[model] = vert_aggregation(radartab[features_dic[model]], 
                                 vweights, grp_vertical,
                                 config[model]['VERT_AGG']['VISIB_WEIGHTING'],
                                 radartab['VISIB_mean'])
                  
            regressors[model] = RandomForestRegressorBC(degree = 1, 
                          bctype = config[model]['BIAS_CORR'],
                          variables = features_dic[model],
                          beta = config[model]['VERT_AGG']['BETA'],
                          visib_weighting=config[model]['VERT_AGG']['VISIB_WEIGHTING'],
                          **config[model]['RANDOMFOREST_REGRESSOR'])
        
        # remove nans
        valid = np.all(np.isfinite(features_VERT_AGG[modelnames[0]]),
                       axis = 1)
        # if (tstart != None) and (tend == None):
        #     (gaugetab['TIMESTAMP'] >= tstart)
        # elif (tstart == None) and (tend != None):
        #     timeperiod = (gaugetab['TIMESTAMP'] <= tend)
        # elif (tstart != None) and (tend != None):
        #     timeperiod = (gaugetab['TIMESTAMP'] >= tstart) & (gaugetab['TIMESTAMP'] <= tend)
        # else:
        #     timeperiod = valid
        if (tstart != None):
            valid[(gaugetab['TIMESTAMP'] < tstart)] = False
        if (tend != None):
            valid[(gaugetab['TIMESTAMP'] > tend)] = False

        for model in modelnames:
            features_VERT_AGG[model] = features_VERT_AGG[model][valid]
        
        gaugetab = gaugetab[valid]
        refertab = refertab[valid]
        grp_hourly = grp_hourly[valid]
        
        # Get R, T and idx test/train
        R = np.array(gaugetab['RRE150Z0'] * 6) # Reference precip in mm/h
        R[np.isnan(R)] = 0
        
        T = np.array(gaugetab['TRE200S0'])  # Reference temp in degrees
        # features must have the same size as gauge

        if cross_val_type == 'random_events':
            idx_testtrain = split_event(gaugetab['TIMESTAMP'].values, len(K))
        elif cross_val_type == 'years':
            idx_testtrain = split_years(gaugetab['TIMESTAMP'].values, years=K)
        else:
            logging.error('Please define your cross validation separation')


        modelnames.extend(reference_products)

        all_scores = {'10min':{},'60min':{}}
        all_stats = {'10min':{},'60min':{}}
        
        if station_scores == True:
            all_station_scores = {'10min': {}, '60min': {}}
            all_station_stats = {'10min': {}, '60min': {}}

        ###############################################################################
        # Initialize outputs
        ###############################################################################
        for model in modelnames:
            all_scores['10min'][model] = {'train': {'solid':[],'liquid':[],'all':[]},
                         'test':  {'solid':[],'liquid':[],'all':[]}}
            all_scores['60min'][model] = {'train': {'solid':[],'liquid':[],'all':[]},
                         'test':  {'solid':[],'liquid':[],'all':[]}}
            
            all_stats['10min'][model] = {'train': {'solid':{},'liquid':{},'all':{}},
                         'test':  {'solid':{},'liquid':{},'all':{}}}
            
            all_stats['60min'][model] = {'train': {'solid':{},'liquid':{},'all':{}},
                         'test':  {'solid':{},'liquid':{},'all':{}}}
            
            if station_scores == True:
                # for station scores we will limit the output to test data only
                for timeagg in all_station_scores.keys():
                    all_station_scores[timeagg][model] = {'solid':{},'liquid':{},'all':{}}
                    all_station_stats[timeagg][model] = {'solid':{},'liquid':{},'all':{}}
            

        for k in K:
            logging.info('Run {:d}/{:d}-{:d} of cross-validation'.format(k,np.nanmin(K),np.nanmax(K)))

            test = idx_testtrain == k
            train = idx_testtrain != k
            
            if cross_val_type == 'years':
                logging.info('Time range for testing set: {} - {} with {:3.2f}% of datapoints'.format(
                                datetime.datetime.utcfromtimestamp(gaugetab['TIMESTAMP'][test].min()),
                                datetime.datetime.utcfromtimestamp(gaugetab['TIMESTAMP'][test].max()),
                                gaugetab['TIMESTAMP'][test].count()/ gaugetab['TIMESTAMP'].count()*100))

            # Get reference values
            R_test_60 = np.squeeze(np.array(pd.DataFrame(R[test])
                            .groupby(grp_hourly[test]).mean()))
    
            R_train_60 = np.squeeze(np.array(pd.DataFrame(R[train])
                            .groupby(grp_hourly[train]).mean()))
            
            T_test_60 = np.squeeze(np.array(pd.DataFrame(T[test])
                            .groupby(grp_hourly[test]).mean()))
            
            T_train_60 = np.squeeze(np.array(pd.DataFrame(T[train])
                            .groupby(grp_hourly[train]).mean()))
            
            
            liq_10_train = T[train] >= constants.THRESHOLD_SOLID
            sol_10_train = T[train] < constants.THRESHOLD_SOLID
            liq_60_train = T_train_60 >= constants.THRESHOLD_SOLID
            sol_60_train = T_train_60 < constants.THRESHOLD_SOLID
    
            liq_10_test = T[test] >= constants.THRESHOLD_SOLID
            sol_10_test = T[test] < constants.THRESHOLD_SOLID
            liq_60_test = T_test_60 >= constants.THRESHOLD_SOLID
            sol_60_test = T_test_60 < constants.THRESHOLD_SOLID
            
            # Fit every regression model
            for model in modelnames:
                logging.info('Checking model {:s}'.format(model))
                logging.info('Evaluating test error')
                # 10 min
                logging.info('at 10 min')

                # Performing fit
                if model not in reference_products:
                    logging.info('Training model on gauge data')

                    regressors[model].fit(features_VERT_AGG[model][train],R[train])
                    R_pred_10 = regressors[model].predict(features_VERT_AGG[model][test])
    
                    if (save_model == True):
                        regressors[model].variables = features_VERT_AGG[model].columns
                        out_name = str(Path(output_folder, '{:s}_BETA_{:2.1f}_BC_{:s}_excl_{}.p'.format(model, 
                                                            config[model]['VERT_AGG']['BETA'],
                                                            config[model]['BIAS_CORR'],k)))
                        logging.info('Saving model to {:s}'.format(out_name))
                        pickle.dump(regressors[model], open(out_name, 'wb'))

                else:
                    R_pred_10 = refertab[model].values[test]
                

                scores_solid = perfscores(R_pred_10[sol_10_test],
                                          R[test][sol_10_test],
                                          bounds = bounds10)
               
                all_scores['10min'][model]['test']['solid'].append(scores_solid)
                
                scores_liquid = perfscores(R_pred_10[liq_10_test],
                                                 R[test][liq_10_test],
                                                 bounds = bounds10)
                all_scores['10min'][model]['test']['liquid'].append(scores_liquid)
                
                scores_all = perfscores(R_pred_10,
                                                 R[test],
                                                 bounds = bounds10)
                all_scores['10min'][model]['test']['all'].append(scores_all)
    
                # 60 min
                logging.info('at 60 min')
                R_pred_60 = np.squeeze(np.array(pd.DataFrame(R_pred_10)
                                    .groupby(grp_hourly[test]).mean()))
    
                scores_solid = perfscores(R_pred_60[sol_60_test],
                                                 R_test_60[sol_60_test],
                                                 bounds = bounds60)
                all_scores['60min'][model]['test']['solid'].append(scores_solid)
                            
                
                scores_liquid = perfscores(R_pred_60[liq_60_test],
                                                 R_test_60[liq_60_test],
                                                 bounds = bounds60)
                all_scores['60min'][model]['test']['liquid'].append(scores_liquid)
                
                scores_all = perfscores(R_pred_60,
                                                 R_test_60,
                                                 bounds = bounds60)
                all_scores['60min'][model]['test']['all'].append(scores_all)
                
                if station_scores == True:
                    logging.info('Calculating station performances for model {}'.format(model)) 
                    
                    stations_60 = np.array(gaugetab['STATION'][test]
                                    .groupby(grp_hourly[test]).first())

                    df = pd.DataFrame(columns=gaugetab['STATION'].unique(),      
                    index = scores_all['all'].keys())

                    for timeagg in all_station_scores.keys():
                        all_station_scores[timeagg][model]['all'][k] = df.copy()
                        all_station_scores[timeagg][model]['liquid'][k] = df.copy()
                        all_station_scores[timeagg][model]['solid'][k] = df.copy()
                    
                    for sta in gaugetab['STATION'].unique():
                        sta_idx = (gaugetab['STATION'][test] == sta)
                        sta_idx_60 = (stations_60 == sta)

                        try:                            
                            scores_all_10 = perfscores(R_pred_10[sta_idx],
                                                    R[test][sta_idx])['all']
                            all_station_scores['10min'][model]['all'][k][sta] = scores_all_10
                            
                            scores_all_60 = perfscores(R_pred_60[sta_idx_60],R_test_60[sta_idx_60])['all']
                            all_station_scores['60min'][model]['all'][k][sta] = scores_all_60
                            
                            del scores_all_10, scores_all_60
                        except:
                            logging.info('No performance score for {}'.format(sta))
                        try:                            
                            scores_liquid_10 = perfscores(R_pred_10[liq_10_test & sta_idx],
                                                    R[test][liq_10_test & sta_idx])['all']
                            all_station_scores['10min'][model]['liquid'][k][sta] = scores_liquid_10
                            
                            scores_liquid_60 = perfscores(R_pred_60[liq_60_test & sta_idx_60],
                                                    R_test_60[liq_60_test & sta_idx_60])['all']
                            all_station_scores['60min'][model]['liquid'][k][sta] = scores_liquid_60  
                        except:
                            logging.info('No performance score for liquid precip for {}'.format(sta))
                        try:                            
                            scores_solid_10 = perfscores(R_pred_10[sol_10_test & sta_idx],
                                                    R[test][sol_10_test & sta_idx])['all']  
                            all_station_scores['10min'][model]['solid'][k][sta] = scores_solid_10 
                            
                            scores_solid_60 = perfscores(R_pred_60[sol_60_test & sta_idx_60],
                                                        R_test_60[sol_60_test & sta_idx_60])['all']
                            all_station_scores['60min'][model]['solid'][k][sta] = scores_solid_60
                        except:
                            logging.info('No performance score for solid precip for {}'.format(sta))

                # train
                logging.info('Evaluating train error')
                # 10 min
                logging.info('at 10 min')
                
                if model not in reference_products:
                    R_pred_10 = regressors[model].predict(features_VERT_AGG[model][train])
                else:
                    R_pred_10 = refertab[model].values[train]
                    
                scores_solid = perfscores(R_pred_10[sol_10_train],
                                                 R[train][sol_10_train],
                                                 bounds = bounds10)
                all_scores['10min'][model]['train']['solid'].append(scores_solid)
                
                scores_liquid = perfscores(R_pred_10[liq_10_train],
                                                 R[train][liq_10_train],
                                                 bounds = bounds10)
                
                all_scores['10min'][model]['train']['liquid'].append(scores_liquid)
                
                scores_all = perfscores(R_pred_10,
                                                 R[train],
                                                 bounds = bounds10)
                all_scores['10min'][model]['train']['all'].append(scores_all)
                
                
                R_pred_60 = np.squeeze(np.array(pd.DataFrame(R_pred_10)
                                    .groupby(grp_hourly[train]).mean()))
                
                # 60 min
                logging.info('at 60 min')
                # Evaluate model 10 min
    
                scores_solid = perfscores(R_pred_60[sol_60_train],
                                        R_train_60[sol_60_train],
                                        bounds = bounds60)
                all_scores['60min'][model]['train']['solid'].append(scores_solid)
                
                scores_liquid = perfscores(R_pred_60[liq_60_train],
                                        R_train_60[liq_60_train],
                                         bounds = bounds60)
                all_scores['60min'][model]['train']['liquid'].append(scores_liquid)
                
                scores_all = perfscores(R_pred_60,R_train_60,
                                        bounds = bounds60)
                all_scores['60min'][model]['train']['all'].append(scores_all)


        # Compute statistics after the 5-fold cross validation
        for agg in all_scores.keys():
            for model in all_scores[agg].keys():
                for veriftype in all_scores[agg][model].keys():
                    for preciptype in all_scores[agg][model][veriftype].keys():
                        bounds = list(all_scores[agg][model][veriftype][preciptype][0].keys())
                        scores =  all_scores[agg][model][veriftype][preciptype][0][bounds[0]].keys()
                        for bound in bounds:
                            all_stats[agg][model][veriftype][preciptype][bound] = {}
                            for score in scores:
                                data = all_scores[agg][model][veriftype][preciptype]
                                for d in data:
                                    if type(d[bound]) != dict:
                                        d[bound] = {'ME':np.nan,
                                                    'CORR':np.nan,
                                                    'STDE':np.nan,
                                                    'MAE':np.nan,
                                                    'scatter':np.nan,
                                                    'bias':np.nan,
                                                    'ED':np.nan}
                                datasc = [d[bound][score] for d in data]
                                all_stats[agg][model][veriftype][preciptype][bound][score] = {}
                                
                                for stat in stats.keys():
                                    sdata = stats[stat](datasc)
                                    all_stats[agg][model][veriftype][preciptype][bound][score][stat] = sdata

        if station_scores == True:
            for agg in all_station_scores.keys():
                for model in all_station_scores[agg].keys():
                    for preciptype in all_station_scores[agg][model].keys():
                        df = pd.DataFrame(columns=gaugetab['STATION'].unique())
                        perfs = {'RMSE': df.copy(), 'scatter':df.copy(), 
                                 'logBias':df.copy(), 'ED':df.copy(), 'N':df.copy(),
                                 'mest':df.copy(), 'mref':df.copy()}
                        all_station_stats[agg][model][preciptype] = {}
                        for score in perfs.keys():
                            for kidx in all_station_scores[agg][model][preciptype].keys():
                                df_dummy = all_station_scores[agg][model][preciptype][kidx].copy()
                                perfs[score] = perfs[score].append(df_dummy.loc[df_dummy.index == score])
            
                            all_station_stats[agg][model][preciptype][score] = pd.concat([perfs[score].mean().rename('mean'),
                                                                                          perfs[score].std().rename('std')],
                                                                                         axis=1)


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plot_crossval_stats(all_stats, output_folder)
        name_file = str(Path(output_folder, 'all_scores.p'))
        pickle.dump(all_scores, open(name_file, 'wb'))
        name_file = str(Path(output_folder, 'all_scores_stats.p'))
        pickle.dump(all_stats, open(name_file, 'wb'))     
        
        if station_scores == True:
            name_file = str(Path(output_folder, 'all_station_scores.p'))
            pickle.dump(all_station_scores, open(name_file, 'wb'))
            name_file = str(Path(output_folder, 'all_station_stats.p'))
            pickle.dump(all_station_stats, open(name_file, 'wb'))
        
        
        logging.info('Finished script and saved all scores to {}'.format(output_folder))
        return all_scores, all_stats
            
