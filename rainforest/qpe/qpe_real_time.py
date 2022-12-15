#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function to compute the QPE estimations on the Swiss grid

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019

Modified by D. Wolfensberger and R. Gugerli
December 2022
"""


import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import datetime
import glob
from pathlib import Path
import os
import fnmatch
import re
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates

from pyart.testing import make_empty_grid
from pyart.aux_io.odim_h5_writer import write_odim_grid_h5
from pyart.aux_io.odim_h5 import proj4_to_dict

logging.getLogger().setLevel(logging.INFO)

from ..common import constants
from ..common.retrieve_data import retrieve_prod, get_COSMO_T, retrieve_hzt_prod
from ..common.lookup import get_lookup
from ..common.utils import split_by_time, nanadd_at, envyaml
from ..common.radarprocessing import Radar, HZT_hourly_to_5min
from ..common.io_data import save_gif

try:
    import pysteps
    _PYSTEPS_AVAILABLE = True
except ImportError:
    _PYSTEPS_AVAILABLE = False

###############################################################################
# Centerpoints of all QPE grid cells
Y_QPE_CENTERS = constants.Y_QPE_CENTERS
X_QPE_CENTERS = constants.X_QPE_CENTERS

NBINS_X = len(X_QPE_CENTERS)
NBINS_Y = len(Y_QPE_CENTERS)

dir_path = os.path.dirname(os.path.realpath(__file__))

def _qpe_to_chgrid(qpe, time, radar_list, precision=2):
    """
    Creates a pyart grid object from a QPE array

    Parameters
    ----------
    qpe : ndarray
        2D numpy array containing the QPE data in the Swiss QPE grid
    time : datetime
        Start time of the scan
    precision : int
        Precision to use when storing the QPE data in the grid, default is 2
        (0.01)
    quality : dictionnary
        Containing all radars with corresponding timestamps that are missing
        

    Returns
    -------
    A pyart Grid object
    """


    grid = make_empty_grid([1, NBINS_X, NBINS_Y], [[0,0],
                                           [1000 * np.min(X_QPE_CENTERS),
                                            1000 * np.max(X_QPE_CENTERS)],
                                           [1000 * np.min(Y_QPE_CENTERS),
                                            1000 * np.max(Y_QPE_CENTERS)]])


    time_start = time - datetime.timedelta(seconds = 5 * 60)
    grid.time['units'] = 'seconds since {:s}'.format(
                    datetime.datetime.strftime(time_start,
                                               '%Y-%m-%dT%H:%M:%SZ'))
    grid.time['data'] = np.arange(0, 5 *60)
    grid.origin_latitude['data'] = 46.9524
    grid.origin_longitude['data'] = 7.43958333
    grid.projection = proj4_to_dict("+proj=somerc +lat_0=46.95240555555556 "+\
        "+lon_0=7.439583333333333 +k_0=1 +x_0=600000 +y_0=200000"+\
            " +ellps=bessel +towgs84=674.4,15.1,405.3,0,0,0,0 +units=m +no_defs")
    data = {}
    data['data'] = np.around(qpe, precision)
    data['units'] = 'mm/hr'
    data['long_name'] = 'Rainforest estimated rain rate'
    data['coordinates'] = 'elevation azimuth range'
    data['product'] = b'RR'
    data['prodname'] = b'CHRFQ'
    data['nodata'] = np.nan
    data['_FillValue'] = np.nan

    grid.fields['radar_estimated_rain_rate'] = data
    grid.metadata['source'] = b'ORG:215, CTY:644, CMT:MeteoSwiss (Switzerland)'
    grid.metadata['version'] = b'H5rad 2.3'
    # Add missing radar information
    quality = 'ADLPW'
    if len(radar_list) != 0:
        rad_list = list(radar_list.keys())
        qual_new = quality
        for rad in rad_list:
            qual_new = qual_new.replace(rad, '-')
        quality = qual_new
    grid.metadata['radar'] = quality.encode()

    return grid


def _outlier_removal(image, N = 3, threshold = 3):
    """
    Performs localized outlier correction by standardizing the data in a moving
    window and remove values that are below - threshold or above + threshold

    Parameters
    ----------
    image : ndarray
        2D numpy array
    N : int
        size of the moving window, for both rows and columns ( the window is
        square)
    threshold : threshold for a standardized value to be considered an outlier

    Returns
    -------
    An outlier removed version of the image with the same shape
    """

    im = np.array(image, dtype=float)
    im2 = im**2

    im_copy = im.copy()
    ones = np.ones(im.shape)

    kernel = np.ones((2*N+1, 2*N+1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")

    mean = (s/ns)
    std = (np.sqrt((s2 - s**2 / ns) / ns))

    z = (image - mean)/std
    im_copy[z >= threshold] = mean[z >= threshold]
    return im_copy

def _disaggregate(R, T = 5, t = 1,):
    """
    Disaggregates a set of two consecutive QPE images to 1 min resolution and
    then averages them to get a new advection corrected QPE estimates

    Parameters
    ----------
    R : list
        List of two numpy 2D arrays, containing the previous and the current
        QPE estimate
    T : int
        The time interval that separates the two QPE images, default is 5 min
    t : int
        The reference time interval used for the disaggregation, 1 min by
        default, should not be touched I think

    Returns
    -------
    An advection corrected QPE estimate

    """
    x,y = np.meshgrid(np.arange(R[0].shape[1],dtype=float),
                  np.arange(R[0].shape[0],dtype=float))
    oflow_method = pysteps.motion.get_method("LK")
    V1 = oflow_method(np.log(R))
    Rd = np.zeros((R[0].shape))

    for i in range(1 + int(T/t)):

        pos1 = (y - i/T * V1[1],x - i/T * V1[0])
        R1 = map_coordinates(R[0],pos1, order = 1)

        pos2 = (y + (T-i)/T * V1[1],x + (T-i)/T * V1[0])
        R2 = map_coordinates(R[1],pos2, order = 1)

        Rd += (T-i) * R1 + i * R2
    return 1/T**2 * Rd


class QPEProcessor_RT(object):
    def __init__(self, config_file, models):
        """
        Creates a QPEProcessor object which can be used to compute QPE
        realizations with the RandomForest Regressors

        Parameters
        ----------
        config : str
            A yaml file containing all necessary options for the QPE algorithm
            check the default_config.yml file to see which keys are required
        models : dict
            A dictionary containing all RF models to use for prediction,
            keys in this dictionary are used to store outputs in separate
            folders whereas the values must be valid RF regressor instances
            as stored in the rf_models subfolder
        """

        try:
            config = envyaml(config_file)
        except:
            logging.warning('Using default config as no valid config file was provided')
            config_file = dir_path + '/default_config.yml'

        config = envyaml(config_file)

        self.config = config

        self.models = models

        if self.config['RADARS'] == 'all':
            self.config['RADARS'] = list(constants.RADARS.Abbrev)

        if self.config['SWEEPS'] == 'all':
            self.config['SWEEPS'] = list(range(1,21))

        # Precompute cart. lookup tables and radar heights
        self.lut_cart = {}
        self.rad_heights = {}

        for rad in self.config['RADARS']:
            self.lut_cart[rad] = np.array(get_lookup('qpegrid_to_rad',
                         radar = rad))
            self.rad_heights[rad] = {}
            coords = get_lookup('cartcoords_rad', rad)
            for sweep in self.config['SWEEPS']:
                self.rad_heights[rad][sweep] = coords[sweep][2]

        self.model_weights_per_var = {}
        self.cosmo_var = []
        # keys of this dict are the variable used for the RF models, their values
        # is a list of lists with 1. the vertical weighting beta parameters to be used, 2. if visib_weighting is used or not
        for k in self.models.keys():
            for var in self.models[k].variables:
                if var not in self.model_weights_per_var.keys():
                    self.model_weights_per_var[var] = []
                if models[k].beta not in self.model_weights_per_var[var]:
                    self.model_weights_per_var[var].append((models[k].beta, models[k].visib_weighting))
                if (var == 'T') or (var == 'ISO0_HEIGHT'):
                    self.cosmo_var.append(var)


    def _retrieve_prod_RT(self, time, product_name, 
                          pattern = None, pattern_type = 'shell', sweeps = None):
        """ Adapted function from rainforest.common.retrieve_data
            Here, it reads the data per timestep, and in the real-time
            operation, the radar data is not zipped

        Args:
            time (datetime object): timestamp to extract
            product_name (string): Name of the product to be extracted
            sweeps (list): List of sweeps if not all want to be extracted. Defaults to None.

        Raises:
            ValueError: If no data is found
            
        Returns:
            dict: dictionary containing with the the file list
        """
       
        # Get all files
        folder_radar = constants.FOLDER_RADAR
        folder_in = folder_radar + product_name + '/'

        # Get list of available files
        content_zip = np.array(os.listdir(folder_in))
        times_zip = np.array([datetime.datetime.strptime(c[3:12],
                            '%y%j%H%M') for c in content_zip])
    
        if pattern != None:
            if pattern_type == 'shell':
                content_zip = [c for c in content_zip 
                            if fnmatch.fnmatch(os.path.basename(c), pattern)]
            elif pattern_type == 'regex':
                content_zip = [c for c in content_zip 
                            if re.match(os.path.basename(c), pattern) != None]
            else:
                raise ValueError('Unknown pattern_type, must be either "shell" or "regex".')
    
        # Get a list of all files to retrieve
        conditions = (times_zip == time)
        
        # Filter on sweeps:
        if sweeps != None:
            sweeps_zip = np.array([int(c[-3:]) for c in content_zip])
                # Get a list of all files to retrieve
            conditions_sweep = np.array([s in sweeps for s in sweeps_zip])
            conditions = np.logical_and(conditions, conditions_sweep)

        if not np.any(conditions):
            msg = '''
            No file was found corresponding to this format, verify pattern and product_name
            '''
            raise ValueError(msg)
        
        files = sorted(np.array([folder_in + c for c in
                                np.array(content_zip)[conditions]]))

        return files    

    def _retrieve_hzt_RT(self, tstep):
        
        """ Retrieves the preprocessed HZT products
            A version adapted to real time implementation
            Only used in for the function retrieve_hzt_prod

        Parameters
        ----------
        
        tstep: datetime
            directory where to store the unzipped files
                    
        Returns
        -------
        A list containing all the filepaths of the retrieved files
    
        """

        # Get list of available files
        folder_in = constants.FOLDER_ISO0
        content_zip = np.array(os.listdir(folder_in))
        
        # HZT files are produced once an hour
        start_time = tstep.replace(minute=0)
        end_time = start_time + datetime.timedelta(hours=1)

        try:            
            # Sort filelist to most recent prediction
            content_filt = np.array([c for c in content_zip if c.endswith('800')])
            times_filt = np.array([datetime.datetime.strptime(c[3:12],
                                '%y%j%H%M')+datetime.timedelta(hours=int(c[-2::])) for c in content_filt])
            conditions = np.array([np.logical_and((t >= start_time), (t <= end_time)) for t in times_filt])
            
            content_filt = content_filt[conditions]
            times_filt = times_filt[conditions]
        except:
            logging.error('HZT data does not exist for '+start_time.strftime('%d-%b-%y'))
            files = None
            return
            
        # Check that an hourly estimate is available
        all_hours = pd.date_range(start=start_time, end=end_time, freq='H')
        
        if len(all_hours) != len(times_filt):
            content_times = np.array([datetime.datetime.strptime(c[3:12],
                    '%y%j%H%M')+datetime.timedelta(hours=int(c[-2::])) for c in content_zip])
            # Find time that is missing:
            for hh in all_hours:
                if not hh in times_filt:
                    hh_last = np.where(hh==content_times)
                    times_filt = np.sort(np.append(times_filt, content_times[hh_last][-1]))
                    content_filt = np.sort(np.append(content_filt, content_zip[hh_last][-1]))
        
        # Get a list of all files to retrieve
        conditions = np.array([np.logical_and(t >= start_time, t <= end_time)
                            for t in times_filt])

        if not np.any(conditions):
            msg = '''
            No file was found corresponding to this format, verify pattern and product_name
            '''
            raise ValueError(msg)
            
        files = sorted(np.array([folder_in + c for c in
                                np.array(content_filt)[conditions]]))
        
        return files


    def fetch_data_RT(self, tstep):
        """
        Retrieves and add new polar radar and status data to the QPEProcessor
        for a given time range

        Parameters
        ----------
        tstep : datetime
                Get data for this timestamp
        """

        self.radar_files = {}
        self.status_files = {}

        # Retrieve polar files and lookup tables for all radars
        for rad in self.config['RADARS']:
            logging.info('Retrieving data for radar '+rad)
            try:
                radfiles = self._retrieve_prod_RT(tstep, product_name = 'ML'+rad,
                                                  sweeps = self.config['SWEEPS'])
                self.radar_files[rad] = split_by_time(radfiles)
                statfiles = self._retrieve_prod_RT(tstep, product_name = 'ST' + rad,
                                                    pattern = 'ST*.xml', sweeps = None)
                self.status_files[rad] = split_by_time(statfiles)
            except:
                logging.error('Failed to retrieve data for radar {:s}'.format(rad))
        
        # Retrieve iso0 height files
        if 'ISO0_HEIGHT' in self.cosmo_var:
            files_hzt = self._retrieve_hzt_RT(tstep)
            self.files_hzt = split_by_time(files_hzt)


    def save_output(self, qpe, t, filepath):
        """ Saves output as defined in config file

        Args:
            qpe (array): Precipitation estimates on grid
            t (datetime object): timestep of QPE
            filepath (str): Destination where file is saved
        """
        # Output in binary data and .gif format
        if (self.config['DATA_FORMAT'] == 'DN'):
            if (self.config['FILE_FORMAT'] == 'DN'):
                # Find idx from CPC scale
                qpe = np.searchsorted(constants.SCALE_CPC, qpe)
                qpe = qpe.astype('B') # Convert to byte
                qpe[constants.MASK_NAN] = 255
                qpe.tofile(filepath)
            else:
                if (self.config['FILE_FORMAT'] != 'gif'):
                    logging.error('Invalid file_format with data format DN, using gif output instead')
                qpe[constants.MASK_NAN] = -99
                filepath += '.gif'
                save_gif(filepath, qpe)

        # Output in binary data and .gif format
        qpe[constants.MASK_NAN] = np.nan
        if (self.config['DATA_FORMAT'] == 'float'):
            if (self.config['FILE_FORMAT'] == 'float'):
                qpe.astype(np.float32).tofile(filepath)
            else:
                if (self.config['FILE_FORMAT'] != 'ODIM'):
                    logging.error('Invalid file format with data format float, using ODIM HDF5 output instead')
                grid = _qpe_to_chgrid(qpe, t, radar_list=self.missing_files)
                filepath += '.h5'
                write_odim_grid_h5(filepath, grid)

    def compute(self, output_folder, t0, t1, timestep = 5,
                basename = 'RFO%y%j%H%M'):
        """
        Computes QPE values for a given time range and stores them in a
        folder, in a binary format

        Parameters
        ----------
        output_folder : str
            Folder where to store the computed QPE fields, note that subfolders
            for every model will be created in this folder
        t0 : datetime
            Start time of the timerange in datetime format
        t1 : datetime
            End time of the timerange in datetime format
        timestep : int (optional)
            In case you don't to generate a new product every 5 minute, you can
            change the time here, f.ex. 10 min, will compute the QPE only every
            two sets of radar scans
        basename: str (optional)
            Pattern for the filenames, default is  'RFQ%y%j%H%M' which uses
            the same standard as other MeteoSwiss products
            (example RFQ191011055)

        """

        for model in self.models.keys():
            if self.config['ADVECTION_CORRECTION']:
                model += '_AC'
            if not os.path.exists(str(Path(output_folder, model))):
                os.makedirs(str(Path(output_folder, model)))

        # Retrieve one timestamp before beginning for lead time
        tL = t0-datetime.timedelta(minutes=timestep)

        # Get all timesteps in time range
        n_incr = int((t1 - tL).total_seconds() / (60 * timestep))
        timeserie = tL + np.array([datetime.timedelta(minutes=timestep*i)
                            for i in range(n_incr + 1)])

        qpe_prev = {}
        X_prev = {}

        for i, t in enumerate(timeserie): # Loop on timesteps
            logging.info('====')
            
            # Get lead time file
            if i == 0 :
                logging.info('Processing LEAD time '+str(t))
                for k in self.models.keys():
                    tL_x_file = self.config['TMP_FOLDER']+'/'+datetime.datetime.strftime(t, basename)+\
                                '_{}_xprev.txt'.format(k)
                    tL_qpe_file = self.config['TMP_FOLDER']+'/'+datetime.datetime.strftime(t, basename)+\
                                '_{}_qpeprev.txt'.format(k)
                    one_file_missing = False
                    try:
                        X_prev[k] = np.load(tL_x_file)
                        qpe_prev[k] = np.load(tL_qpe_file)
                    except:
                        one_file_missing = True
                # If all the files could be loaded, go directly to current timestep
                if one_file_missing == False:
                    continue
            
            logging.info('Processing time '+str(t))

            # Retrieve data for time range
            self.fetch_data_RT(t)

            # Log missing radar files
            self.missing_files = {}

            # Initialize RF features
            rf_features_cart = {}
            weights_cart = {}
            
            for var in self.model_weights_per_var.keys():
                for weight in self.model_weights_per_var[var]:
                    if weight not in list(rf_features_cart.keys()):
                        rf_features_cart[weight] = {}

                    rf_features_cart[weight][var] = np.zeros((NBINS_X, NBINS_Y))

                    # add weights
                    if weight not in weights_cart.keys():
                        weights_cart[weight] = np.zeros((NBINS_X, NBINS_Y))

            # Get COSMO temperature for all radars for this timestamp
            if ('T' in self.cosmo_var) :
                T_cosmo_fields = get_COSMO_T(t, radar = self.config['RADARS'])
            if ('ISO0_HEIGHT' in self.cosmo_var) :
                if ('hzt_cosmo_fields' not in locals()) or (t not in hzt_cosmo_fields):
                    logging.info('Interpolating HZT fields for timestep {}'.format(t.strftime('%Y%m%d%H%M')))
                    hzt_cosmo_fields = HZT_hourly_to_5min(t, self.files_hzt, tsteps_min=timestep)

            """Part one - compute radar variables and mask"""
            # Begin compilation of radarobject
            radobjects = {}
            for rad in self.config['RADARS']:                
                # if radar does not exist, add to missing file list
                if self.radar_files[rad] == None:
                    self.missing_files[rad] = t
                    continue

                # if file does not exist, add to missing file list
                if (t not in self.radar_files[rad]) or (self.radar_files[rad][t] == None):
                    self.missing_files[rad] = t
                    continue
                
                # Read raw radar file and create a RADAR object
                radobjects[rad] = Radar(rad, self.radar_files[rad][t],
                                        self.status_files[rad][t])
                
                # if problem with radar file, exclude it and add to missing files list
                if len(radobjects[rad].radarfields) == 0:
                    self.missing_files[rad] = t
                    logging.info('Removing timestep {:s} of radar {:s}'.format(str(t), rad))
                    return
                
                # Process the radar data
                radobjects[rad].visib_mask(self.config['VISIB_CORR']['MIN_VISIB'],
                                self.config['VISIB_CORR']['MAX_CORR'])
                # If it cannot compute noise from the status file, remove timestep
                try:
                    radobjects[rad].snr_mask(self.config['SNR_THRESHOLD'])
                except Exception as e:
                    self.missing_files[rad] = t
                    logging.info(e)
                    logging.info('Removing timestep {:s} of radar {:s}'.format(str(t), rad))
                    return 
                
                radobjects[rad].compute_kdp(self.config['KDP_PARAMETERS'])

                # Add temperature indication to radar object
                if 'T' in self.cosmo_var:
                    radobjects[rad].add_cosmo_data(T_cosmo_fields[rad])
                if 'ISO0_HEIGHT' in self.cosmo_var:
                    radobjects[rad].add_hzt_data(hzt_cosmo_fields[t])
            
            for sweep in self.config['SWEEPS']: # Loop on sweeps
                logging.info('---')
                logging.info('Processing sweep ' + str(sweep))

                for rad in self.config['RADARS']: # Loop on radars, A,D,L,P,W
                    # If there is no radar file for the specific radar, continue to next radar
                    if rad not in radobjects.keys():
                        logging.info('Processing radar {} - no data for this timestep!'.format(rad))
                        continue
                    elif sweep not in radobjects[rad].radsweeps.keys():
                        logging.info('Processing sweep {} of {} - no data for this timestep!'.format(sweep, rad))
                        continue
                    else:
                        logging.info('Processing radar ' + str(rad))
                    
                    try:
                        """Part two - retrieve radar data at every sweep"""
                        datasweep = {}
                        ZH = np.ma.filled(radobjects[rad].get_field(sweep,'ZH'),
                                          np.nan)

                        for var in self.model_weights_per_var.keys():
                            if 'RADAR' in var:
                                datasweep['RADAR_{:s}'.format(rad)] = np.isfinite(ZH).astype(float)
                            elif var == 'HEIGHT':
                                datasweep['HEIGHT'] = self.rad_heights[rad][sweep].copy()
                            else:
                                datasweep[var] = np.ma.filled(radobjects[rad].get_field(sweep, var),
                                          np.nan)

                        # Mask on minimum zh
                        invalid = np.logical_or(np.isnan(ZH),
                                                ZH < self.config['ZH_THRESHOLD'])

                        for var in datasweep.keys():
                            # Because add.at does not support nan we replace by zero
                            datasweep[var][invalid] = 0

                        """Part three - convert to Cartesian"""
                        # get cart index of all polar gates for this sweep
                        lut_elev = self.lut_cart[rad][self.lut_cart[rad][:,0]
                                                        == sweep-1] # 0-indexed
                        
                        # Convert from Swiss-coordinates to array index
                        idx_ch = np.vstack((len(X_QPE_CENTERS)  -
                                        (lut_elev[:,4] - np.min(X_QPE_CENTERS)),
                                        lut_elev[:,3] -  np.min(Y_QPE_CENTERS))).T

                        idx_ch = idx_ch.astype(int)
                        idx_polar = [lut_elev[:,1], lut_elev[:,2]]

                        for weight in rf_features_cart.keys():
                            # Compute altitude weighting
                            W = 10 ** (weight[0] * (datasweep['HEIGHT']/1000.))
                            W[invalid] = 0
                            
                            # Compute visib weighting
                            if weight[1]:
                                W *= datasweep['VISIB'] / 100

                            for var in rf_features_cart[weight].keys():
                                if var in datasweep.keys():
                                    # Add variable to cart grid
                                    nanadd_at(rf_features_cart[weight][var],
                                        idx_ch, (W * datasweep[var])[idx_polar[0], idx_polar[1]])
                            # Add weights to cart grid
                            nanadd_at(weights_cart[weight], idx_ch, W[idx_polar[0], idx_polar[1]])
                    except:
                        raise
                        logging.error('Could not compute sweep {:d}'.format(sweep))
                        pass


            """Part four - RF prediction"""
            # Get QPE estimate
            # X: current time step; X_prev: previous timestep (t-5min)
            for k in self.models.keys():
                model = self.models[k]
                X = []
                for v in model.variables:
                    dat = (rf_features_cart[(model.beta,model.visib_weighting)][v] 
                             / weights_cart[(model.beta,model.visib_weighting)])
                    X.append(dat.ravel())

                X = np.array(X).T
                
                if i == 0:
                    X_prev[k] = X

                # Take average between current timestep and t-5min
                Xcomb = np.nanmean((X_prev[k] , X),axis = 0)
                X_prev[k]  = X
                Xcomb[np.isnan(Xcomb)] = 0

                # Remove axis with only zeros
                validrows = (Xcomb>0).any(axis=1)

                # Convert radar data to precipitation estimates
                qpe = np.zeros((NBINS_X, NBINS_Y), dtype = np.float32).ravel()
                try:
                    qpe[validrows] = self.models[k].predict(Xcomb[validrows,:])
                except:
                    logging.error('Model failed!')
                    pass

                """Temporal disaggregation; Rescale qpe through rproxy"""
                idx_zh = np.where(np.array(model.variables)
                                  == 'zh_VISIB')[0][0]

                rproxy = (X[:,idx_zh]/constants.A_QPE)**(1/constants.B_QPE)
                rproxy[np.isnan(rproxy)] = 0
                
                rproxy_mean = (Xcomb[:,idx_zh]/constants.A_QPE)**(1/constants.B_QPE)
                rproxy_mean[np.isnan(rproxy_mean)] = 0
                
                disag = rproxy / rproxy_mean
                disag = np.reshape(disag, (NBINS_X, NBINS_Y))
                qpe = np.reshape(qpe, (NBINS_X, NBINS_Y))
                disag[np.isnan(disag)] = 0
                qpe = qpe * disag
                
                """Part five - Postprocessing"""
                if self.config['OUTLIER_REMOVAL']:
                    qpe = _outlier_removal(qpe)

                if self.config['GAUSSIAN_SIGMA'] > 0:
                    qpe = gaussian_filter(qpe,
                       self.config['GAUSSIAN_SIGMA'])

                if i == 0:
                    qpe_prev[k] = qpe
                    # Because of lead time, we can go out here:
                    logging.info('Processing time '+str(t)+' was only used as lead time and the final QPE map will not be saved')
                    continue
                
                # Save files in a temporary format
                np.save(self.config['TMP_FOLDER']+'/'+datetime.datetime.strftime(t, basename)+\
                            '{}_xprev.txt'.format(k), X_prev[k])
                np.save(self.config['TMP_FOLDER']+'/'+datetime.datetime.strftime(t, basename)+\
                            '{}_qpeprev.txt'.format(k), qpe_prev[k])

                comp = np.array([qpe_prev[k].copy(), qpe.copy()])
                qpe_prev[k] = qpe
                if self.config['ADVECTION_CORRECTION'] and i > 0:
                    if not _PYSTEPS_AVAILABLE:
                        logging.error("Pysteps is not available, no qpe disaggregation will be performed!")
                    qpe = _disaggregate(comp)

                tstr = datetime.datetime.strftime(t, basename)
                filepath = output_folder + '/' + k
                if self.config['ADVECTION_CORRECTION']:
                    filepath += '_AC'

                filepath += '/' + tstr
                
                """Part six - Save output"""
                self.save_output(qpe, t, filepath)