#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function to compute the randomForest QPE estimate

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019

Modified by D. Wolfensberger and R. Gugerli
March 2022
"""


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import datetime
import os
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates

from pyart.testing import make_empty_grid
from pyart.aux_io.odim_h5_writer import write_odim_grid_h5
from pyart.aux_io.odim_h5 import proj4_to_dict

logging.getLogger().setLevel(logging.INFO)


from ..common import constants
from ..common.retrieve_data import retrieve_prod, get_COSMO_T
from ..common.lookup import get_lookup
from ..common.utils import split_by_time, nanadd_at, envyaml
from ..common.radarprocessing import Radar
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


class QPEProcessor(object):
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
        # keys of this dict are the variable used for the RF models, their values
        # is a list of lists with 1. the vertical weighting beta parameters to be used, 2. if visib_weighting is used or not
        for k in self.models.keys():
            for var in self.models[k].variables:
                if var not in self.model_weights_per_var.keys():
                    self.model_weights_per_var[var] = []
                if models[k].beta not in self.model_weights_per_var[var]:
                    self.model_weights_per_var[var].append((models[k].beta, models[k].visib_weighting))

    def fetch_data(self, t0, t1):
        """
        Retrieves and add new polar radar and status data to the QPEProcessor
        for a given time range

        Parameters
        ----------
        t0 : datetime
            Start time of the timerange in datetime format
        t1 : datetime
            End time of the timerange in datetime format
        """

        self.radar_files = {}
        self.status_files = {}

        # Retrieve polar files and lookup tables for all radars
        for rad in self.config['RADARS']:
            logging.info('Retrieving data for radar '+rad)
            try:
                radfiles = retrieve_prod(self.config['TMP_FOLDER'], t0, t1,
                                   product_name = 'ML' + rad,
                                   sweeps = self.config['SWEEPS'])
                self.radar_files[rad] = split_by_time(radfiles)
                statfiles = retrieve_prod(self.config['TMP_FOLDER'], t0, t1,
                                   product_name = 'ST' + rad, pattern = 'ST*.xml')
                self.status_files[rad] = split_by_time(statfiles)
            except:
                logging.error('Failed to retrieve data for radar {:s}'.format(rad))

    def compute(self, output_folder, t0, t1, timestep = 5,
                                                    basename = 'RFQ%y%j%H%M'):
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
        
        # Retrieve data for time range
        self.fetch_data(tL, t1)

        # Get all timesteps in time range
        n_incr = int((t1 - tL).total_seconds() / (60 * timestep))
        timeserie = tL + np.array([datetime.timedelta(minutes=timestep*i)
                            for i in range(n_incr + 1)])

        qpe_prev = {}
        X_prev = {}

        for i, t in enumerate(timeserie): # Loop on timesteps
            logging.info('====')
            logging.info('Processing time '+str(t))

            # Log missing radar files
            missing_files = {}

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

            """Part one - compute radar variables and mask"""
            # Get COSMO temperature for all radars for this timestamp
            T_cosmo = get_COSMO_T(t, radar = self.config['RADARS'])
            radobjects = {}

            for rad in self.config['RADARS']:
                # if radar does not exist, add to missing file list
                if self.radar_files.get(rad) == None:
                    missing_files[rad] = t
                    continue

                # if file does not exist, add to missing file list
                if self.radar_files[rad].get(t) == None:
                    missing_files[rad] = t
                    continue
                
                # Read raw radar file and create a RADAR object
                radobjects[rad] = Radar(rad, self.radar_files[rad][t],
                                        self.status_files[rad][t])
                
                # if problem with radar file, exclude it and add to missing files list
                if len(radobjects[rad].radarfields) == 0:
                    missing_files[rad] = t
                    logging.info('Removing timestep {:s} of radar {:s}'.format(str(t), rad))
                    continue
                
                # Process the radar data
                radobjects[rad].visib_mask(self.config['VISIB_CORR']['MIN_VISIB'],
                                self.config['VISIB_CORR']['MAX_CORR'])
                # If it cannot compute noise from the status file, remove timestep
                try:
                    radobjects[rad].snr_mask(self.config['SNR_THRESHOLD'])
                except Exception as e:
                    missing_files[rad] = t
                    logging.info(e)
                    logging.info('Removing timestep {:s} of radar {:s}'.format(str(t), rad))
                    continue
                radobjects[rad].compute_kdp(self.config['KDP_PARAMETERS'])
                radobjects[rad].add_cosmo_data(T_cosmo[rad])

                # Delete files if config files requires
                try:
                    if self.config['CLEANUP'] == 'delete_all':
                        for f in self.radar_files[rad][t]:
                            if os.path.exists(f):
                                os.remove(f)
                        if os.path.exists(self.status_files[rad][t]):
                            os.remove(self.status_files[rad][t])
                except:
                    logging.error('No cleanup was defined, unzipped files remain in temp-folder')
            
            for sweep in self.config['SWEEPS']: # Loop on sweeps
                logging.info('---')
                logging.info('Processing sweep ' + str(sweep))

                for rad in self.config['RADARS']: # Loop on radars, A,D,L,P,W
                    # If there is no radar file for the specific radar, continue to next radar
                    if rad not in radobjects.keys():
                        logging.info('Processing radar {} - no data for this timestep!'.format(rad))
                        continue
                    logging.info('Processing radar ' + str(rad))
                    if sweep not in radobjects[rad].radsweeps.keys():
                        continue
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
                                        idx_ch, (W * datasweep[var])[idx_polar])
                            # Add weights to cart grid
                            nanadd_at(weights_cart[weight], idx_ch, W[idx_polar])
                    except:
                        logging.error('Could not compute sweep {:d}'.format(sweep))
                        pass


            """Part four - RF prediction"""
            # Get QPE estimate
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

                idx_zh = np.where(np.array(model.variables)
                                  == 'zh_VISIB')[0][0]

                rproxy = (X[:,idx_zh]/constants.A_QPE)**(1/constants.B_QPE)
                rproxy[np.isnan(rproxy)] = 0

                Xcomb = np.nanmean((X_prev[k] , X),axis = 0)
                X_prev[k]  = X

                Xcomb[np.isnan(Xcomb)] = 0
                rproxy_mean = (Xcomb[:,idx_zh]/constants.A_QPE)**(1/constants.B_QPE)
                rproxy_mean[np.isnan(rproxy_mean)] = 0

                # Remove axis with only zeros
                validrows = (Xcomb>0).any(axis=1)

                # Convert radar data to precipitation estimates
                qpe = np.zeros((NBINS_X, NBINS_Y), dtype = np.float32).ravel()
                try:
                    qpe[validrows] = self.models[k].predict(Xcomb[validrows,:])
                except:
                    logging.error('RF failed!')
                    pass

                # Rescale qpe through rproxy
                disag = rproxy / rproxy_mean
                disag = np.reshape(disag, (NBINS_X, NBINS_Y))
                qpe = np.reshape(qpe, (NBINS_X, NBINS_Y))
                disag[np.isnan(disag)] = 0
                qpe = qpe * disag

                # Postprocessing
                if self.config['OUTLIER_REMOVAL']:
                    qpe = _outlier_removal(qpe)

                if self.config['GAUSSIAN_SIGMA'] > 0:
                    qpe = gaussian_filter(qpe,
                       self.config['GAUSSIAN_SIGMA'])

                if i == 0:
                    qpe_prev[k] = qpe
                    # Because of lead time, we can go out here:
                    logging.info('Processing time '+str(t)+' was only used as lead time and discarded')
                    continue

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
                        grid = _qpe_to_chgrid(qpe, t, radar_list=missing_files)
                        filepath += '.h5'
                        write_odim_grid_h5(filepath, grid)
