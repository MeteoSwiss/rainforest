#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function to compute the randomForest QPE estimate

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019
"""


import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import datetime
import os
import yaml
import logging
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
logging.basicConfig(level=logging.INFO)


import os, sys
sys.path.append('/store/msrad/radar/rainforest/rainforest/')
 

from common import constants
from common.retrieve_data import retrieve_prod, get_COSMO_T
from common.lookup_tables import get_lookup
from common.utils import split_by_time, nanadd_at
from common.radarprocessing import Radar

from rfdefinitions import read_rf

def _outlier_removal(image, N = 3, threshold = 3):
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

###############################################################################

# Centerpoints of all QPE grid cells
Y_QPE_CENTERS = 0.5 * (constants.Y_QPE[0:-1] + constants.Y_QPE[1:])
X_QPE_CENTERS = 0.5 * (constants.X_QPE[0:-1] + constants.X_QPE[1:])

NBINS_X = len(X_QPE_CENTERS)
NBINS_Y = len(Y_QPE_CENTERS)


class QPEProcessor(object):
    def __init__(self, config_file, models):
        self.config = yaml.load(open(config_file, 'r'),
                                Loader = yaml.FullLoader)
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
        # is a list of all vertical weighting to be used
        for k in self.models.keys():
            for var in self.models[k].variables:
                if var not in self.model_weights_per_var.keys():
                    self.model_weights_per_var[var] = []
                if models[k].vw not in self.model_weights_per_var[var]:
                    self.model_weights_per_var[var].append(models[k].vw)
            
    def fetch_data(self, t0, t1):
        self.radar_files = {}
        self.status_files = {}

        # Retrieve polar files and lookup tables for all radars
        for rad in self.config['RADARS']:
            logging.info('Retrieving data for radar '+rad)
            
            radfiles = retrieve_prod(self.config['TMP_FOLDER'], t0, t1, 
                               product_name = 'ML' + rad, 
                               sweeps = self.config['SWEEPS'])
            self.radar_files[rad] = split_by_time(radfiles)
            statfiles = retrieve_prod(self.config['TMP_FOLDER'], t0, t1, 
                               product_name = 'ST' + rad, pattern = 'ST*.xml')
            self.status_files[rad] = split_by_time(statfiles)
    
    def compute(self, output_folder, t0, t1, timestep = 5, 
                                                    basename = 'RF%y%j%H%M'):
        
        for model in self.models.keys():
            if not os.path.exists(output_folder + '/' + model):
                os.makedirs(output_folder + '/' + model)
        
        # Retrieve data for time range
        self.fetch_data(t0, t1)
        
        # Get all timesteps in time range
        n_incr = int((t1 - t0).total_seconds() / (60 * timestep))
        timeserie = t0 + np.array([datetime.timedelta(minutes = timestep * i) 
                            for i in range(n_incr + 1)])
    
        for t in timeserie: # Loop on timesteps
            logging.info('====')
            logging.info('Processing time '+str(t))
        
            # Initialize RF features 
            rf_features_cart = {}
            weights_cart = {}
            for var in self.model_weights_per_var.keys():
                for weight in self.model_weights_per_var[var]:
                    if weight not in rf_features_cart.keys():
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
                radobjects[rad] = Radar(rad, self.radar_files[rad][t],
                          self.status_files[rad][t])
                radobjects[rad].visib_mask(self.config['VISIB_CORR']['MIN_VISIB'],
                                 self.config['VISIB_CORR']['MAX_CORR'])
                radobjects[rad].snr_mask(self.config['SNR_THRESHOLD'])
                radobjects[rad].compute_kdp(self.config['KDP_PARAMETERS'])
                radobjects[rad].add_cosmo_data(T_cosmo[rad])
            
            for sweep in self.config['SWEEPS']: # Loop on sweeps
                logging.info('---')
                logging.info('Processing sweep ' + str(sweep))
                
                for i,rad in enumerate(self.config['RADARS']): # Loop on radars, A,D,L,P,W
                    logging.info('Processing radar ' + str(rad))
                    if sweep not in radobjects[rad].radsweeps.keys():
                        continue
                    
                    """Part two - retrieve radar data at every sweep"""
                    datasweep = {}
                    ZH = np.ma.filled(radobjects[rad].get_field(sweep,'ZH'), np.nan)
                    for var in self.model_weights_per_var.keys():
                        if 'RADAR' in var:
                            datasweep['RADAR_{:s}'.format(rad)] = np.isfinite(ZH).astype(float)
                        elif var == 'HEIGHT':
                            datasweep['HEIGHT'] = self.rad_heights[rad][sweep]
                        else:
                            datasweep[var] = radobjects[rad].get_field(sweep, var).data
                    
                    # Mask on minimum zh
                    invalid = np.logical_or(np.isnan(ZH), 
                                            ZH < self.config['ZH_THRESHOLD'])
                    for var in datasweep.keys():
                        # Because add.at does not support nan we replace by zero
                        datasweep[var][invalid] = np.nan
                    
                    """Part three - convert to Cartesian"""
                    # get cart index of all polar gates for this sweep
                    lut_elev = self.lut_cart[rad][self.lut_cart[rad][:,0] == sweep-1] # 0-indexed
                    
                    # Convert from Swiss-coordinates to array index
                    idx_ch = np.vstack((len(X_QPE_CENTERS)  -
                                            (lut_elev[:,4] - np.min(X_QPE_CENTERS)),  
                                            lut_elev[:,3] -  np.min(Y_QPE_CENTERS))).T
                                            
                    idx_ch = idx_ch.astype(int)
                    idx_polar = [lut_elev[:,1],lut_elev[:,2]]
        
                    for weight in rf_features_cart.keys():
                        # Compute altitude weighting
                        W = 10 ** (weight * (datasweep['HEIGHT']/1000.))
        
                        for var in rf_features_cart[weight].keys():
                            if var in datasweep.keys():
                                # Add variable to cart grid
                                nanadd_at(rf_features_cart[weight][var], 
                                    idx_ch, (W * datasweep[var])[idx_polar])
                        # Add weights to cart grid
                        nanadd_at(weights_cart[weight], 
                                idx_ch, W[idx_polar])
            
            """Part four - RF prediction"""
            # Get QPE estimate
            for k in self.models.keys():
                model = self.models[k]
                X = []
                for v in model.variables:
                    dat = rf_features_cart[model.vw][v] / weights_cart[model.vw]
                    X.append(dat.ravel())
                
                X = np.array(X).T
                X[np.isnan(X)] = 0
                # Remove axis with only zeros
                validrows = (X>0).any(axis=1)
            
                qpe = np.zeros((NBINS_X, NBINS_Y), dtype = np.float32).ravel()
                try:
                    qpe[validrows] = self.models[k].predict(X[validrows,:])
                except:
                    logging.error('RF failed!')
                    pass
                
                # Reshape to Cartesian grid
                qpe = np.reshape(qpe, (NBINS_X, NBINS_Y))
                
                # Postprocessing
                if self.config['OUTLIER_REMOVAL']:
                    qpe = _outlier_removal(qpe)
                
                if self.config['GAUSSIAN_SIGMA'] > 0:
                    qpe = gaussian_filter(qpe,
                       self.config['GAUSSIAN_SIGMA'])
            
                tstr = datetime.datetime.strftime(t, basename)
                filepath = output_folder + '/' + k + '/' + tstr
                qpe.tofile(filepath)
                
        return qpe
        
if __name__ == '__main__':
    
            
    t0 = datetime.datetime(2018,6,11,21)
    t1 = datetime.datetime(2018,6,11,21)
    
    
    config = '/store/msrad/radar/rainforest/rainforest/qpe/config.yml'
    models = {'dualpol': read_rf('dualpol_model_BC_raw.p'),'hpol': read_rf('hpol_model_BC_raw.p')}

    qpe = QPEProcessor(config, models)
    out = qpe.compute('/scratch/wolfensb/', t0, t1)
    
    