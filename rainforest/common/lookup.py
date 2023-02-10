#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions to read and compute lookup tables

The available lookup tables are

**qpebias_station** : dict with the bias correction of RZC at every station

**cosmo1_to_rad** : dict which maps COSMO grid to polar radar grid
it has  keys [sweep][coord_type], coord_type is 'idx_0' : 
first index of COSMO grids, 'idx_1': second, 'idx_3': third, 
mask' is 1 for points that fall outside of COSMO domain. This lookup table
is valid only for COSMO data stored in /store/s83/owm/COSMO-1/

**cosmo2_to_rad** : same thing but for COSMO 1E data

**cosmo2_to_rad** : same thing but for COSMO 2 data

**cosmo1T_to_rad** : same thing but for netCDF files of COSMO1 temperature
extracted for MDR and stored in /store/s83/owm/COSMO-1/ORDERS/MDR/

**cosmo1T_to_rad** : same thing but for netCDF files of COSMO2 temperature
extracted for MDR and stored in /store/msrad/cosmo/cosmo2/data/

**station_to_rad** : maps the SMN stations to radar coordinates, it is an 
extraction of the more generic but less convenient **qpegrid_to_rad** table
It is list of 3 elements, first element is a dict with keys [station][sweep][ncode]
and gives the polar gates (azimuth_idx, range_idx) that fall within a Cartesian pixel
at a given radar elevation and for a given station neighbour (00 = station location,
-1-1 = one to the south-west, 22 = two km to north and 2 km to east)
second element is a dict giving distance from every station to the radar
third element is a dict of keys [station][sweep] giving the height above
ground of the radar observations above that station at a given elevation (sweep)
MISSING KEYS IMPLY NO RADAR VISIBILITY

**cartcoords_rad** : gives the Cartesian (Swiss LV03) coordinates of all
polar gates. It is a dict that gives for every sweep a 3D array
of shape ncoords x nazimuth x nrange, ncoords is 3, first slice is 
Swiss Y coordinate (west to east), second is Swiss X-coordinate (south to north)
and last is Swiss Z coordinate (altitude)

**qpegrid_to_rad** : maps the radar polar data to any gridpoint of the Swiss
QPE grid. It is simply a 2D array with 5 columns 
| sweep | azimuth_idx | range_idx | Swiss Y coord | Swiss X coord|

**qpegrid_to_height_rad** : gives for every pixel of the Swiss Cartesian QPE grid
the average height of radar observations for a given radar, for every sweep. It is a dictionary
where the key is the sweep number ( 1 - 20) and the value is a 640 x 710 array of heights above ground

**station_to_qpegrid** : maps every station to the corresponding QPE gridpoint
it is a dict of keys [station][ncode] and gives the index of every 
neighbour of every station in the QPE 2D grid 640 x 710 pixels

**visibility_rad** : gives the (static) visibility of every polar gate 
for a given sweep number in the form of a 2D field of size nazimuth x nrange
    
"""

import pickle
import os
import glob
import pandas as pd
import logging
import numpy as np
from pathlib import Path
import netCDF4
from pyart.map.polar_to_cartesian import get_earth_radius
from pyart.aux_io import read_metranet

# Local imports
from . import constants
from .utils import nanadd_at
from .wgs84_ch1903 import GPSConverter
from .object_storage import ObjectStorage
ObjStorage = ObjectStorage()

from .constants import data_folder
from .constants import metadata_folder
from .constants import lut_folder
from .constants import lut_boscacci_folder
from .constants import cosmo_coords_folder
from .constants import radar_samples_folder
from .constants import rfmodels_folder

LOOKUP_FOLDER = Path(os.environ['RAINFOREST_DATAPATH'], 'references', 'lookup_data')

def get_lookup(lookup_type, radar = None):
    """Read a lookup table from the /data/lookup_data folder

    Parameters
    ----------
    lookup_type : str
        the lookup table type, must be one of
        
        -   qpebias_station
        -   cosmo1_to_rad
        -   cosmo1e_to_rad
        -   cosmo2_to_rad
        -   cosmo1T_to_rad
        -   cosmo2T_to_rad
        -   station_to_rad
        -   cartcoords_rad
        -   qpegrid_to_rad
        -   station_to_qpegrid
        -   visibility_rad
    radar : char or list of chars (optional) 
        the radar for which to retrieve the lookup table, needed only
        if the lookup_type contains the term 'rad', must be either 'A', 'D', 'L',
        'W' or 'P'

    Returns
    -------
    lut: dict
        The lookup table in the form of a python dict
    """
    
    if 'rad' in lookup_type and radar == None:
        raise ValueError('Please indicate radar name for this lookup type')
    
    if radar == None:
        lut_name = str(Path(lut_folder, 'lut_' + lookup_type + '.p'))
        lut_name = ObjStorage.check_file(lut_name)
        lut = pickle.load(open(lut_name,'rb'))
    else:
        if type(radar) != list:
            radar = [radar]
            
        lut = {}
        for r in radar:
            lut_name = str(Path(lut_folder, 'lut_' + lookup_type + r + '.p'))
            lut_name = ObjStorage.check_file(lut_name)
            lut[r] = pickle.load(open(lut_name,'rb'))
            
        if len(lut.keys()) == 1:
            lut = lut[r]
        
    return lut


def calc_lookup(lookup_type, radar = None):
    """Calculates a lookup table and stores it in the /data/lookup_data folder

    Parameters
    ----------
    lookup_type : str
        the lookup table type, must be one of
    
        -   qpebias_station
        -   cosmo1_to_rad
        -   cosmo1e_to_rad
        -   cosmo2_to_rad
        -   cosmo1T_to_rad
        -   cosmo2T_to_rad
        -   station_to_rad
        -   cartcoords_rad
        -   qpegrid_to_rad
        -   station_to_qpegrid
        -   visibility_rad
    radar : char or list of chars (optional) 
        the radar for which to retrieve the lookup table, needed only
        if the lookup_type contains the term 'rad', must be either 'A', 'D', 'L',
        'W' or 'P'
    """
    
    # Default is to use all sweeps, and a window of 25 (5 x 5) pixels 
    # around gauge
    sweeps = range(1,21)
    neighb_x = 5
    neighb_y = 5
    
    # List that contains the computed lookup tables
    lut_objects = []
    # List that contains the lookup table filenames
    lut_names = []

    if lookup_type == 'station_to_rad':
        offset_x = int((neighb_x-1)/2)
        offset_y = int((neighb_y-1)/2)
            
        for r in radar:
            lut_name =  Path(LOOKUP_FOLDER, 'lut_station_to_rad{:s}.p'.format(r))
            logging.info('Creating lookup table {:s}'.format(str(lut_name)))
            try:
                lut_cart = get_lookup('qpegrid_to_rad', radar)
            except:
                raise IOError('Could not load qpegrid_to_rad lookup for radar {:s}, compute it first!'.format(r))
            try:
                lut_coords = get_lookup('cartcoords_rad', radar)
            except:    
                raise IOError('Could not load cartcoords_rad lookup for radar {:s}, compute it first!'.format(r))
                
            stations = list(constants.METSTATIONS['Abbrev'])
            
            all_idx_sta = {}
            all_distances_sta = {}
            all_heights_sta = {}
            
            # Get x and y of all radar pixels    
            for sweep in sweeps:        
                sweep_idx = sweep - 1
                for station in stations:
                    station_data = constants.METSTATIONS[constants.METSTATIONS.Abbrev 
                                                      == station]
                    x_sta =  float(station_data.X)
                    y_sta = float(station_data.Y)
                    
                    # For x the columns in the Cartesian lookup tables are lower bounds
                    # e.g. x = 563, means that radar pixels are between 563 and 564
                    #x_llc_sta = np.int(x_sta/constants.CART_GRID_SIZE)
                    # For y the columns in the Cartesian lookup tables are upper bounds
                    # e.g. y = 182, means that radar pixels are between 181 and 182            
                    #y_llc_sta = np.int(np.ceil(y_sta/constants.CART_GRID_SIZE))
                    
                    # x and y are reversed (following the Swiss convention),
                    # therefore, the station cell number needs to be 
                    # defined as follows:

                    x_llc_sta = int(np.ceil(x_sta/constants.CART_GRID_SIZE))
                    y_llc_sta = int(y_sta/constants.CART_GRID_SIZE)
                    
                    # Distance from all gates to gauge
                    idx = lut_cart[np.logical_and(lut_cart[:,0] == sweep_idx,
                                          np.logical_and(lut_cart[:,3] == y_llc_sta, 
                                          lut_cart[:,4] == x_llc_sta)), 0:3]
                    
                    if not len(idx):
                        continue
                    
                    if station not in all_idx_sta.keys():
                        all_idx_sta[station] = {}
                        all_heights_sta[station] = {}
                        all_distances_sta[station] = np.sqrt((lut_coords[sweep][0] - x_sta)**2+
                                 (lut_coords[sweep][1] - y_sta)**2)
                    
                    if sweep not in  all_idx_sta[station].keys():
                        all_idx_sta[station][sweep] = {}
                        
                        
                    all_heights_sta[station][sweep] = np.nanmean(lut_coords[sweep][2][idx[:,1],
                                        idx[:,2]])
            
                    for i in range(-offset_x, offset_x + 1):
                        for j in range(-offset_y, offset_y + 1):
                            x_llc = x_llc_sta + i
                            y_llc = y_llc_sta + j 
                            
                            idx = lut_cart[np.logical_and(lut_cart[:,0] == sweep_idx,
                                          np.logical_and(lut_cart[:,3] == y_llc, 
                                          lut_cart[:,4] == x_llc)), 0:3]
               
                            key = str(i)+str(j)
                            if len( idx[:,1:3]):
                                all_idx_sta[station][sweep][key] = idx[:,1:3]

            lut_names.append(str(lut_name))
            lut_objects.append([all_idx_sta, all_distances_sta, all_heights_sta])


    elif lookup_type in ['cosmo1_to_rad', 'cosmo1e_to_rad', 'cosmo2_to_rad']:
        
        converter = GPSConverter()
        
        cosmo_version = lookup_type.split('cosmo')[1].split('_')[0]
        fname_cosmo_coords =  Path(cosmo_coords_folder, 'coords_COSMO{:s}.nc'.
                                   format(cosmo_version))
        fname_cosmo_coords = ObjStorage.check_file(fname_cosmo_coords)

        coords_COSMO = netCDF4.Dataset(fname_cosmo_coords)
        
        x_c = coords_COSMO.variables['x_1'][:]
        y_c = coords_COSMO.variables['y_1'][:]
        z_c = coords_COSMO.variables['HHL'][:]
        
        lat = coords_COSMO.variables['lat_1'][:]
        lon = coords_COSMO.variables['lon_1'][:]
        
        z_c = 0.5*(z_c[0:-1] + z_c[1:])
        min_x = np.min(x_c)
        max_x = np.max(x_c)
        min_y = np.min(y_c)
        max_y = np.max(y_c)
            
        for r in radar:
            lut = {}
            lut_name =  Path(lut_folder, 'lut_' + lookup_type+'{:s}.p'.format(r))
            logging.info('Creating lookup table {:s}'.format(str(lut_name)))
            try:
                lut_coords = get_lookup('cartcoords_rad', radar)
            except:    
                raise IOError('Could not load cartcoords_rad lookup for radar {:s}, compute it first!'.format(r))
        
            for sweep in sweeps:
                lut[sweep] = {}
                y = lut_coords[sweep][0]
                x = lut_coords[sweep][1]
                z = lut_coords[sweep][2]
                
                lat, lon, z = converter.LV03toWGS84(x,y,z)
                y,x = _WGS_to_COSMO([lat,lon])
                
                idxx = np.round((x - min_x)/(max_x - min_x) * len(x_c)).astype(int)
                mask = np.zeros(idxx.shape)
                
                idxx[np.logical_or(idxx < 0, idxx > len(x_c) -1 )] = 0
                mask[np.logical_or(idxx < 0, idxx > len(x_c) -1 )] = 1
                
                idxy = np.round((y - min_y)/(max_y - min_y) * len(y_c)).astype(int)
                idxy[np.logical_or(idxy < 0, idxy > len(y_c) -1 )] = 0
                mask[np.logical_or(idxy < 0, idxy > len(y_c) -1 )] = 1
                
                idxz = np.argmin(np.abs(z_c[:,idxy,idxx] - z), axis=0) 
                
                mask[z > z_c[0,idxy,idxx]] = 1
                mask[z < z_c[-1,idxy,idxx]] = 1
                
                lut[sweep]['idx0'] = idxz.astype(np.uint16)
                lut[sweep]['idx1'] = idxy.astype(np.uint16)
                lut[sweep]['idx2'] = idxx.astype(np.uint16)
                lut[sweep]['mask'] = mask.astype(np.bool_)
            
            lut_names.append(str(lut_name))
            lut_objects.append(lut)

    
    elif lookup_type in ['cosmo1T_to_rad', 'cosmo2T_to_rad']:
        cosmo_version = int(lookup_type[5])
        fname_cosmo_coords =  Path(cosmo_coords_folder,'coords_cosmo', 'coords_COSMO{:d}_T.nc'.
                                   format(cosmo_version))
        fname_cosmo_coords = ObjStorage.check_file(fname_cosmo_coords)
        coords_COSMO = netCDF4.Dataset(fname_cosmo_coords)
        
        x_c = coords_COSMO.variables['x_1'][:]
        y_c = coords_COSMO.variables['y_1'][:]
        z_c = coords_COSMO.variables['HFL'][:]
        
        lat = coords_COSMO.variables['lat_1'][:]
        lon = coords_COSMO.variables['lon_1'][:]

        min_x = np.min(x_c)
        max_x = np.max(x_c)
        min_y = np.min(y_c)
        max_y = np.max(y_c)

        for r in radar:
            lut = {}
            lut_name =  Path(LOOKUP_FOLDER, 'lut_' + lookup_type+'{:s}.p'.format(r))
            logging.info('Creating lookup table {:s}'.format(str(lut_name)))
            try:
                lut_coords = get_lookup('cartcoords_rad', radar)
            except:    
                raise IOError('Could not load cartcoords_rad lookup for radar {:s}, compute it first!'.format(r))
             
            for sweep in sweeps:
                lut[sweep] = {}
                y = lut_coords[sweep][0]
                x = lut_coords[sweep][1]
                z = lut_coords[sweep][2]
                        
                idxx = np.round((x - min_x)/(max_x - min_x) * len(x_c)).astype(int)
                mask = np.zeros(idxx.shape)
                
                idxx[np.logical_or(idxx < 0, idxx > len(x_c) -1 )] = 0
                mask[np.logical_or(idxx < 0, idxx > len(x_c) -1 )] = 1
                
                idxy = np.round((y - min_y)/(max_y - min_y) * len(y_c)).astype(int)
                idxy[np.logical_or(idxy < 0, idxy > len(y_c) -1 )] = 0
                mask[np.logical_or(idxy < 0, idxy > len(y_c) -1 )] = 1
                
                idxz = np.argmin(np.abs(z_c[:,idxy,idxx] - z), axis=0) 
                
                mask[z > z_c[0,idxy,idxx]] = 1
                mask[z < z_c[-1,idxy,idxx]] = 1
             
                lut[sweep]['idx0'] = idxz.astype(np.uint16)
                lut[sweep]['idx1'] = idxy.astype(np.uint16)
                lut[sweep]['idx2'] = idxx.astype(np.uint16)
                lut[sweep]['mask'] = mask.astype(np.bool_)

            lut_names.append(str(lut_name))
            lut_objects.append(lut)


    elif lookup_type == 'qpebias_station':
        biasfile = Path(metadata_folder, 'lbias_af_map15.dat')
        biasfile = ObjStorage.check_file(str(biasfile))

        BIAS_CORR =  np.fromfile(str(biasfile),
                         dtype = np.float32).reshape(640,710)

        # TODO

    elif lookup_type == 'qpegrid_to_height_rad':
        for r in radar:
            lut_name =  Path(LOOKUP_FOLDER, 'lut_' + lookup_type+'{:s}.p'.format(r))

            try:
                lut_cart = get_lookup('qpegrid_to_rad', r)
            except:
                raise IOError('Could not load qpegrid_to_rad lookup for radar {:s}, compute it first!'.format(r))
            try:
                lut_coords = get_lookup('cartcoords_rad', r)
            except:    
                raise IOError('Could not load cartcoords_rad lookup for radar {:s}, compute it first!'.format(r))

            Y_QPE_CENTERS = constants.Y_QPE_CENTERS
            X_QPE_CENTERS = constants.X_QPE_CENTERS

            all_heights_qpe = {}

            for sweep in sweeps:        
                sweep_idx = sweep - 1
                heights_qpe =  np.zeros((len(X_QPE_CENTERS), len(Y_QPE_CENTERS)))
                weights = np.zeros((len(X_QPE_CENTERS), len(Y_QPE_CENTERS)))

                lut_elev = lut_cart[lut_cart[:,0] == sweep-1] # 0-indexed
                                    
                # Convert from Swiss-coordinates to array index
                # wod: 9.02.2022: after careful examination and tests with
                # random fields it seems that the second index must be incremented by 1
                # for it to work
                idx_ch = np.vstack((len(X_QPE_CENTERS)  -
                                (lut_elev[:,4] - np.min(X_QPE_CENTERS)),
                                lut_elev[:,3] -  np.min(Y_QPE_CENTERS) + 1)).T

                idx_ch = idx_ch.astype(int)
                idx_polar = [lut_elev[:,1], lut_elev[:,2]]

                nanadd_at(heights_qpe, idx_ch, lut_coords[sweep][2][idx_polar[0],
                                                    idx_polar[1]])
                nanadd_at(weights, idx_ch, np.isfinite(lut_coords[sweep][2][idx_polar[0],
                                                    idx_polar[1]]))
                            
                all_heights_qpe[sweep] = heights_qpe / weights

            lut_names.append(str(lut_name))
            lut_objects.append(all_heights_qpe)

    elif lookup_type == 'station_to_qpegrid':
           
        df_stations = constants.METSTATIONS
    
        offset_x = int((neighb_x-1)/2)
        offset_y = int((neighb_y-1)/2)
        
        df_stations = df_stations.append(constants.RADARS)
        stations = df_stations['Abbrev']
    
        lut = {}
        x_qpe = constants.X_QPE
        y_qpe = constants.Y_QPE
        y,x = np.meshgrid(y_qpe,x_qpe)
        # Get x and y of all radar pixels                 
        for station in stations:
            lut[station] = {}
            station_data = df_stations[df_stations.Abbrev == station]
            y_sta =  float(station_data.Y)
            x_sta = float(station_data.X)
            
            # For x the columns in the Cartesian lookup tables are lower bounds
            # e.g. x = 563, means that radar pixels are between 563 and 564
            y_llc_sta = int(y_sta / constants.CART_GRID_SIZE)
            # For y the columns in the Cartesian lookup tables are upper bounds
            # e.g. x = 182, means that radar pixels are between 181 and 182            
            x_llc_sta = int(np.ceil(x_sta / constants.CART_GRID_SIZE))
        
            for i in range(-offset_y, offset_y + 1):
                for j in range(-offset_x, offset_x + 1):
                    x_llc = x_llc_sta + j
                    y_llc = y_llc_sta + i 
                    # Find index of station in cart grid
                    idx =  np.where(np.logical_and(x == x_llc, 
                                  y == y_llc))
                 
                    idx = [int(idx[0]),int(idx[1])]
                    key = str(i)+str(j)
                    if len( idx):
                        lut[station][key] = idx
        
        lut_name = Path(LOOKUP_FOLDER, 'lut_station_to_qpegrid.p')
        lut_names.append(str(lut_name))
        lut_objects.append(lut)

    elif 'cartcoords_rad' in lookup_type :
        if lookup_type == 'cardcoords_radL':
            res = 'L'
        elif lookup_type == 'cartcoords_radH':
            res = 'H'
        converter = GPSConverter()
        
        for r in radar:
            lut = {}
            lut_name =  Path(LOOKUP_FOLDER, 'lut_' + lookup_type+'{:s}.p'.format(r))
            logging.info('Creating lookup table {:s}'.format(str(lut_name)))
            files = sorted(glob.glob(str(Path(folder_radar_samples, 
                                              'M{:s}{:s}*'.format(res, r)))))
            rad_pos = constants.RADARS[constants.RADARS.Abbrev == r]
            x_rad = float(rad_pos.X)
            y_rad = float(rad_pos.Y)
            z_rad = float(rad_pos.Z)
            lat_rad = converter.LV03toWGS84(x_rad,y_rad,z_rad)[0]
            
            RE = get_earth_radius(lat_rad)
            
            coords_x = {}
            coords_y = {}
            coords_z = {}
                
            for sweep, f in enumerate(files):
                f = ObjStorage.check_file(f)
                # Get x and y of all radar pixels
                data = read_metranet(f)
                range_vec = data.range['data']
                elevation_angle = np.deg2rad(constants.ELEVATIONS[sweep])
                az_angle = np.deg2rad(data.azimuth['data'][0:360])
                # Use 4/3 earth radius model
                temp = np.sqrt(range_vec** 2 + (constants.KE * RE) ** 2 + 2 * range_vec *
                               constants.KE * RE * np.sin(elevation_angle))
                
                h = temp - constants.KE * RE + float(data.altitude['data'])
                s = constants.KE * RE * np.arcsin((range_vec * np.cos(elevation_angle)) /
                                    (constants.KE * RE + h))
            
                # Get coordiantes of all radar gates
                coord_x = (x_rad + np.cos(az_angle) *  s[:,None]).T
                coord_y = (y_rad + np.sin(az_angle) *  s[:,None]).T
                coord_z = h
                coord_z = np.tile(coord_z, reps=(coord_x.shape[0],1))
                
                coords_z[sweep] = coord_z
                coords_x[sweep] = coord_x
                coords_y[sweep] = coord_y

                lut[sweep] = []
                lut[sweep].append(coord_y)
                lut[sweep].append(coord_x)
                lut[sweep].append(coord_z)
                lut[sweep] = np.array(lut[sweep])
            
            lut_names.append(str(lut_name))
            lut_objects.append(lut)
          
            
    elif lookup_type == 'qpegrid_to_rad':
        ''' Currently it just uses the csv files of Marco Boscacci and 
        converts them to pickle to be consistent with the other luts'''
        for r in radar:
            lut_name =  Path(LOOKUP_FOLDER, 'lut_' + lookup_type+'{:s}.p'.format(r))
            lut_boscacci = Path(lut_boscacci, 'lut_PL{:s}.csv'.format(r))
            lut_boscacci = ObjStorage.check_file(lut_boscacci)
            lut = np.array(pd.read_csv(str(lut_boscacci)))

            lut_names.append(str(lut_name))
            lut_objects.append(lut)
    else:
        raise ValueError('Invalid lookup type!')
 
    # Dump luts to disk and upload to cloud
    for i in range(len(lut_objects)):
        pickle.dump(lut_objects[i],
                            open(lut_names[i],'wb'))
        try:
            ObjStorage.upload_file(lut_names[i])
        except AttributeError as e:
	        print('Could not upload lut {:s} to cloud, please verify AWS_KEY env variable...'.format(lut_names[i]))

def _WGS_to_COSMO(coords_WGS, SP_coords = (-43,10)): 
     if isinstance(coords_WGS, tuple): 
         coords_WGS=np.vstack(coords_WGS) 
     if isinstance(coords_WGS, np.ndarray ): 
         if coords_WGS.shape[0]<coords_WGS.shape[1]: 
             coords_WGS=coords_WGS.T 
         lon = coords_WGS[:,1] 
         lat = coords_WGS[:,0] 
         input_is_array=True 
     else: 
         lon=coords_WGS[1] 
         lat=coords_WGS[0] 
         input_is_array=False 
 
     SP_lon=SP_coords[1] 
     SP_lat=SP_coords[0] 

 
     lon = (lon*np.pi)/180 # Convert degrees to radians 
     lat = (lat*np.pi)/180 
 
 
     theta = 90+SP_lat # Rotation around y-axis 
     phi = SP_lon # Rotation around z-axis  
 
     phi = (phi*np.pi)/180 # Convert degrees to radians 
     theta = (theta*np.pi)/180 
 
     x = np.cos(lon)*np.cos(lat) # Convert from spherical to cartesian coordinates 
     y = np.sin(lon)*np.cos(lat) 
     z = np.sin(lat) 
 
 
     x_new = np.cos(theta)*np.cos(phi)*x + np.cos(theta)*np.sin(phi)*y + np.sin(theta)*z 
     y_new = -np.sin(phi)*x + np.cos(phi)*y 
     z_new = -np.sin(theta)*np.cos(phi)*x - np.sin(theta)*np.sin(phi)*y + np.cos(theta)*z 
 
 
     lon_new = np.arctan2(y_new,x_new) # Convert cartesian back to spherical coordinates 
     lat_new = np.arcsin(z_new) 
 
 
     lon_new = (lon_new*180)/np.pi # Convert radians back to degrees 
     lat_new = (lat_new*180)/np.pi 
 
     if input_is_array: 
         coords_COSMO = np.vstack((lat_new, lon_new)).T 
     else: 
         coords_COSMO=np.asarray([lat_new, lon_new]) 
  
     return coords_COSMO.astype('float32') 
