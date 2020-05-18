#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to read MeteoSwiss products

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019
"""


import os
from imageio import imread, imwrite
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pyart.aux_io import read_metranet
from pyart.aux_io import read_cartesian_metranet
from pyart.util import join_radar
from matplotlib import colors
from PIL import Image

import datetime
import xmltodict

from . import constants
from .utils import sweepnumber_fromfile, hex_to_rgb
from . import retrieve_data as retrieve
from .lookup import get_lookup

def read_xls(xls_file):
    """Reads an excel file such as those used for CPC vlaidation

    Parameters
    ----------
    fname : str
        Full path of the excel file to be read
    
    Returns
    -------
    The excel file as a Pandas dataframe
    """
    
    data = pd.read_excel(xls_file, sheet_name  = None)
    keys = list(data.keys())
    hourly_keys = []
    for k in keys:
        if 'Data Hourly' in k:
            hourly_keys.append(k)
    out = pd.concat([data[k] for k in hourly_keys])
    return out


def read_status(status_file, add_wet_radome = False):
    """Reads a radar xml status file

    Parameters
    ----------
    fname : str
        Full path of the status xml file to be read
    add_wet_Radome : boolean (optional)
        For older files, there is not information about the wet radome. 
        If this is true, the script will estimate the wet radome precipitation
        as a 3 x 3 mean of the RZC product at the given time (as is done for
        more recent files)
        
    Returns
    -------
    The status as a python dict
    """
    
    # Reads a xml status file 
    status = xmltodict.parse(open(status_file,'r').read())
    
    #  if wetradome is missing computes it
    radstat = status['status']['sweep'][-1]['RADAR']['STAT']
    
    if 'WET_RADOME' not in radstat.keys() and add_wet_radome:
        # get radar and time from filename
        bname = os.path.basename(status_file)
        radar = bname[2]
        time = datetime.strptime(bname[3:12],'%y%j%H%M')
        
        file_rzc = retrieve.retrieve_prod('/tmp/',time,time,'RZC')[0]
        rzc = read_cart(file_rzc)
        rzc[rzc < constants.MIN_RZC_VALID] = 0
        
        radprecip = []
        lut = get_lookup('station_to_qpegrid')
        for i in range(-1,2): # 3 x 3
            for j in range(-1,2):
                coord = lut[radar]['{:d}{:d}'.format(i,j)]
                radprecip.append(rzc[coord[0],coord[1]])
                
        radprecip = np.nanmean(radprecip)
        # Assign radprecip following the structure in more recent files
        if radprecip == 0:
            radstat['WET_RADOME'] = None
        else:
            dic = {'wetradome_mmh':{'@value': radprecip}}
            radstat['WET_RADOME'] = dic
            
    return status


def read_polar(polar_files, physic_value = True):
    '''
    Reads a polar radar metranet file as a pyart radar instance, if multiple
    files corresponding to multiple elevations (sweeps) are provided they are
    merged into a single pyart instance
    
    Parameters
    ----------
    polar_files: str or list of strings
        Can be either:
        - Full path of the file to read as a a string
        - string with wildcard pointing to the files to read,
            e.g. ...../MLA192711055.*
        - a list of full filepaths

    Returns
    -------
    sweepnumbers: list
        list of sweeps numbers corresponding to all files that were read
        
    merged :  a pyart Radar instance
    '''
        
    if type(polar_files) == str:
        if '*' in polar_files:
            all_files = np.sort(glob.glob(polar_files.split('.')[0]+'*'))
        else:
            all_files = [polar_files]
    elif type(polar_files) in [list, np.ndarray]:
        all_files = polar_files
        
    else:
        raise ValueError('Invalid input type, must be list/array or string')
    
    sweepnumbers = []
    radar = None
    for f in all_files:
        try:
            r = read_metranet(f, physic_value = True)

            if not radar:
                radar = r
            else:
                radar = join_radar(radar, r)
            sweepnumbers.append(sweepnumber_fromfile(f))
        except:
            pass

    return sweepnumbers, radar
    
    
def read_cart(cart_file):
    '''
    Generic function that reads a Cartesian radar file, either in gif or 
    binary format (ELDES or as generated by the qpe module of this library)
    and converts its data to floating numbers
    
    Parameters
    ----------
    cart_file: str 
        Full path of the file to read as a a string
           
    Returns
    ----------
    The cartesian data in a numpy array
        
    '''
    # Generic reader
    extension = os.path.splitext(cart_file)[1]
    
    if extension == '.gif' or 'CPC' in cart_file:
        data = read_gif(cart_file)
    elif 'RF' in cart_file:
         # Get from filesize if it is DN or float
         size = os.path.getsize(cart_file)
         DN = False
         nbins_x, nbins_y = constants.NBINS_X, constants.NBINS_Y
         if size == nbins_x * nbins_y:
             DN = True
         if DN:
             data = np.fromfile(cart_file, dtype = 
                                'B').reshape(nbins_x, nbins_y)
             data = constants.SCALE_CPC[data] # Convert to float
         else:
             data = np.fromfile(cart_file, dtype = 
                                np.float32).reshape(nbins_x, nbins_y)
    else:
        data = read_cartesian_metranet(cart_file, physic_value = True)
        data = list(data.fields.values())[0]['data'].data.copy()
        data[data < constants.MIN_RZC_VALID] = 0
        data = np.flipud(np.squeeze(data))
    return data
    
def save_gif(gif_file, precip):
    '''
    Saves a precipitation map in float to a gif file (AQC format)
    
    Parameters
    ----------
    gif_file: str 
         Full path of the file to write as a a string
    precip : 2D array
        2D array containing the precipitation intensities
           
        
    '''
    # Rescale ascending order of values
    scale = constants.SCALE_RGB
    idx = np.searchsorted(scale['values'],precip.ravel())
    
    N,M = precip.shape
    
    dn = np.reshape(idx,(N,M)).astype(np.uint8)
    dn[precip < 0] = 255 # ensure correct masking
    cmap =  colors.ListedColormap(scale['colors'])
 
    pil_im = Image.fromarray(dn, mode='P')
    pil_im.save(
    fp=gif_file,
    loop=0,
    palette=cmap
    )

    
def read_gif(gif_file):
    '''
    Reads a Cartesian radar file in gif format
    
    Parameters
    ----------
    gif_file: str 
         Full path of the file to read as a a string
           
    Returns
    ----------
    The cartesian data in a numpy array
        
    '''
    
    scale  = constants.SCALE_RGB
    colors = np.array([hex_to_rgb(c) for c in scale['colors']])
    values = scale['values']
    
    colors_bin = colors[:,0]*255**2 + colors[:,1]*255 + colors[:,2]

    img = imread(gif_file).astype(np.uint64)

    img_bin = img[:,:,0]*255**2 + img[:,:,1]*255 + img[:,:,2]

    precip = np.empty((img.shape[0],img.shape[1]))
    precip[:] = -1
    for i in range(len(colors_bin)):
        precip[img_bin == colors_bin[i]] = values[i]
        
    return precip
    
    
def read_station_data(gauge_file):
    
    # TODO
    
    '''
    Reads gauge data as given by the Climap software
    
    Parameters
    ----------
    gauge_file: str 
         Full path of the file to read as a a string
           
    Returns
    ----------
    The cartesian data in a numpy array
        
    '''
    
    idx_header = 0
    with open(gauge_file, encoding='latin-1') as ff:
        l = 1
        while l != '\n':
            l = ff.readline()
            idx_header += 1

    data = pd.read_csv(gauge_file, skiprows = idx_header )
    
    
    data[:,-1][data[:,-1] > 300] = np.nan
    idx = data[:,0]
    dates = np.array([datetime.datetime(year = c[0], 
                                month = c[1],
                                day = c[2],
                                hour = c[3],
                                minute = c[4]) for c in data[:,1:6].astype(int)])
    
    unique_idx = np.unique(idx)
    
    data_by_station = {}
    all_abbrev = np.array(constants.STATIONS.Abbrev).astype(str)
    all_idx = np.array(constants.STATIONS.ID)
    for k in unique_idx:
       
        abbrev = all_abbrev[all_idx == k][0]
        data_by_station[abbrev] = data[idx == k,-1] 
        
    return dates[idx == k], data_by_station


def read_vpr(xml_file, radar = None):
    
    '''
    Reads a vpr xml file and returns an interpolator that can be used to obtain
    the vpr correction at any altitude
    
    Parameters
    ----------
    vpr_file: str 
         Full path of the file to read
     radar : char (optional)
         the radar is required to know the vpr reference height, it needs
         to be either 'A','D','L','P' or 'W', if not specified, it will be
         inferred from the vpr filename
             
    Returns
    ----------
    An interpolator that returns the vpr correction as a function of
    the height, the correction is a multiplicative factor
        
    '''

    if radar == None:
        # Infer radar from filename
        radar = os.path.basename(xml_file)[2]
        
    with open(xml_file) as fd:
        doc = xmltodict.parse(fd.read())
        
    vpr =  [float(doc['VPR']['DATA']['slice'][i]['value']) 
                for i in range(len(doc['VPR']['DATA']['slice']))]
    
    
    alt = np.arange(len(vpr)) * float(doc['VPR']['HEADER']['vpr_res'])
    
    vpr = np.array(vpr)
    
    ref = np.argmin(np.abs(alt - constants.VPR_REF_HEIGHTS[radar]))
    vpr_norm = vpr[ref] / vpr 
    corr_max_lin = 10 ** (0.1 * constants.MAX_VPR_CORRECTION_DB)
    
    vpr_norm[vpr_norm < 1./corr_max_lin] = 1/corr_max_lin
    vpr_norm[vpr_norm>corr_max_lin] = corr_max_lin

    maxval_lin = 10**(0.1* constants.MAX_VPR_CORRECTION_DB)
    interp = interp1d(alt,vpr_norm, bounds_error = False,
                    fill_value = maxval_lin)
    
    return interp
