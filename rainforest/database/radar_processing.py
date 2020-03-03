#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:10:43 2018

@author: wolfensb
"""

import numpy as np
from pyart.util import rolling_window
from pyart.config import get_fillvalue, get_field_name, get_metadata
from pyart.correct.phase_proc import _correct_sys_phase
from pyart.correct import calculate_attenuation_zphi


def kdp_leastsquare_single_window(
        radar, wind_len=11, min_valid=6, phidp_field=None, kdp_field=None):
    """
    Compute the specific differential phase (KDP) from differential phase data
    using a piecewise least square method. For optimal results PhiDP should
    be already smoothed and clutter filtered out.

    Parameters
    ----------
    radar : Radar
        Radar object.
    wind_len : int
        The lenght of the moving window
    min_valid : int
        Minimum number of valid bins to consider the retrieval valid
    phidp_field : str
        Field name within the radar object which represent the differential
        phase shift. A value of None will use the default field name as
        defined in the Py-ART configuration file.
    kdp_field : str
        Field name within the radar object which represent the specific
        differential phase shift. A value of None will use the default field
        name as defined in the Py-ART configuration file.

    Returns
    -------
    kdp_dict : dict
        Retrieved specific differential phase data and metadata.

    """
    # parse the field parameters
    if phidp_field is None:
        phidp_field = get_field_name('differential_phase')
    if kdp_field is None:
        kdp_field = get_field_name('specific_differential_phase')

    # extract fields from radar
    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data']

    kdp_dict = get_metadata(kdp_field)
    kdp_dict['data'] = leastsquare_method_scan(
        phidp, radar.range['data'], wind_len=wind_len, min_valid=min_valid)


    return kdp_dict


def leastsquare_method_scan(phidp, rng_m, wind_len=11, min_valid=6):
    """
    Compute the specific differential phase (KDP) from differential phase data
    using a piecewise least square method. For optimal results PhiDP should
    be already smoothed and clutter filtered out. This function computes the
    whole radar volume at once

    Parameters
    ----------
    phidp : masked array
        phidp field
    rng_m : array
        radar range in meters
    wind_len : int
        the window length
    min_valid : int
        Minimum number of valid bins to consider the retrieval valid

    Returns
    -------
    kdp : masked array
        Retrieved specific differential phase field

    """
    # we want an odd window
    if wind_len % 2 == 0:
        wind_len += 1
    half_wind = int((wind_len-1)/2)

    # initialize kdp
    nrays, nbins = np.shape(phidp)
    kdp = np.ma.zeros((nrays, nbins))
    kdp[:] = np.ma.masked
    kdp.set_fill_value(get_fillvalue())

    # check which gates are valid
    valid = np.logical_not(np.ma.getmaskarray(phidp))
    valid_wind = rolling_window(valid, wind_len)
    mask_wind = np.logical_not(valid_wind)
    nvalid = np.sum(valid_wind, axis=-1, dtype=int)
    ind_valid = np.logical_and(
        nvalid >= min_valid, valid[:, half_wind:-half_wind]).nonzero()
    nvalid = nvalid[ind_valid]
    del valid, valid_wind

    rng_mat = np.broadcast_to(rng_m.reshape(1, nbins), (nrays, nbins))
    rng_mat = rolling_window(rng_mat/1000., wind_len)
    rng_wind_ma = np.ma.masked_where(mask_wind, rng_mat, copy=False)
    phidp_wind = rolling_window(phidp, wind_len)

    rng_sum = np.ma.sum(rng_wind_ma, -1)[ind_valid]
    rng_sum2 = np.ma.sum(np.ma.power(rng_wind_ma, 2.), -1)[ind_valid]

    phidp_sum = np.ma.sum(phidp_wind, -1)[ind_valid]
    rphidp_sum = np.ma.sum(phidp_wind * rng_wind_ma, -1)[ind_valid]
    del rng_wind_ma, phidp_wind

    kdp[ind_valid[0], ind_valid[1]+half_wind] = (
        0.5*(rphidp_sum-rng_sum*phidp_sum/nvalid) /
        (rng_sum2-rng_sum*rng_sum/nvalid))

    return kdp


def smooth_phidp_single_window(
        radar, ind_rmin=10, ind_rmax=500, min_rcons=11, zmin=20., zmax=40,
        wind_len=11, min_valid=6, psidp_field=None, refl_field=None,
        phidp_field=None):
    """
    correction of the system offset and smoothing using one window

    Parameters
    ----------
    radar : Radar
        Radar object for which to determine the system phase.
    ind_rmin, ind_rmax : int
        Min and max range index where to look for continuous precipitation
    min_rcons : int
        The minimum number of consecutive gates to consider it a rain cell.
    zmin, zmax : float
        Minimum and maximum reflectivity to consider it a rain cell
    wind_len : int
        Length of the moving window used to smooth
    min_valid : int
        Minimum number of valid bins to consider the smooth data valid
    psidp_field : str
        Field name within the radar object which represent the differential
        phase shift. A value of None will use the default field name as
        defined in the Py-ART configuration file.
    refl_field : str
        Field name within the radar object which represent the reflectivity.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    phidp_field : str
        Field name within the radar object which represent the corrected
        differential phase shift. A value of None will use the default field
        name as defined in the Py-ART configuration file.

    Returns
    -------
    phidp_dict : dict
        The corrected phidp field

    """
    # parse the field parameters
    if psidp_field is None:
        psidp_field = get_field_name('differential_phase')
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if phidp_field is None:
        phidp_field = get_field_name('corrected_differential_phase')

    if psidp_field in radar.fields:
        psidp = radar.fields[psidp_field]['data']
    else:
        raise KeyError('Field not available: ' + psidp_field)
    if refl_field in radar.fields:
        refl = radar.fields[refl_field]['data']
    else:
        raise KeyError('Field not available: ' + refl_field)

    # correction of system offset
    phidp = _correct_sys_phase(
        psidp, refl, radar.nsweeps, radar.nrays, radar.ngates,
        radar.sweep_start_ray_index['data'],
        radar.sweep_end_ray_index['data'], ind_rmin=ind_rmin, zmin=zmin,
        zmax=zmax, ind_rmax=ind_rmax, min_rcons=min_rcons)

    phidp = smooth_masked_scan(phidp, wind_len=wind_len, min_valid=min_valid,
                               wind_type='median')

    # create specific differential phase field dictionary and store data
    phidp_dict = get_metadata(phidp_field)
    phidp_dict['data'] = phidp

    return phidp_dict


def smooth_masked_scan(raw_data, wind_len=11, min_valid=6, wind_type='median'):
    """
    smoothes the data using a rolling window.
    data with less than n valid points is masked.
    Processess the entire scan at once

    Parameters
    ----------
    raw_data : float masked array
        The data to smooth.
    window_len : float
        Length of the moving window
    min_valid : float
        Minimum number of valid points for the smoothing to be valid
    wind_type : str
        type of window. Can be median or mean

    Returns
    -------
    data_smooth : float masked array
        smoothed data

    """
    valid_wind = ['median', 'mean']
    if wind_type not in valid_wind:
        raise ValueError(
            "Window "+window+" is none of " + ' '.join(valid_windows))

    # we want an odd window
    if wind_len % 2 == 0:
        wind_len += 1
    half_wind = int((wind_len-1)/2)

    # initialize smoothed data
    nrays, nbins = np.shape(raw_data)
    data_smooth = np.ma.zeros((nrays, nbins))
    data_smooth[:] = np.ma.masked
    var = np.ma.zeros((nrays, nbins))
    var[:] = np.ma.masked
    # data_smooth.set_fill_value(get_fillvalue())

    # check which gates are valid
    valid = np.logical_not(np.ma.getmaskarray(raw_data))
    valid_rolled = rolling_window(valid, wind_len)
    nvalid = np.sum(valid_rolled, axis=-1, dtype=int)
    ind_valid = np.logical_and(
        nvalid >= min_valid, valid[:, half_wind:-half_wind]).nonzero()
    del valid, valid_rolled, nvalid

    data_wind = rolling_window(raw_data, wind_len)
    # get rolling window and mask data
    data_smooth[ind_valid[0], ind_valid[1]+half_wind] = eval(
        'np.ma.'+wind_type +
        '(data_wind, overwrite_input=True, axis=-1)')[ind_valid]


#    
    return data_smooth

if __name__ == '__main__':
    from MCH_constants import read_metranet_allsweeps
    import time
    t0 = time.time()
    m = read_metranet_allsweeps('/scratch/wolfensb/files_qpe/radar/PLA1406000057U.001')
    DCFG_phidp_1w = {}
    DCFG_phidp_1w['rmin'] = 1000.
    DCFG_phidp_1w['rmax'] = 50000.
    DCFG_phidp_1w['rcell'] = 1000.
    DCFG_phidp_1w['Zmin'] = 20.
    DCFG_phidp_1w['Zmax'] = 40.
    DCFG_phidp_1w['rwind'] = 6000.
    
    kdp = compute_kdp(m, DCFG_phidp_1w )
    print(t0-time.time())