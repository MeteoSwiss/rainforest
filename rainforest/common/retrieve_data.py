#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to retrieve MeteoSwiss products from the archives

Daniel Wolfensberger, Rebecca Gugerli
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch, rebecca.gugerli@epfl.ch
December 2019, July 2022
"""


import numpy as np
import os
import zipfile
import datetime
import glob
import subprocess
import netCDF4
import logging
import fnmatch
import re
from textwrap import dedent

import pandas as pd # function used in retrieve_hzt_prod
from . import constants 
from .lookup import get_lookup
from .utils import round_to_hour
from . import io_data as io # avoid circular


#-----------------------------------------------------------------------------------------
def retrieve_hzt_prod(folder_out, start_time, end_time,pattern_type='shell'):
    """ Retrieves the preprocessed HZT products from the CSCS repository for a specified
    time range, unzips them and places them in a specified folder

    Parameters
    ----------
    
    folder_out: str
        directory where to store the unzipped files
    start_time : datetime.datetime instance
        starting time of the time range
    end_time : datetime.datetime instance
        end time of the time range
    pattern_type: either 'shell' or 'regex' (optional)
        use 'shell' for standard shell patterns, which use * as wildcard
        use 'regex' for more advanced regex patterns
                
    Returns
    -------
    A list containing all the filepaths of the retrieved files
   
    """
    dt = datetime.timedelta(hours=1)
    delta = end_time - start_time
    if delta.total_seconds()== 0:
        times = [start_time]
    else:
        times = start_time + np.arange(int(delta.total_seconds()/(60*60)) + 2)*dt
    dates = []
    for t in times:
        dates.append(datetime.datetime(year = t.year, month = t.month,
                                       day = t.day))
    dates = np.unique(dates)
    
    t0 = start_time
    t1 = end_time
    
    all_files = []
    for i, d in enumerate(dates):
        if i == 0:
            start_time = datetime.datetime(year = t0.year, month = t0.month,
                                           day = t0.day, hour=t0.hour)
            #print('*first start time: ', start_time)
        else:
            start_time = datetime.datetime(year = d.year, month = d.month,
                                           day = d.day)
            #print('*all other start times', start_time)
        if (i == len(dates) - 1):
            end_time = datetime.datetime(year = t1.year, month = t1.month,
                                         day = t1.day, hour=t1.hour) + datetime.timedelta(hours=1)
        else:
            end_time = datetime.datetime(year = d.year, month = d.month,
                                           day = d.day, hour=23)
            #print('*end_time: ', end_time)

        files = _retrieve_hzt_prod_daily(folder_out, start_time, end_time,
                                            pattern_type)

        if files != None:
            all_files.extend(files)
        
    return all_files

def retrieve_hzt_RT(tstep):
    
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
    content_zip = np.array([c for c in os.listdir(folder_in) 
                            if (len(c.split('.')) == 2) and (int(c.split('.')[-1])>=800)])
    
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

#-----------------------------------------------------------------------------------------
def _retrieve_hzt_prod_daily(folder_out, start_time, end_time, pattern_type = 'shell'):
    
    """ Retrieves the preprocessed HZT products from the CSCS repository for a day,
        Only used in for the function retrieve_hzt_prod

    Parameters
    ----------
    
    folder_out: str
        directory where to store the unzipped files
    start_time : datetime.datetime instance
        starting time of the time range
    end_time : datetime.datetime instance
        end time of the time range
    pattern_type: either 'shell' or 'regex' (optional)
        use 'shell' for standard shell patterns, which use * as wildcard
        use 'regex' for more advanced regex patterns
                
    Returns
    -------
    A list containing all the filepaths of the retrieved files
   
    """
    
    folder_out += '/'
    
    suffix =  str(start_time.year)[-2:] + str(start_time.timetuple().tm_yday).zfill(3)
    folder_in = constants.FOLDER_ISO0 + str(start_time.year) + '/' +  suffix + '/'
    name_zipfile = 'HZT'+ suffix+'.zip'
    
    try:
        # Get list of files in zipfile
        zipp = zipfile.ZipFile(folder_in + name_zipfile)
        content_zip = np.sort(np.array(zipp.namelist()))
        
        # Sort filelist to most recent prediction
        content_filt = np.array([c for c in content_zip if c.endswith('800')])
        times_filt = np.array([datetime.datetime.strptime(c[3:12],
                            '%y%j%H%M')+datetime.timedelta(hours=int(c[-2::])) for c in content_filt])
        content_filt = content_filt[np.where((times_filt >= start_time) & (times_filt <= end_time))]
        times_filt = times_filt[np.where((times_filt >= start_time) & (times_filt <= end_time))]    
    except:
        logging.error('Zip file with HZT data does not exist for '+start_time.strftime('%d-%b-%y'))
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
        
    files_to_retrieve = ' '.join(content_filt[conditions])
    
    # Check if files are already unzipped (saves time if they already exist)
    for fi in content_filt[conditions]:
        if os.path.exists(folder_out+fi):
            files_to_retrieve = files_to_retrieve.replace(fi,'')
            
    # Only unzip if at least one file does not exist
    if len(files_to_retrieve.strip()) > 0:
        logging.info('Unzippping: '+ files_to_retrieve)   
        cmd = 'unzip -j -o -qq "{:s}" {:s} -d {:s}'.format(folder_in + name_zipfile,
            files_to_retrieve , folder_out)
        subprocess.call(cmd, shell=True)
        
    files = sorted(np.array([folder_out + c for c in
                            content_filt[conditions]]))    
    
    return files

#-----------------------------------------------------------------------------------------
def get_COSMO_T(time, sweeps = None, radar = None):
    
    """Retrieves COSMO temperature data from the CSCS repository, and 
    interpolates them to the radar gates, using precomputed lookup tables

    Parameters
    ----------
    time : datetime.datetime instance
        the time at which to get the COSMO data in datetime format
    sweeps: list of integers
         specify which sweeps (elevations) need to be retrieved in the form
         of a list, if not specified, all 20 will be retrieved
    radar: list of chars
        list of radars for which to retrieve COSMO data, if not specified
        all 5 radars will be used ('A','L','D','W','P')
            
    Returns
    -------
    T_at_radar : dict
        A dict containing the temperature at the radar gates, in the following form:
        dict[radar]['T'][sweep_number]
    
    """
        
    if np.any(radar == None):
        radar = constants.RADARS['Abbrev']
 
    if np.any(sweeps == None):
        sweeps = range(1,21)

    if time > constants.COSMO1E_START and time < constants.TIMES_COSMO1E_T[0]:
        msg = """No COSMO1E temp file available for this timestep,
        retrieving COSMO1 temp file instead
        """
        logging.warning(dedent(msg))

    if time > constants.COSMO1_START and time < constants.TIMES_COSMO1_T[0]:
        msg = """No temp file available for this timestep, using the slow 
        more exhaustive function instead
        """
        logging.warning(dedent(msg))
        return get_COSMO_variables(time, ['T'], sweeps, radar)
        
    elif time < constants.COSMO1_START:
        msg = """
        Currently all COSMO-2 files have been archived and it is not possible
        to retrieve them with this function, sorry
        """
        raise ValueError(dedent(msg))
        
    # Get the closest COSMO-1 or 2 file in time
    if time < constants.TIMES_COSMO1E_T[0]: 
        times_cosmo = constants.TIMES_COSMO1_T
        files_cosmo = constants.FILES_COSMO1_T
    else:
        times_cosmo = constants.TIMES_COSMO1E_T
        files_cosmo = constants.FILES_COSMO1E_T

    idx_closest = np.where(time >= times_cosmo)[0][-1]
    file_COSMO = files_cosmo[idx_closest]
    dt = (time - times_cosmo[idx_closest]).total_seconds()

    file_COSMO = netCDF4.Dataset(file_COSMO)
    # old version
    #idx_time = np.argmin(np.abs(dt - file_COSMO.variables['time'][:]))
    # new:
    # The dimension 'time' in the COSMO-1E comes in the unit of hours since filename
    cosmo_hours = [times_cosmo[idx_closest]+datetime.timedelta(hours=(int(hh))) for hh in file_COSMO.variables['time'][:]]
    idx_time = np.argmin(np.abs([(time-hh).total_seconds() for hh in cosmo_hours]))
       
    T = np.squeeze(file_COSMO.variables['T'][idx_time,:,:,:])

    T_at_radar = {}
    for r in radar:
        lut_rad = get_lookup('cosmo1T_to_rad', r)
        T_at_radar[r] = {'T':{}}
        for s in sweeps:
  
            # Finally get temperature at radar
            m1 = lut_rad[s]['idx0']
            m2 = lut_rad[s]['idx1']
            m3 = lut_rad[s]['idx2']
            mask = lut_rad[s]['mask']
            # Finally get variables at radar
            T_at_radar[r]['T'][s] = np.ma.array(T[m1, m2, m3], mask = mask)
    file_COSMO.close()        
    
    return T_at_radar

#-----------------------------------------------------------------------------------------
def get_COSMO_variables(time, variables, sweeps = None, radar = None,
                        tmp_folder = '/tmp/', cleanup = True):
    
    """Retrieves COSMO data from the CSCS repository, and 
    interpolates them to the radar gates, using precomputed lookup tables
    This is a more generic but much slower function than the previous one,
    as it reads all COSMO variables directly from the GRIB files

    Parameters
    ----------
    time : datetime.datetime instance
        the time at which to get the COSMO data in datetime format
    variables: list of strings
        List of COSMO variables to retrieve, ex. P, T, QV, QR, RH, etc...
    sweeps: list of integers (optional)
         specify which sweeps (elevations) need to be retrieved in the form
         of a list, if not specified, all 20 will be retrieved
    radar = list of chars (optional)
        list of radars for which to retrieve COSMO data, if not specified
        all 5 radars will be used ('A','L','D','W','P')
    tmp_folder = str (optional)
        Directory where to store the extracted files
    cleanup = boolean (optional)
        If true all extracted files will be deleted before returning the output
        (recommended)
        
    Returns
    -------
    A dict containing the COSMO variables at the radar gates, in the following
    form: dict[radar][variables][sweep_number]
    
    """
    
    if np.any(radar == None):
        radar = constants.RADARS['Abbrev']
 
    if np.any(sweeps == None):
        sweeps = range(1,21)
        
    if time < constants.COSMO1_START:
        msg = """
        Currently all COSMO-2 files have been archived and it is not possible
        to retrieve them with this function, sorry
        """
        raise ValueError(dedent(msg))
        
    # Round time to nearest hour
    t_near = round_to_hour(time)
    
    cosmo_version = None
    if t_near < constants.COSMO1E_START:
        folder_cosmo = constants.FOLDER_COSMO1
        subfolder = ''
        cosmo_version = '1'
    else:
        folder_cosmo = constants.FOLDER_COSMO1E
        subfolder = 'det/'
        cosmo_version = '1e'

    # Get the closest COSMO-1 or 2 file in time
    grb = folder_cosmo + 'ANA{:s}/{:s}laf{:s}'.format(str(t_near.year)[2:],
                                subfolder,
                                datetime.datetime.strftime(t_near,'%Y%m%d%H')) 
    
    # Extract fields and convert to netCDF
    list_variables = ','.join(variables)
    tmp_name = tmp_folder + os.path.basename(grb) + '_filtered'
    
    cmd_filter = {'{:s} {:s} --force -s {:s} -o {:s}'.format(
                constants.FILTER_COMMAND, grb, list_variables, tmp_name)}

    subprocess.call(cmd_filter, shell = True)
    
    cmd_convert = {'{:s} --force -o {:s} nc {:s}'.format(
            constants.CONVERT_COMMAND, tmp_name + '.nc', tmp_name)}

    subprocess.call(cmd_convert, shell = True)
    
    # Finally interpolate to radar grid
    file_COSMO = netCDF4.Dataset(tmp_name + '.nc')
    
    # Interpolate for all radars and sweeps
    var_at_radar = {}
    for r in radar:
        lut_rad = get_lookup('cosmo{:s}_to_rad'.format(cosmo_version), r)
        var_at_radar[r] = {}
        for v in variables:
            data = np.squeeze(file_COSMO.variables[v][:])
            var_at_radar[r][v] = {}
            for s in sweeps:
                m1 = lut_rad[s]['idx0']
                m2 = lut_rad[s]['idx1']
                m3 = lut_rad[s]['idx2']
                mask = lut_rad[s]['mask']
                # Finally get variables at radar
                d = np.ma.array(data[m1, m2, m3], mask = mask)
                var_at_radar[r][v][s] = d
    file_COSMO.close() 
    if cleanup:
        os.remove(tmp_name)
        os.remove(tmp_name + '.nc')
        
    return var_at_radar

#-----------------------------------------------------------------------------------------
def retrieve_prod(folder_out, start_time, end_time, product_name,
                  pattern = None, pattern_type = 'shell', sweeps = None):
    
    """ Retrieves radar data from the CSCS repository for a specified
    time range, unzips them and places them in a specified folder

    Parameters
    ----------
    
    folder_out: str
        directory where to store the unzipped files
    start_time : datetime.datetime instance
        starting time of the time range
    end_time : datetime.datetime instance
        end time of the time range
    product_name: str
        name of the product, as stored on CSCS, e.g. RZC, CPCH, MZC, BZC...
    pattern: str
        pattern constraint on file names, can be used for products which contain 
        multiple filetypes, f.ex CPCH folders contain both rda and gif files,
        if only gifs are wanted : file_type = '*.gif'
    pattern_type: either 'shell' or 'regex' (optional)
        use 'shell' for standard shell patterns, which use * as wildcard
        use 'regex' for more advanced regex patterns
    sweeps: list of int (optional)
        For polar products, specifies which sweeps (elevations) must be
        retrieved, if not specified all available sweeps will be retrieved
                
    Returns
    -------
    A list containing all the filepaths of the retrieved files
   
    """
    
    if product_name == 'ZZW' or product_name == 'ZZP': # no vpr for PPM and WEI
        product_name = 'ZZA'

    if product_name == 'CPC':
        folder_out = folder_out + '/CPC'
    if product_name == 'CPCH':
        folder_out = folder_out + '/CPCH'

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    
    dt = datetime.timedelta(minutes = 5)
    delta = end_time - start_time
    if delta.total_seconds()== 0:
        times = [start_time]
    else:
        times = start_time + np.arange(int(delta.total_seconds()/(5*60)) + 1) * dt
    dates = []
    for t in times:
        dates.append(datetime.datetime(year = t.year, month = t.month,
                                       day = t.day))
    dates = np.unique(dates)
    
    t0 = start_time
    t1 = end_time
    
    all_files = []
    for i, d in enumerate(dates):
        if i == 0:
            start_time = t0
        else:
            start_time = datetime.datetime(year = d.year, month = d.month,
                                           day = d.day)
        if i == len(dates) - 1:
            end_time = t1
        else:
            end_time = datetime.datetime(year = d.year, month = d.month,
                                           day = d.day, hour = 23, minute = 59)
        files = _retrieve_prod_daily(folder_out, start_time, end_time,
                                     product_name, pattern, pattern_type,
                                     sweeps)

        all_files.extend(files)
            
    return all_files

def retrieve_prod_RT(time, product_name, 
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

    if pattern != None:
        if pattern_type == 'shell':
            content_zip = [c for c in content_zip 
                        if fnmatch.fnmatch(os.path.basename(c), pattern)]
        elif pattern_type == 'regex':
            content_zip = [c for c in content_zip 
                        if re.match(os.path.basename(c), pattern) != None]
        else:
            raise ValueError('Unknown pattern_type, must be either "shell" or "regex".')

    # Derive datetime of each file
    times_zip = np.array([datetime.datetime.strptime(c[3:12],
                '%y%j%H%M') for c in content_zip])

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

#-----------------------------------------------------------------------------------------
def _retrieve_prod_daily(folder_out, start_time, end_time, product_name,
                  pattern = None, pattern_type = 'shell', sweeps = None):
    
    """ This is a version that works only for a given day (i.e. start and end
    time on the same day)
    """
    if product_name[0:2] == 'MH':
        folder_radar = constants.FOLDER_RADARH
    else:
        folder_radar = constants.FOLDER_RADAR
 
    folder_out += '/'
    
    suffix =  str(start_time.year)[-2:] + str(start_time.timetuple().tm_yday).zfill(3)
    folder_in = folder_radar + str(start_time.year) + '/' +  suffix + '/'
    name_zipfile = product_name + suffix+'.zip'
    
    # Get list of files in zipfile
    zipp = zipfile.ZipFile(folder_in + name_zipfile)
    content_zip = np.array(zipp.namelist())
    
    if pattern != None:
        if pattern_type == 'shell':
            content_zip = [c for c in content_zip 
                           if fnmatch.fnmatch(os.path.basename(c), pattern)]
        elif pattern_type == 'regex':
            content_zip = [c for c in content_zip 
                           if re.match(os.path.basename(c), pattern) != None]
        else:
            raise ValueError('Unknown pattern_type, must be either "shell" or "regex".')
            
    content_zip = np.array(content_zip)
        
    times_zip = np.array([datetime.datetime.strptime(c[3:12],
                          '%y%j%H%M') for c in content_zip])
  
    # Get a list of all files to retrieve
    conditions = np.array([np.logical_and(t >= start_time, t <= end_time)
        for t in times_zip])
    
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
        
    # Create string to retrieve files over unzip
    files_to_retrieve = ' '.join(content_zip[conditions])

    # Check if files are already unzipped (saves time if they already exist)
    for fi in content_zip[conditions]:
        if os.path.exists(folder_out+fi):
            files_to_retrieve = files_to_retrieve.replace(fi,'')

    # Only unzip if at least one file does not exist
    if len(files_to_retrieve.strip()) > 0:
        cmd = 'unzip -j -o -qq "{:s}" {:s} -d {:s}'.format(folder_in + name_zipfile,
            files_to_retrieve , folder_out)
        subprocess.call(cmd, shell=True)
    
    files = sorted(np.array([folder_out + c for c in
                                  content_zip[conditions]]))    
    
    return files

#-----------------------------------------------------------------------------------------
def retrieve_CPCCV(time, stations):
    
    """ Retrieves cross-validation CPC data for a set of stations from
    the xls files prepared by Yanni

    Parameters
    ----------

    time : datetime.datetime instance
        starting time of the time range
    stations : list of str
        list of weather stations at which to retrieve the CPC.CV data
    
    Returns
    -------
    A numpy array corresponding at the CPC.CV estimations at every specified 
    station
    """

    year = time.year

    folder = constants.FOLDER_CPCCV + str(year) + '/'
    
    files = sorted([f for f in glob.glob(folder + '*.xls') if '.s' not in f])
    
    def _start_time(fname):
        bname = os.path.basename(fname)
        times = bname.split('.')[1]
        tend = times.split('_')[1]
        return datetime.datetime.strptime(tend,'%Y%m%d%H00')

    tend = np.array([_start_time(f) for f in files])
    
    match = np.where(time < tend)[0]
    
    if not len(match):
        logging.warn('Could not find CPC CV file for time {:s}'.format(time))
        return np.zeros((len(stations))) + np.nan
    
    data = io.read_xls(files[match[0]])
    
    hour = int(datetime.datetime.strftime(time, '%Y%m%d%H00'))
    idx = np.where(np.array(data['time.stamp']) == hour)[0]
    data_hour = data.iloc[idx]
    data_hour_stations = data_hour.iloc[np.isin(np.array(data_hour['nat.abbr']), 
                                                stations)]
    cpc_cv = []
    cpc_xls = []
    for sta in stations:
        if sta in np.array(data_hour_stations['nat.abbr']):
            cpc_cv.append(float(data_hour_stations.loc[data_hour_stations['nat.abbr'] 
                    == sta]['CPC.CV']))
            cpc_xls.append(float(data_hour_stations.loc[data_hour_stations['nat.abbr']
                    == sta]['CPC']))
        else:
            cpc_cv.append(np.nan)
            cpc_xls.append(np.nan)

    return np.array(cpc_cv), np.array(cpc_xls)

#-----------------------------------------------------------------------------------------
def retrieve_AQC_XLS(time, stations):
    
    """ Retrieves cross-validation CPC data for a set of stations from
    the xls files prepared by Yanni

    Parameters
    ----------

    time : datetime.datetime instance
        starting time of the time range
    stations : list of str
        list of weather stations at which to retrieve the CPC.CV data
    
    Returns
    -------
    A numpy array corresponding at the CPC.CV estimations at every specified 
    station
    """

    year = time.year

    folder = constants.FOLDER_CPCCV + str(year) + '/'
    
    files = sorted([f for f in glob.glob(folder + '*.xls') if '.s' not in f])
    
    def _start_time(fname):
        bname = os.path.basename(fname)
        times = bname.split('.')[1]
        tend = times.split('_')[1]
        return datetime.datetime.strptime(tend,'%Y%m%d%H00')

    tend = np.array([_start_time(f) for f in files])
    
    match = np.where(time < tend)[0]
    
    if not len(match):
        logging.warn('Could not find CPC CV file for time {:s}'.format(time))
        return np.zeros((len(stations))) + np.nan
    
    data = io.read_xls(files[match[0]])
    
    hour = int(datetime.datetime.strftime(time, '%Y%m%d%H00'))
    idx = np.where(np.array(data['time.stamp']) == hour)[0]
    data_hour = data.iloc[idx]
    data_hour_stations = data_hour.iloc[np.isin(np.array(data_hour['nat.abbr']), 
                                                stations)]
    aqc_xls = []
    for sta in stations:
        if sta in np.array(data_hour_stations['nat.abbr']):
            aqc_xls.append(float(data_hour_stations.loc[data_hour_stations['nat.abbr']
                    == sta]['AQC']))
        else:
            aqc_xls.append(np.nan)

    return np.array(aqc_xls)