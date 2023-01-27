#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions that can be useful

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019
"""

# Global imports
import datetime
import io
import os
from collections import OrderedDict
import numpy as np
from scipy.stats import energy_distance
from dateutil import parser
import glob
import yaml
import  dask.dataframe as dd
import re

# Local imports
from .logger import logger
from .wgs84_ch1903 import GPSConverter
from . import constants

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def envyaml(filename):
    """
    Reads a yaml configuration file while parsing environment variables.
    Environment variables must be written as ${NAME_OF_VAR} in the yaml file
    
    Parameters
    ----------
    filename : str
        path of the input yaml file

    Returns
    -------
    dict
        the yaml content in the form of a python dict

    """
    pattern = "(\\$\\{[A-Za-z0-9]+\\})"
    file = open(filename,'r')
    filebuff = io.StringIO()
    for l in file.readlines():
        matches = re.findall(pattern, l)
        for m in matches:
            l = l.replace(m, os.environ[m.replace('${','').replace('}','')])
        filebuff.write(l)
    filebuff.seek(0)
    
    return yaml.load(filebuff, Loader = yaml.FullLoader)

def perfscores(est_data, ref_data, bounds = None, array = False):
    """
    Computes a set of precipitation performance scores, on different data ranges.
    The scores are
        - scatter: 0.5 * (Qw84(x) - Qw16(x)), where Qw is a quantile weighted
          by ref_data / sum(ref_data) and x is est_data / ref_data in dB scale
        - RMSE: root mean  square error (linear error)
        - bias:  (ME/mean(ref_data) + 1) in dB
        - ED: the energy distance which is a measure of the distance between
          two distributions (https://en.wikipedia.org/wiki/Energy_distance)
          
    Parameters
    ----------
    est_data : ndarray
        array of estimates (ex. precip from QPE)
    ref_data : ndarray
        array of reference (ex. precip from gauge)
    bounds : list (optional)
        list of bounds on ref_data for which to compute the error metrics,
        by default all data will be used (unbounded), note that even if you 
        prescribe bounds the scores for the overall data will always be 
        added in the output
    array: boolean (optional)
        Whether or not to convert the output dict to a numpy array
        
    Returns
    -------
    all_metrics : dict or ndarray
        a dictionary containing all the scores, organized in the following way
        all_metrics[bound][score] 
    """
    all_metrics = OrderedDict()
    
    valid = np.logical_and(est_data >= 0, ref_data >= 0)
    est_data = est_data[valid > 0]
    ref_data = ref_data[valid > 0]
    
    est = est_data
    ref = ref_data       
    
    
    all_metrics['all'] = _perfscores(est, ref)
    
    if bounds != None:
        for i in range(len(bounds) -1):
            bound_str = '{:2.1f}-{:2.1f}'.format(bounds[i],bounds[i+1])
            cond = np.logical_and(ref_data < bounds[i+1],
                                  ref_data >= bounds[i])
            if np.sum(cond) > 0:
                est = est_data[cond]
                ref = ref_data[cond]
                
                all_metrics[bound_str] = _perfscores(est, ref)
            
    if array == True:
        arr = []
        for k in all_metrics:
            arr.append(list(all_metrics[k].values()))
        arr = np.array(arr)
        all_metrics = np.array(arr)
        
    return all_metrics

def _perfscores(est_data, ref_data, doublecond_thresh=0.1):
    """An unbounded version of the previous function"""
    doublecond = np.logical_and(ref_data > doublecond_thresh, 
                                est_data > doublecond_thresh)
    rmse = np.sqrt(np.nanmean((est_data[doublecond]-ref_data[doublecond])**2))
    db_err = 10 * np.log10(est_data[doublecond] / ref_data[doublecond])
    weights = ref_data[doublecond]/np.sum(ref_data[doublecond])
    scatter = 0.5 * (quantile(db_err,weights,0.84) -quantile(db_err,weights,0.16))
    bias_db = 10*np.log10(np.sum(est_data[doublecond]) / np.sum(ref_data[doublecond]))
    ed = energy_distance(est_data[np.isfinite(est_data)], ref_data[np.isfinite(est_data)])
    
    mest = np.nanmean(est_data[doublecond])
    mref = np.nanmean(ref_data[doublecond])
    stdest = np.nanstd(est_data[doublecond])
    stdref = np.nanstd(ref_data[doublecond])

    metrics = {'RMSE':rmse,
            'scatter':scatter,
            'logBias':bias_db,
            'ED':ed,
            'N':len(ref_data[doublecond]),
            'N_all':len(ref_data),
            'est_mean':mest,
            'ref_mean':mref,
            'est_std':stdest,
            'ref_std':stdref}
    
    return metrics

def split_by_time(files_rad):
    """Separate a list of files by their timestamp"""
    out = {}
    if type(files_rad) == dict:
        for k in files_rad.keys():
            out[k] = _split_by_time(files_rad[k])
    else:
        out = _split_by_time(files_rad)
    return out    
    
def _split_by_time(files_rad):
    out = {}
    
    for f in files_rad:

        t = timefromfilename(f)
        if t in out.keys():
            out[t].append(f)
        else:
            out[t] = [f]
    # Avoid lists with size 1
    for k in out.keys():
        if len(out[k]) == 1:
            out[k] = out[k][0]
    return out


def timestamp_from_datetime(dt):
    return  dt.replace(tzinfo = datetime.timezone.utc).timestamp()

def timestamp_from_datestr(datestr):
    # Datetstr in YYYYmmdd or YYYYmmddHH or YYYYmmddHHMM format
    datestr = parser.parse(datestr)
    epoch = datetime.datetime(1970,1,1)

    return int((datestr - epoch).total_seconds())

def timefromfilename(fname):
    """Returns the datetime of a file based on its name"""
    bname = os.path.basename(fname)
    tstr = bname[3:12]
    time = datetime.datetime.strptime(tstr,'%y%j%H%M')
    # If a forecast hour is taken from a future run:
    if (bname.startswith('HZT')) & (bname[-2::] != '00'):
        time = time + datetime.timedelta(hours=int(bname[-2::]))
    return time

def sweepnumber_fromfile(fname):
    """Returns the sweep number of a polar file based on its name"""
    return int(os.path.basename(fname).split('.')[1])

def round_to_hour(dt):
    """Returns the sweep number of a polar file based on its name"""
    round_delta = 60 * 30
    round_timestamp = dt.timestamp() + round_delta
    round_dt = datetime.datetime.fromtimestamp(round_timestamp)
    return round_dt.replace(microsecond=0, second=0, minute=0)

def idx_cart(x,y):
    """Returns the Cartesian index of a set of coordinates x and y"""
    '''
    Returns the Cartesian index of a set of coordinates x and y

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    if type(x) != np.ndarray:
        x = np.array([x])
    if type(y) != np.ndarray:
        y = np.array([y])  
        
    x_qpe = constants.X_QPE
    y_qpe = constants.Y_QPE

    # For x the columns in the Cartesian lookup tables are lower bounds
    # e.g. x = 563, means that radar pixels are between 563 and 564
    y_llc = (y/constants.CART_GRID_SIZE).astype(int)
    # For y the columns in the Cartesian lookup tables are upper bounds
    # e.g. x = 182, means that radar pixels are between 181 and 182            
    x_llc = (np.ceil(x/constants.CART_GRID_SIZE)).astype(int)
    
    idx =  [(np.max(x_qpe) - x_llc).astype(int),
            (y_llc - np.min(y_qpe)).astype(int)]

    return np.squeeze(idx)     
    
        
def stack_uneven(arrays, fill_value = np.nan):
    '''
    Fits mmltiple into a single numpy array, even if they are
    different sizes, assigning a fill_value to parts with no data

    Parameters
    ----------
    arrays: list of np arrays 
        list of numpy array to stack, they can have different dimensions
        
    fill_value: (float, optional)
        the fill value with which to fill the missing pixels

    Returns
    -------
        a np.ndarray with size N x M, where N is the sum of the number of 
        rows of all arrays and M is the maximal number of columns in all arrays
    '''
        
    dim0 = [a.shape[0] for a in arrays]
    dim1 = [a.shape[1] for a in arrays]
    
    dim2max = max(dim1)
    
    stacked = np.ones((np.sum(dim0), dim2max)) + fill_value
    
    idx_row = 0
    for arr in arrays:
        stacked[idx_row:idx_row + arr.shape[0], 0:arr.shape[1]] = arr
        idx_row += arr.shape[0]
        
    return stacked


def quantile_1D(data, weights, quantile):
    """
    Compute the weighted quantile of a 1D numpy array.

    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile_1D : float
        The output value.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if ((quantile > 1.) or (quantile < 0.)):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    #assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn-0.5*sorted_weights)/np.sum(sorted_weights)
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)


def quantile(data, weights, quantile):
    """
    Weighted quantile of an array with respect to the last axis.

    Parameters
    ----------
    data : ndarray
        Input array.
    weights : ndarray
        Array with the weights. It must have the same size of the last 
        axis of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.

    Returns
    -------
    quantile : float
        The output value.
    """
    # TODO: Allow to specify the axis
    nd = data.ndim
    if nd == 0:
        TypeError("data must have at least one dimension")
    elif nd == 1:
        return quantile_1D(data, weights, quantile)
    elif nd > 1:
        n = data.shape
        imr = data.reshape((np.prod(n[:-1]), n[-1]))
        result = np.apply_along_axis(quantile_1D, -1, imr, weights, quantile)
        return result.reshape(n[:-1])
    
    

def wgs84toCH1903(lat, lon, heights):
    """
    Converts a set of WGS84, lat/lon/heights to Swiss CH1903 coordinates,
    east, north and height

    Parameters
    ----------
    lat : ndarray
        latitudes in decimal format (degrees)
    lon : ndarray
        longitudes in decimal format (degrees)
    heights : ndarray
        heights a.s.l in WGS84 coordinates

    Returns
    -------
    east, north and height coordinates in CHLV190
    """
    
    conv = GPSConverter()
    lv03 = conv.WGS84toLV03(lat, lon, heights)
    return lv03[0], lv03[1], lv03[2]

def LV03toWGS84(east, north, heights):
    """
    Converts a set of WGS84, lat/lon/heights to Swiss CH1903 coordinates

    Parameters
    ----------
    east : ndarray
        Easterly Swiss coordinates (CHY)
    north : ndarray
        northerly Swiss coordinates (CHX)
    heights : ndarray
        heights a.s.l in WGS84 coordinates

    Returns
    -------
    lat, lon and height coordinates in WGS84
    """
    
    conv = GPSConverter()
    wgs = conv.LV03toWGS84(east, north, heights)
    return wgs[0], wgs[1], wgs[2]


def chunks(l, n):
    """
    Divides a list l into n sublists of similar sizes
    """
    o = int(np.round(len(l)/n))
    out = []
    # For item i in a range that is a length of l,
    for i in range(0, n):
        # Create an index range for l of n items:
        if i == n-1:
            sub = l[i*o:]
        else:
            sub = l[i*o:i*o+o]
        
        if len(sub):
            out.append(sub)
    return out


def dict_flatten(mydict):
    """
    Flattens a nested dictionary
    """
    new_dict = {}
    for key,value in mydict.items():
        if type(value) == dict:
            _dict = {':'.join([key,str(_key)]):_value for _key, _value in
                     dict_flatten(value).items()}
            new_dict.update(_dict)
        else:
            new_dict[key]=value
    return new_dict
      
def nested_dict_values(d):
    """
    Extracts all values from a nested dictionary
    """
    listvals = list(nested_dict_gen(d))
    listvals_unwrapped = []
    for l in listvals:
        if type(l) == list or type(l) == np.ndarray:
            for ll in l:
                listvals_unwrapped.append(ll)
        else:
            listvals_unwrapped.append(l)
    return listvals_unwrapped


def nested_dict_gen(d):
    """
    The generator for the previous function
    """
    for v in d.values():
        if isinstance(v, dict):
            yield from nested_dict_gen(v)
        else:
            yield v
      
def nanadd_at(a, indices, b):
    """ Replaces nans by zero in call to np.add.at """
    mask = np.isfinite(b)
    b = b[mask]
    indices = indices[mask]
    indices = tuple([indices[:,0], indices[:,1]])
    return np.add.at(a, indices, b)
    
def aggregate_multi(array_3d, agg_operators):
    """
    Aggregates a 3D numpy array alongs its first axis, using different
    aggregation operators
    
    Parameters
    ----------
    array_3d : ndarray
        3D numpy array, of shape (N x M x L)
    agg_operators : ndarray of integers
        array of aggregation operators as defined in the constants.py file,
        must have the same length as the first dimension of array_3D (N)
           
    Returns
    -------
    An aggregated array of size M x L
    """
    out = np.zeros(array_3d[0].shape) + np.nan
    op_un, idx = np.unique(agg_operators, return_inverse = True)
    for i, op in enumerate(op_un):
        out[:,idx == i] = constants.AVG_METHODS[op](array_3d[:,:,idx == i],
           axis = 0)
    
    return out



def rename_fields(data):
    """
    Rename pyart fields from pyrad names to simpler names, according to the
    dictionary PYART_NAMES_MAPPING in the constants.py module
    """
    old_keys =  list(data.fields.keys())
    for k in old_keys:
        if k in constants.PYART_NAMES_MAPPING.keys():
            new_name = constants.PYART_NAMES_MAPPING[k]
            data.fields[new_name] = data.fields.pop(k)
            
def read_task_file(task_file):    
    """
    Reads a database processing task file
    """
    tasks_dic = OrderedDict() # We want a sorted dict
    
    with open(task_file,'r') as f:
        for line in f:
            line = line.strip('\n').split(',')
            line = np.array([s.replace(' ','') for s in line])
            tasks_dic[int(line[0])] = line[1:]
    return tasks_dic

def read_df(pattern, dbsystem = 'dask', sqlContext = None):
    """
    Reads a set of data contained in a folder as a spark or dask DataFrame
    
    Parameters
    ----------
    pattern : str
        Unix style wildcard pattern pointing to the files, for example
        /store/msrad/folder/*.csv will read all csv files in that folder
    dbsystem : str
        Either "dask" if you want a Dask DataFrame or "spark" if you want a 
        spark dataframe
    sqlContext : sqlContext instance
        sqlContext to use, required only if dbystem = 'spark'
        
    Returns
    -------
    A spark or dask DataFrame instance
    """
    
    if dbsystem not in ['spark','dask']:
        raise NotImplementedError('Only dbsystem = "spark" or "dask" are supported!')
    if dbsystem == 'spark' and sqlContext == None:
        raise ValueError('sqlContext must be provided if dbystem = "spark"!')
        
    files = glob.glob(pattern)
    df = None
    if '.parq' in files[0] or '.parquet' in files[0]:
        # For some reason wildcards are not accepted with parquet
        if dbsystem == 'spark':
            df = sqlContext.read.parquet(*files)
        else:
            df = dd.read_parquet(pattern) 
    elif '.csv' in files[0]:
        if dbsystem == 'spark':
            df = sqlContext.read.csv(pattern,
                           header = True, inferSchema = True)
        else:
            if '.gz' in files[0]:
                df = dd.read_csv(pattern, compression  = 'gzip', dtype={'TIMESTAMP':'float64'} )
            else:
                df = dd.read_csv(pattern)
    else:
        logger.error("""Invalid data, only csv and parquet files are accepted.
        Make sure that they have a valid suffix (.csv, .csv.gz, .parquet,
        .parq)""")

    return df


def nearest_time(dt, reference):
    """
    Gets the nearest earlier reference timestep to a given datetime, for ex.
    if dt = 1 Jan 2020 10:12, and reference is 10 it will return
     1 Jan 2020 10:10, or if dt = 1 Jan 2020 10:59 and reference is 60
     it will return 1 Jan 2020 10:00
    
    Parameters
    ----------
    dt : datetime
        The datetime to check
    reference : int
        The reference timestep in minutes
  
    Returns
    -------
    The closest earlier timestep in datetime format
    
    """
    dt2 = dt - datetime.timedelta(minutes =  dt.minute % reference,
             seconds = dt.second,
             microseconds = dt.microsecond) 
    if dt2 != dt:
       dt2 += datetime.timedelta(minutes = reference)
        
    return dt2

def get_qpe_files(input_folder, t0 = None, t1 = None, time_agg = None,
                  list_models = None):
    """
    Gets the list of all qpe files in a folder (as saved by qpe_compute)
    and separates them by qpe type and timestep
    
    Parameters
    ----------
    input_folder : str
        main directory where the qpe files are saved, it contains one subfolder
        for every qpe model (type) that was used
    t0 : datetime (optional)
        Starting time of the period to retrieve, will be used to filter files,
        if not provided starting time will be time of first file
    t1 : datetime (optional)
        End time of the period to retrieve, will be used to filter files,
        if not provided end time will be time of last file
    time_agg : minutes (optional)
        Will aggregate all files to a reference time in minutes (e.g. use 10 to
        put together all files that correspond to a single gauge measurement)
    list_models: (optional)
        List of qpe types to retrieve , if not provided all folders in input_folder
        will be used
    Returns
    -------
    A dictionary where every key is a QPE model and every value is a list
    with all files in chronological order
    """
    all_files = {}
    
    for sub in glob.glob(input_folder + '/*'):
        model = os.path.basename(sub)

        if list_models != None:
            if model not in list_models:
                continue
        
        files = glob.glob(sub + '/*')
        for f in files:
            try:
                t = str(re.match('.*[a-zA-Z]([0-9]{9}).*',f)[1])
                t = datetime.datetime.strptime(t,'%y%j%H%M')
                if time_agg != None:
                    t = nearest_time(t, time_agg)
                    
                if t0 != None:
                    if t < t0:
                        continue
                if t1 != None:
                    if t > t1:
                        continue
                    
                if t not in all_files.keys():
                    all_files[t] = {}
                if model not in all_files[t].keys():
                    all_files[t][model] = []
                
                all_files[t][model].append(f)   
            except:
                pass
            
    return all_files

def get_qpe_files_multiple_dirs(input_folder, t0 = None, t1 = None, time_agg = None,
                                list_models = None):
    """
    Gets the list of all qpe files from multiple folders sorted according to DOY (as saved by qpe_compute)
    and separates them by qpe type and timestep
    
    Parameters
    ----------
    input_folder : str
        main directory where the qpe files are saved, it contains one subfolder
        for every qpe model (type) that was used
    t0 : datetime (optional)
        Starting time of the period to retrieve, will be used to filter files,
        if not provided starting time will be time of first file
    t1 : datetime (optional)
        End time of the period to retrieve, will be used to filter files,
        if not provided end time will be time of last file
    time_agg : minutes (optional)
        Will aggregate all files to a reference time in minutes (e.g. use 10 to
        put together all files that correspond to a single gauge measurement)
    list_models: (optional)
        List of qpe types to retrieve , if not provided all folders in the first input_folder
        will be used
    Returns
    -------
    A dictionary where every key is a QPE model and every value is a list
    with all files in chronological order
    """
    
    # If there is only one input folder, use function above
    if (type(input_folder) != list):
        logger.info('Only one directory is given, switch to get_qpe_files() routine')
        all_files = get_qpe_files(input_folder, t0, t1, time_agg,list_models)
        return all_files
    
    # Create a model list
    if (list_models == None):
        list_models = []
        for sub in glob.glob(input_folder[0] + '/*'):
            list_models.append(os.path.basename(sub))
    
    # Create file lists for each model
    file_list = {}
    for model in list_models:
        files = []
        for infolder in input_folder:
            dir = os.path.join(infolder+model)
            # Check whether directory exists
            if not os.path.isdir(dir):
                continue
            # Append all files
            files += glob.glob(dir+'/*')
        file_list[model] = files
    
    # Get the filepaths and names according to model and timestep
    all_files = {}
    for model in file_list.keys():
        for f in file_list[model]:
            try:
                t = str(re.match('.*[a-zA-Z]([0-9]{9}).*',f)[1])
                t = datetime.datetime.strptime(t,'%y%j%H%M')
                if time_agg != None:
                    t = nearest_time(t, time_agg)
                    
                if t0 != None:
                    if t < t0:
                        continue
                if t1 != None:
                    if t > t1:
                        continue
                    
                if t not in all_files.keys():
                    all_files[t] = {}
                if model not in all_files[t].keys():
                    all_files[t][model] = []
                
                all_files[t][model].append(f)   
            except:
                pass
            
    return all_files