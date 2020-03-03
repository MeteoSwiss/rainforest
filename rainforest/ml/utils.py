#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the ML submodule
"""

# Global imports
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

# Local imports
from ..common.graphics import REFCOLORS
from ..common.utils import autolabel

def vert_aggregation(radar_data, vert_weights, grp_vertical, 
                  visib_weight = True, visib = None):
    """
    Performs vertical aggregation of radar observations aloft to the ground
    using a weighted average. Categorical variables such as 'RADAR',
    'HYDRO', 'TCOUNT', will be assigned dummy variables and these dummy
    variables will be aggregated, resulting in columns such as RADAR_propA
    giving the weighted proportion of radar observation aloft that were
    obtained with the Albis radar
    
    Parameters
    ----------
    radar_data : Pandas DataFrame
        A Pandas DataFrame containing all required input features aloft as
        explained in the rf.py module 
    vert_weights : np.array of float
        vertical weights to use for every observation in radar, must have
        the same len as radar_data
    grp_vertical : np.array of int
        grouping index for the vertical aggregation. It must have the same
        len as radar_data. All observations corresponding to the same
        timestep must have the same label
    visib_weight: bool
        if True the input features will be weighted by the visibility
        when doing the vertical aggregation to the ground
    visib : np array
        visibily of every observation, required only if visib_weight = True
    """    
    if visib_weight and not np.any(visib == None):
        vert_weights = vert_weights * visib / 100.
    else:
        vert_weights = pd.DataFrame(vert_weights)
            
    X =  pd.DataFrame()  # output
    sum_wvisib = vert_weights.groupby(grp_vertical).sum()

    for v in radar_data.columns:
        if v not in ['RADAR','HYDRO','TCOUNT']:
            X[v] = (radar_data[v] * vert_weights).groupby(grp_vertical).sum() / sum_wvisib
        else:
            # For these variables we aggregate dummy variables
            vals = np.unique(radar_data[v])
            for val in vals:
                X[v+'_prop_'+str(val)] = (((radar_data[v] == val).astype(int) * vert_weights).
                        groupby(grp_vertical).sum() / sum_wvisib)
    return X

def nesteddictvalues(d):
  for v in d.values():
    if isinstance(v, dict):
      yield from nesteddictvalues(v)
    else:
      yield v
      
def chunks(l, n):
    '''Cuts list l into maximum n chunks of similar sizes'''
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


def split_event(timestamps, n = 5, threshold_hr = 12):
    """
    Splits the dataset into n subsets by separating the observations into
    separate precipitation events and attributing these events randomly
    to the subsets
    
    Parameters
    ----------
    timestamps : int array
        array containing the UNIX timestamps of the precipitation observations
    n : int
        number of subsets to create
    threshold_hr : int
        threshold in hours to distinguish precip events. Two timestamps are
        considered to belong to a different event if there is a least 
        threshold_hr hours of no observations (no rain) between them.
    
    Returns
    ---------
    split_idx : int array
        array containing the subset grouping, with values from 0 to n - 1
    """  
    logging.info('Splitting dataset in {:d} parts using different events'.format(n))

    tstamps_gau = np.array(timestamps - timestamps%  3600)
    order = np.argsort(tstamps_gau)
    revorder = np.argsort(order)
    
    tstamp = tstamps_gau[order]
    hours_elapsed = (tstamp - tstamp[0]) / 3600
    dif = np.diff(hours_elapsed)
    dif = np.insert(dif,0,0)   
    
    # label the events
    jumps = np.zeros((len(dif)))
    jumps[dif > threshold_hr] = 1
    labels = np.cumsum(jumps)

    maxlabel = labels[-1]
    allevents = np.arange(maxlabel)
    np.random.shuffle(allevents) # randomize
    
    # split events in n groups
    events_split = chunks(allevents, n)
    
    split_idx = np.zeros((len(timestamps)))
    
    for i in range(n):
        split_idx[np.isin(labels, events_split[i])] = i
    split_idx = split_idx[revorder]
    
    return split_idx

def plot_crossval_stats(stats, output_folder):
    """
    Plots the results of a crossvalidation intercomparion as performed in
    the rf.py module
    
    Parameters
    ----------
    stats : dict
        dictionary containing the result statistics
    output_folder : str
        where to store the plots
    
    """  
    
    width = 0.9
    # Convert dict to array    
    success = True
    all_keys = []
    all_dims = []
    cdict = stats
    while success:
        try:
            keys = list(cdict.keys())
            all_keys.append(keys)
            all_dims.append(len(keys))
            cdict = cdict[keys[0]]
            
        except:
            success = False
            pass
    
    # convert to array
    data = np.reshape(list(nesteddictvalues(stats)), all_dims)
    
    # Flip method/bound axis
    data = np.swapaxes(data, 1,4)
    all_keys[1], all_keys[4] = all_keys[4], all_keys[1] 
    all_dims[1], all_dims[4] = all_dims[4], all_dims[1] 
    
    for i, agg in enumerate(all_keys[0]):
        for j, bound in enumerate(all_keys[1]):
            for k, veriftype in enumerate(all_keys[2]):
                fig, ax = plt.subplots(all_dims[3],1, figsize = (7,12))
                n = all_dims[4]
                for l, precipttype in enumerate(all_keys[3]):
                    dataplot = data[i,j,k,l]
                    x = np.arange(len(dataplot[0]))
                    
                    idx = 0
                    for m,d in enumerate(dataplot):
                        
                        name = all_keys[-3][m]
                        if name in REFCOLORS.keys():
                            c = REFCOLORS[name]
                        else:
                            c = 'C'+str(idx)
                            idx += 1 
                        rec = ax[l].bar(x + (m-int(n/2))*width/n, d[:,0],
                                    width = width/n,
                                    yerr = d[:,1], color = c)
                        
                        autolabel(ax[l],rec)
 
                    ax[l].set_xticklabels(all_keys[-2])
                    ax[l].set_xticks(x)
                    fig.legend(all_keys[-3])
                    ax[l].set_ylabel('precip: {:s}'.format(precipttype))
                plt.suptitle('{:s} errors, Agg : {:s}, R-range {:s}'.format(veriftype,
                          agg, bound))
                nfile = '{:s}_{:s}_{:s}'.format(veriftype, agg, bound) + '.png'
                plt.savefig(output_folder + '/' + nfile, dpi = 300, 
                            bbox_inches = 'tight')
