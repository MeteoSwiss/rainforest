#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of functions to display QPE precipitation data and verification scores

"""
# Global imports
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

REFCOLORS = OrderedDict()
REFCOLORS['RZC'] = 'k'
REFCOLORS['CPC'] = 'dimgrey'
REFCOLORS['CPCH'] = 'slategrey'
REFCOLORS['CPC.CV'] = 'lightgray'

# Local imports
from . import constants
from .utils import nested_dict_values


def _autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:3.2f}'.format(height), rotation = 90 + 180 * (height < 0),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0,int(height < 0) * -27 + int(height > 0) * 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color = rect._facecolor)
        
        
        
class MidpointNormalize(Normalize):
    """
    Normalizing that is linear up to a certain transition value, logarithmic 
    afterwards
    """
    def __init__(self, vmin = None, vmax = None, transition = None, 
                 clip = False):
        
        self.transition = transition
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x = [self.vmin, self.transition]
        x.extend(np.linspace(self.transition + 1E-6, self.vmax,
                             30))
        y = [0, 0.25]
        y.extend(np.logspace(np.log10(0.25 + 1E-6), 0, 30))
        return np.ma.masked_array(np.interp(value, x, y))


class QPE_cmap( mpl.colors.LinearSegmentedColormap):
    """
    Colormap that uses purple tones for small values, and then a transition
    from blue to red for values above the transition
    """
    def __init__(self):
        colors = np.array([(255,255,255),
                  (122,1,119),
                  (43,66,181),
                  (67,222,139),
                  (245,245,45), 
                  (252,45,45)])/255 # This example uses the 8-bit RGB
        
        position = [0,0.25,0.251,0.5,0.75,1.0]
        cdict = {'red':[], 'green':[], 'blue':[]}
        for pos, color in zip(position, colors):
            cdict['red'].append((pos, color[0], color[0]))
            cdict['green'].append((pos, color[1], color[1]))
            cdict['blue'].append((pos, color[2], color[2]))
             
        mpl.colors.LinearSegmentedColormap.__init__(self, 'qpe', cdict, 256)

def qpe_plot(data, subplots = None, figsize = None,
             vmin = 0.04, vmax = 120, transition = 10, ch_border = True,
             xlim = None, ylim = None, cbar_orientation = 'horizontal',
             **kwargs):
    
    """Plots one or multiple QPE realizations using a special colormap, that
    shows a clear transition between low and high precipitation intensities,
    for low precipitation it is linear whereas for high precipitation it is
    logarithmic

    If multiple QPE realizations are given, they will be displayed as subplots
    
    Parameters
    ----------
    data : list of numpy arrays or numpy array
        the set of QPE realizations to display
        
    subplots: 2-element tuple (optional)
        Tuple indicating the number of subplots in each direction,
        the product of its elements must be equal to the number of QPE realizations
        If not provided, the default will be (1,n) where n is the number of 
        realizations
    
    figsize: 2-element tuple (optional)
        Tuple indicating the size of the figure in inches in both directions 
        (w,h)
        
    vmin : float (optional)
        Minimum value of precipitation to display, values below will be blank
        
    vmax : float (optional)
        Maximum value of precipitation to display, values below above will
        be shown with the color corresponding to vmax
    
    transition: float (optional)
        The transition value from which to change colormap and switch 
        from linear to logarithmic scale
    
    ch_border: bool (optiona)
        Whether to overlay the shapefile of the Swiss borders
    
    xlim: 2 element tuple (optional)
        limits of the plots in the west-east direction (in Swiss coordinates)
        
    ylim: 2 element tuple (optional)
        limits of the plots in the south-north direction (in Swiss coordinates)
    
    cbar_orientation : str (optional)
        colorbar orientation, either 'horizontal' or 'vertical'
        
    **kwargs:
        All additional arguments that can be passed to imshow

    Returns
    -------
    Nothing
    """
    
    cmap_qpe = QPE_cmap()
    cmap_qpe.set_under(color = 'w')
    
    if type(data) != list:
        data = [data]
        
    norm = MidpointNormalize(vmin, vmax, transition)
    
    n = len(data)
    if subplots == None:
        subplots = (1,n)
    
    if np.any(figsize == None):
        figsize = (4 * subplots[1], 3 * subplots[0]+1)
        
    fig,ax = plt.subplots(subplots[0], subplots[1], sharex = True,
                          sharey = True, figsize = figsize)
    
    if type(ax) == np.ndarray:
        ax = ax.ravel()
    else:
        ax = [ax]
    
    if len(ax) < n:
        raise ValueError('The total number of subplots is smaller than the number of QPE models to plot!')
        
    x = 0.5 * (constants.X_QPE[1:] + constants.X_QPE[0:-1])
    y = 0.5 * (constants.Y_QPE[1:] + constants.Y_QPE[0:-1])
    extent = [np.min(y),np.max(y),
              np.min(x),np.max(x)]

    for i,dd in enumerate(data):
        m = ax[i].imshow(dd,vmin = vmin, vmax = vmax, extent = extent,
              cmap = cmap_qpe,  norm = norm, **kwargs)
        
        if ch_border:
            for shape in constants.BORDER_SHP.shapeRecords():
                x = [i[0]/1000. for i in shape.shape.points[:]]
                y = [i[1]/1000. for i in shape.shape.points[:]]
                ax[i].plot(x,y,'k',linewidth=1.)
    if xlim != None:
        plt.xlim(xlim)
    else:
        plt.xlim([400,900])
    if ylim != None:
        plt.ylim(ylim)
    else:
        plt.ylim([0,350])
        
        
    
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    if cbar_orientation == 'horizontal':
        fig.subplots_adjust(bottom = 0.2)
        cbar_ax = fig.add_axes([0.18, 0.15, 0.7, 0.03])
        cbar=plt.colorbar(m,format='%.2f',orientation='horizontal', cax=cbar_ax,
                          norm = norm, extend = 'max')
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(),
                                rotation='vertical')
    else:
        fig.subplots_adjust(right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.2, 0.04, 0.6])
        cbar=plt.colorbar(m,format='%.2f',orientation='vertical', cax=cbar_ax,
                          norm = norm, extend = 'max')

        
    if vmax <= 50:
        ticks = np.array([vmin,5,10,15,20,25,30,35,40,45,50, 
                          transition, vmax])
    else:
        ticks = np.array([vmin,5,10,20,30,40,50,60,70,80,90,100,110,120, 
                          transition, vmax])
        
    ticks = ticks[ticks < vmax]
    cbar.set_ticks(ticks)
    cbar.set_label("Rainfall intensity [mm/hr]")
    return fig, ax
        
def score_plot(scores, title_prefix = '', figsize = (10,5)):
    """Plots a series of QPE verification scores in the form of stacked
    barplots, for different ranges of precipitation
    
    IMPORTANT: the scores dictionary must have the following structure
    
    scores[model][precip_range][score]
    
    for example
    
    scores['RF_dualpol']['0.0-2.0']['ME'] = -0.27
    
    you can get such a dictionary with the perfscores function in common.utils
    i.e. scores['RZC'] = perfscores(...)
    
    Parameters
    ----------
    scores : dict of dict of dict of scores
        the set of scores to display
        
    title_prefix: str (optional)
        a prefix for the suptitle (global title)
    
    figsize: 2-element tuple (optional)
        Tuple indicating the size of the figure in inches in both directions 
        (w,h)
        
    Returns
    -------
    Nothing
    """
    

    models = list(scores.keys())
    
    models_reordered = []
    colors = []
    for m in REFCOLORS.keys():
        if m in models:
            models_reordered.append(m)
            colors.append(REFCOLORS[m])
    i = 0
    for m in models:
        if m not in models_reordered:
            models_reordered.append(m)
            colors.append('C'+str(i))
            i+=1
        
    precip_ranges = list(scores[models[0]].keys())
    scorenames = list(scores[models[0]][precip_ranges[0]].keys())
    scorenames.remove('N')
    fig, ax = plt.subplots(2, int(np.ceil(len(precip_ranges)/2)),
                           figsize = figsize)
    ax = ax.ravel()
    for i, precip_range in enumerate(precip_ranges):
        x = []
        labels = []
        offset  = len(models) + 1
        for j,s in enumerate(scorenames):
            for k, m in enumerate(models_reordered):
                sc = scores[m][precip_range][s]
                rec = ax[i].bar([offset*j+k],[sc], color = colors[k], 
                                width = 1)
                _autolabel(ax[i], rec)
               
                
            labels.append(m)
            x.append(offset*j+0.5)      

        ax[i].set_xticks(x)
        ax[i].set_xticklabels(scorenames, rotation=65)
        ax[i].set_title('precip_range = {:s}, N = {:d} samples'.format(precip_range,scores[m][precip_range]['N']))
        fig.legend(models_reordered,loc="center right",   # Position of legend
           borderaxespad=0.1)
    plt.subplots_adjust(right=0.85)
    plt.suptitle(title_prefix )
    # remove extra subplots
    for i in range(len(precip_ranges), len(ax)):
        ax[i].set_visible(False)
    fig.subplots_adjust(hspace=0.3)
    
def qpe_scatterplot( qpe_est, ref, title_prefix = '', figsize = (10,7.5)):
    """Plots the results of multiple QPE models as a function of the
    reference gauge measurement
    
    
    Parameters
    ----------
  
    qpe_est : dict of arrays
        Every value in the dictionary is a set of QPE estimates, every key
        is a model

    ref: np.ndarray
        contains the reference observations (gauge), must have the same shape
        as any element in qpe_est
        
        
    title_prefix: str (optional)
        a prefix for the suptitle (global titl    
  
    figsize: 2-element tuple (optional)
        Tuple indicating the size of the figure in inches in both directions 
        (w,h)
        
    Returns
    -------
    Nothing
    """
    

    models = list(qpe_est.keys())
    
    models_reordered = []
    for m in REFCOLORS.keys():
        if m in models:
            models_reordered.append(m)
    for m in models:
        if m not in models_reordered:
            models_reordered.append(m) 
        
    if len(models)  > 3:
        fig, ax = plt.subplots(2, int(np.ceil(len(models)/2)),
                               figsize = figsize, sharey=True, sharex = True)
    else:
        fig, ax = plt.subplots(1, len(models),
                               figsize = figsize,sharey=True, sharex = True)
        
    if type(ax) == np.ndarray :
        ax = ax.ravel()
    elif type(ax) != list:
        ax = [ax]
   
    plt.setp(ax, aspect=1.0, adjustable='box')
    
    gmax = np.nanmax(ref)
    for i,m in enumerate(models_reordered):
        pl = ax[i].hexbin(ref.ravel(), qpe_est[m].ravel(), bins = 'log',
                       mincnt = 1, vmax = len(ref.ravel())/100, vmin = 1)
        ax[i].plot([0,gmax],[0,gmax],'r')
        ax[i].grid()
        ax[i].set_title(m)
        ax[i].set_xlabel(r'Observation $R$ [mm]')
        ax[i].set_ylabel(r'Prediction $R$ [mm]')
        
        plt.xlim([0,gmax])
        plt.ylim([0,gmax])

    plt.suptitle(title_prefix )
    # remove extra subplots
    for i in range(len(models), len(ax)):
        ax[i].set_visible(False)
        
    fig.subplots_adjust(bottom = 0.3)
    fig.subplots_adjust(hspace=0.25)
    cax = fig.add_axes([0.18, 0.15, 0.7, 0.03])
    fig.colorbar(pl, cax, orientation = 'horizontal', label = 'Counts')
    
    
def plot_crossval_stats(stats, output_folder):
    """
    Plots the results of a crossvalidation intercomparion as performed in
    the rf.py module 
    
    Parameters
    ----------
    stats : dict
        dictionary containing the result statistics as obtained in the 
        rf.py:model_intercomparison function
    output_folder : str
        where to store the plots
    
    """  
    
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
    data = np.reshape(list(nested_dict_values(stats)), all_dims)
    
    # Flip method/bound axis
    data = np.swapaxes(data, 1,4)
    all_keys[1], all_keys[4] = all_keys[4], all_keys[1] 
    all_dims[1], all_dims[4] = all_dims[4], all_dims[1] 

    
    # get nested dict keys
    aggtype = list(stats.keys())
    qpetype = list(stats[aggtype[0]].keys())
    veriftype = list(stats[aggtype[0]][qpetype[0]].keys())
    preciptype = list(stats[aggtype[0]][qpetype[0]][veriftype[0]].keys())
    boundtype = {}
    boundtype['10min'] = list(stats['10min'][qpetype[0]][veriftype[0]][preciptype[0]].keys())
    boundtype['60min'] = list(stats['60min'][qpetype[0]][veriftype[0]][preciptype[0]].keys())
    scoretype = list(stats[aggtype[0]][qpetype[0]][veriftype[0]][preciptype[0]][boundtype['10min'][0]].keys())
    
    models_reordered = []
    for m in REFCOLORS.keys():
        if m in qpetype:
            models_reordered.append(m)
    for m in qpetype:
        if m not in models_reordered:
            models_reordered.append(m) 
    
    qpetype = models_reordered
    scoretype.remove('N')
    
    colors = []
    idx = 0
    for i, q in enumerate(qpetype):
        if q in REFCOLORS.keys():
            c = REFCOLORS[q]
        else:
            c = 'C'+str(idx)
            idx += 1 
        colors.append(c)


    for a in aggtype:
        for b in boundtype[a]:
            for v in veriftype:
                 fig, ax = plt.subplots(len(preciptype),1, figsize = (7,12))
                 for i, p in enumerate(preciptype):
                     offset  = len(qpetype) + 1
                     x = []
                     for j,s in enumerate(scoretype):
                         for k, q in enumerate(qpetype):
                             mean = stats[a][q][v][p][b][s]['mean']
                             std = stats[a][q][v][p][b][s]['std']
                 
                             rec = ax[i].bar([offset*j+k],[mean], 
                                             color = colors[k],
                                             yerr = std, width = 1.)
                             _autolabel(ax[i], rec)
                         x.append(offset*j+0.6)      

                     ax[i].set_xticks(x)
                     ax[i].set_xticklabels(scoretype, rotation=65)
                     ax[i].set_ylabel('precip: {:s}'.format(p))
                  
                 fig.legend(qpetype)
                 plt.suptitle('{:s} errors, Agg : {:s}, R-range {:s} \n Nsamples = {:d}'.format(v,
                          a, b, 
                          int( stats[a][q][v][p][b]['N']['mean'])))
                
                 nfile = '{:s}_{:s}_{:s}'.format(v, a, b) + '.png'
                 plt.savefig(output_folder + '/' + nfile, dpi = 300, 
                            bbox_inches = 'tight')
            
