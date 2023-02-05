#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set of constants regarding MeteoSwiss radars and QPE

"""

import os
import pandas as pd
import numpy as np
import shapefile
from pathlib import Path
import glob
import datetime
from scipy.stats import mode

import socket

from .object_storage import ObjectStorage
ObjStorage = ObjectStorage()

###############
# DATA
###############

data_folder = os.environ['RAINFOREST_DATAPATH']
metadata_folder = str(Path(data_folder, 'references', 'metadata'))
lut_folder = str(Path(data_folder, 'references', 'lookup_data'))
lut_boscacci_folder = str(Path(data_folder, 'references', 'lut_boscacci'))
cosmo_coords_folder = str(Path(data_folder, 'references', 'coords_cosmo'))
radar_samples_folder = str(Path(data_folder, 'references', 'radar_samples')) 
rfmodels_folder = str(Path(data_folder, 'rf_models'))


###############
# PHYSIC
###############

KE = 1.25 # value of KE used by MeteoSwiss
LAPSE_RATE = 0.7 # avg lapse rate in the atmosphere
THRESHOLD_SOLID = 2 # Below 2 °C is considerd to be solid precipitation

###############
# STATIONS
###############

METSTATIONS = pd.read_csv(ObjStorage.check_file(str(Path(metadata_folder, 'data_stations.csv'))), 
                           sep=';', encoding='latin-1')

###############HE
# RADARS
###############


RADARS = pd.read_csv(ObjStorage.check_file(str(Path(metadata_folder, 'data_radars.csv'))),
                           sep=';', encoding='latin-1')

ELEVATIONS = [-0.2, 0.4, 1.0, 1.6, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 
               9.5, 11.0, 13.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0]

RADIAL_RESOLUTION = {'L':500, 'H': 83.3}

NGATES = {}
NGATES['H'] = [2952,
 2520,
 2952,
 1944,
 2196,
 1944,
 1752,
 1944,
 1680,
 1452,
 1332,
 1200,
 1044,
 900,
 744,
 600,
 492,
 408,
 360,
 324]

NGATES['L'] = [int(n/6) for n in NGATES['H']]

CART_GRID_SIZE = 1000

NSAMPLES = [33.3, 38.9,33.3,37.5,33.3,37.5,27.8,37.5,27.8,33.3,33.3,
            33.3,41.7,41.7,41.7,41.7,41.7,41.7,41.7,41.7 ]
NYQUIST_VEL = [8.3,9.6,8.3,12.4,11,12.4,13.8,12.4,13.8,16.5,16.5,16.5,
               20.6,20.6,20.6,20.6,20.6,20.6,20.6,20.6]


###############
# QPE
###############

# Quality checks to avoid the inclusion of erroneous data
STATUS_QC_FLAG_TH = 0.01 #dBADU
STATUS_QC_FLAG_SWEEP = 19
STATUS_QC_REMOVE_TH = 13

VPR_REF_HEIGHTS = {'A':1500,'D':2000,'L':2000,'P':1500,'W':1500}
VPR_REF_RADAR = {'A':'A','P':'A','L':'L','W':'A','D':'D'}

A_QPE = 316
B_QPE = 1.5

MAX_VPR_CORRECTION_DB = 4.77

NBINS_Y = 710
NBINS_X = 640

LOCAL_BIAS =  np.fromfile(ObjStorage.check_file(str(Path(metadata_folder, 'lbias_qpegrid.dat'))),
                         dtype = np.float32).reshape(NBINS_Y,NBINS_X)

GLOBAL_BIAS = {'A':3.5, 'D': 2., 'L':4., 'P':2., 'W':1.5}
Y_QPE = np.linspace(255, 965, NBINS_Y + 1)
X_QPE = np.linspace(480, -160,  NBINS_X + 1)
Z_QPE = np.load(ObjStorage.check_file(str(Path(metadata_folder, 'z_qpegrid.npy'))))

try: # when sphinx imports this module it crashes here because it considers numpy as mock module
    Y_QPE_CENTERS = 0.5 * (Y_QPE[0:-1] + Y_QPE[1:])
except:
    Y_QPE_CENTERS = [1]

try: # when sphinx imports this module it crashes here because it considers numpy as mock module
    X_QPE_CENTERS = 0.5 * (X_QPE[0:-1] + X_QPE[1:])
except:
    X_QPE_CENTERS = [1]

MASK_NAN = np.load(ObjStorage.check_file(str(Path(metadata_folder, 'mask_nan.npy'))))

        
dic = np.load(ObjStorage.check_file(str(Path(metadata_folder, 'scale_RGB.npz'))))
SCALE_RGB = {'colors':dic['arr_0'],'values':dic['arr_1']}


SCALE_CPC = np.array([0.000000e+00,0.000000e+00,7.177341e-02,1.095694e-01,1.486983e-01,
1.892071e-01,2.311444e-01,2.745606e-01,3.195080e-01,3.660402e-01,
4.142135e-01,4.640857e-01,5.157166e-01,5.691682e-01,6.245048e-01,
6.817929e-01,7.411011e-01,8.025010e-01,8.660660e-01,9.318726e-01,
1.000000e+00,1.070530e+00,1.143547e+00,1.219139e+00,1.297397e+00,
1.378414e+00,1.462289e+00,1.549121e+00,1.639016e+00,1.732080e+00,
1.828427e+00,1.928171e+00,2.031433e+00,2.138336e+00,2.249010e+00,
2.363586e+00,2.482202e+00,2.605002e+00,2.732132e+00,2.863745e+00,
3.000000e+00,3.141060e+00,3.287094e+00,3.438278e+00,3.594793e+00,
3.756828e+00,3.924578e+00,4.098242e+00,4.278032e+00,4.464161e+00,
4.656854e+00,4.856343e+00,5.062866e+00,5.276673e+00,5.498019e+00,
5.727171e+00,5.964405e+00,6.210004e+00,6.464264e+00,6.727490e+00,
7.000000e+00,7.282120e+00,7.574187e+00,7.876555e+00,8.189587e+00,
8.513657e+00,8.849155e+00,9.196485e+00,9.556064e+00,9.928322e+00,
1.031371e+01,1.071269e+01,1.112573e+01,1.155335e+01,1.199604e+01,
1.245434e+01,1.292881e+01,1.342001e+01,1.392853e+01,1.445498e+01,
1.500000e+01,1.556424e+01,1.614837e+01,1.675311e+01,1.737917e+01,
1.802731e+01,1.869831e+01,1.939297e+01,2.011213e+01,2.085664e+01,
2.162742e+01,2.242537e+01,2.325146e+01,2.410669e+01,2.499208e+01,
2.590869e+01,2.685762e+01,2.784002e+01,2.885706e+01,2.990996e+01,
3.100000e+01,3.212848e+01,3.329675e+01,3.450622e+01,3.575835e+01,
3.705463e+01,3.839662e+01,3.978594e+01,4.122425e+01,4.271329e+01,
4.425483e+01,4.585074e+01,4.750293e+01,4.921338e+01,5.098415e+01,
5.281737e+01,5.471524e+01,5.668003e+01,5.871411e+01,6.081992e+01,
6.300000e+01,6.525696e+01,6.759350e+01,7.001244e+01,7.251669e+01,
7.510925e+01,7.779324e+01,8.057188e+01,8.344851e+01,8.642657e+01,
8.950967e+01,9.270148e+01,9.600586e+01,9.942677e+01,1.029683e+02,
1.066347e+02,1.104305e+02,1.143601e+02,1.184282e+02,1.226398e+02,
1.270000e+02,1.315139e+02,1.361870e+02,1.410249e+02,1.460334e+02,
1.512185e+02,1.565865e+02,1.621438e+02,1.678970e+02,1.738531e+02,
1.800193e+02,1.864030e+02,1.930117e+02,1.998535e+02,2.069366e+02,
2.142695e+02,2.218609e+02,2.297201e+02,2.378564e+02,2.462797e+02,
2.550000e+02,2.640278e+02,2.733740e+02,2.830498e+02,2.930668e+02,
3.034370e+02,3.141730e+02,3.252875e+02,3.367940e+02,3.487063e+02,
3.610387e+02,3.738059e+02,3.870234e+02,4.007071e+02,4.148732e+02,
4.295390e+02,4.447219e+02,4.604402e+02,4.767129e+02,4.935594e+02,
5.110000e+02,5.290557e+02,5.477480e+02,5.670995e+02,5.871335e+02,
6.078740e+02,6.293459e+02,6.515750e+02,6.745881e+02,6.984126e+02,
7.230773e+02,7.486119e+02,7.750469e+02,8.024141e+02,8.307465e+02,
8.600779e+02,8.904438e+02,9.218805e+02,9.544258e+02,9.881188e+02,
1.023000e+03,1.059111e+03,1.096496e+03,1.135199e+03,1.175267e+03,
1.216748e+03,1.259692e+03,1.304150e+03,1.350176e+03,1.397825e+03,
1.447155e+03,1.498224e+03,1.551094e+03,1.605828e+03,1.662493e+03,
1.721156e+03,1.781888e+03,1.844761e+03,1.909852e+03,1.977238e+03,
2.047000e+03,2.119223e+03,2.193992e+03,2.271398e+03,2.351534e+03,
2.434496e+03,2.520384e+03,2.609300e+03,2.701352e+03,2.796650e+03,
2.895309e+03,2.997448e+03,3.103188e+03,3.212656e+03,3.325986e+03,
3.443312e+03,3.564775e+03,3.690522e+03,3.820703e+03,3.955475e+03,
4.095000e+03,4.239445e+03,4.388984e+03,4.543796e+03,4.704068e+03,
4.869992e+03,5.041768e+03,5.219600e+03,5.403705e+03,5.594301e+03,
5.79e+03    ,0           ,0           ,0           ,0  ,       0])


SCALE_CPC_OLD = np.array([0,0.000000e+00,3.526497e-02,7.177341e-02,1.095694e-01,1.486983e-01,
1.892071e-01,2.311444e-01,2.745606e-01,3.195080e-01,3.660402e-01,
4.142135e-01,4.640857e-01,5.157166e-01,5.691682e-01,6.245048e-01,
6.817929e-01,7.411011e-01,8.025010e-01,8.660660e-01,9.318726e-01,
1.000000e+00,1.070530e+00,1.143547e+00,1.219139e+00,1.297397e+00,
1.378414e+00,1.462289e+00,1.549121e+00,1.639016e+00,1.732080e+00,
1.828427e+00,1.928171e+00,2.031433e+00,2.138336e+00,2.249010e+00,
2.363586e+00,2.482202e+00,2.605002e+00,2.732132e+00,2.863745e+00,
3.000000e+00,3.141060e+00,3.287094e+00,3.438278e+00,3.594793e+00,
3.756828e+00,3.924578e+00,4.098242e+00,4.278032e+00,4.464161e+00,
4.656854e+00,4.856343e+00,5.062866e+00,5.276673e+00,5.498019e+00,
5.727171e+00,5.964405e+00,6.210004e+00,6.464264e+00,6.727490e+00,
7.000000e+00,7.282120e+00,7.574187e+00,7.876555e+00,8.189587e+00,
8.513657e+00,8.849155e+00,9.196485e+00,9.556064e+00,9.928322e+00,
1.031371e+01,1.071269e+01,1.112573e+01,1.155335e+01,1.199604e+01,
1.245434e+01,1.292881e+01,1.342001e+01,1.392853e+01,1.445498e+01,
1.500000e+01,1.556424e+01,1.614837e+01,1.675311e+01,1.737917e+01,
1.802731e+01,1.869831e+01,1.939297e+01,2.011213e+01,2.085664e+01,
2.162742e+01,2.242537e+01,2.325146e+01,2.410669e+01,2.499208e+01,
2.590869e+01,2.685762e+01,2.784002e+01,2.885706e+01,2.990996e+01,
3.100000e+01,3.212848e+01,3.329675e+01,3.450622e+01,3.575835e+01,
3.705463e+01,3.839662e+01,3.978594e+01,4.122425e+01,4.271329e+01,
4.425483e+01,4.585074e+01,4.750293e+01,4.921338e+01,5.098415e+01,
5.281737e+01,5.471524e+01,5.668003e+01,5.871411e+01,6.081992e+01,
6.300000e+01,6.525696e+01,6.759350e+01,7.001244e+01,7.251669e+01,
7.510925e+01,7.779324e+01,8.057188e+01,8.344851e+01,8.642657e+01,
8.950967e+01,9.270148e+01,9.600586e+01,9.942677e+01,1.029683e+02,
1.066347e+02,1.104305e+02,1.143601e+02,1.184282e+02,1.226398e+02,
1.270000e+02,1.315139e+02,1.361870e+02,1.410249e+02,1.460334e+02,
1.512185e+02,1.565865e+02,1.621438e+02,1.678970e+02,1.738531e+02,
1.800193e+02,1.864030e+02,1.930117e+02,1.998535e+02,2.069366e+02,
2.142695e+02,2.218609e+02,2.297201e+02,2.378564e+02,2.462797e+02,
2.550000e+02,2.640278e+02,2.733740e+02,2.830498e+02,2.930668e+02,
3.034370e+02,3.141730e+02,3.252875e+02,3.367940e+02,3.487063e+02,
3.610387e+02,3.738059e+02,3.870234e+02,4.007071e+02,4.148732e+02,
4.295390e+02,4.447219e+02,4.604402e+02,4.767129e+02,4.935594e+02,
5.110000e+02,5.290557e+02,5.477480e+02,5.670995e+02,5.871335e+02,
6.078740e+02,6.293459e+02,6.515750e+02,6.745881e+02,6.984126e+02,
7.230773e+02,7.486119e+02,7.750469e+02,8.024141e+02,8.307465e+02,
8.600779e+02,8.904438e+02,9.218805e+02,9.544258e+02,9.881188e+02,
1.023000e+03,1.059111e+03,1.096496e+03,1.135199e+03,1.175267e+03,
1.216748e+03,1.259692e+03,1.304150e+03,1.350176e+03,1.397825e+03,
1.447155e+03,1.498224e+03,1.551094e+03,1.605828e+03,1.662493e+03,
1.721156e+03,1.781888e+03,1.844761e+03,1.909852e+03,1.977238e+03,
2.047000e+03,2.119223e+03,2.193992e+03,2.271398e+03,2.351534e+03,
2.434496e+03,2.520384e+03,2.609300e+03,2.701352e+03,2.796650e+03,
2.895309e+03,2.997448e+03,3.103188e+03,3.212656e+03,3.325986e+03,
3.443312e+03,3.564775e+03,3.690522e+03,3.820703e+03,3.955475e+03,
4.095000e+03,4.239445e+03,4.388984e+03,4.543796e+03,4.704068e+03,
4.869992e+03,5.041768e+03,5.219600e+03,5.403705e+03,5.594301e+03,
0           ,0           ,0           ,0                      ,0])

###############
# Graphics
###############

path_shp = ObjStorage.check_file(str(Path(metadata_folder,'swiss_border_shp', 'Border_CH.shp')))
BORDER_SHP = shapefile.Reader(str(path_shp))


###############
# Data retrieval
###############
FILTER_COMMAND = '~owm/bin/fxfilter'
CONVERT_COMMAND = '~owm/bin/fxconvert'
OFFSET_CCS4 = [297,-100]

# Folder depends on server:
if ('lom' in socket.gethostname()) or ('meteoswiss' in socket.gethostname()):
    FOLDER_RADAR = '/srn/data/'
    FOLDER_ISO0 = '/srn/data/HZT/'
elif 'tsa' in socket.gethostname():
    FOLDER_DATABASE = '/store/msrad/radar/radar_database/'
    FOLDER_RADAR = '/store/msrad/radar/swiss/data/'
    FOLDER_RADARH = '/store/msrad/radar/polarHR/data/'
    FOLDER_CPCCV = '/store/msrad/radar/cpc_validation/daily/'

    FOLDER_ISO0 = '/store/msrad/radar/swiss/data/'
    COSMO1_START = datetime.datetime(2015,10,1)
    COSMO1E_START = datetime.datetime(2019,7,2)
    FOLDER_COSMO1 = '/store/s83/owm/COSMO-1/'
    FOLDER_COSMO1E = '/store/s83/osm/KENDA-1/'
    FOLDER_COSMO1_T = '/store/s83/owm/COSMO-1/ORDERS/MDR/'
    FOLDER_COSMO1E_T = '/store/s83/osm/COSMO-1E/ORDERS/MDR/'
    FOLDER_COSMO2_T = '/store/msrad/cosmo/cosmo2/data/'
    FILES_COSMO1_T = sorted(glob.glob(FOLDER_COSMO1_T + '*.nc'))
    TIMES_COSMO1_T = np.array([datetime.datetime.strptime(f[-13:-3],'%Y%m%d%H')
        for f in FILES_COSMO1_T])
    FILES_COSMO1E_T = sorted(glob.glob(FOLDER_COSMO1E_T + '*.nc'))
    TIMES_COSMO1E_T = np.array([datetime.datetime.strptime(f[-13:-3],'%Y%m%d%H')
        for f in FILES_COSMO1E_T])

 ###############
 # Radar processing
###############

PYART_NAMES_MAPPING = {'reflectivity':'ZH',
                       'differential_reflectivity':'ZDR',
                       'uncorrected_differential_phase':'PSIDP',
                       'spectrum_width':'SW',
                       'velocity':'RVEL',
                       'reflectivity_vv':'ZV',
                       'uncorrected_cross_correlation_ratio':'RHOHV'}

NOISE_100 = 5 # Noise level at 100 km

MIN_RZC_VALID = 0.04  # everything below will be put to zero

def MODE(x):
    if not np.any(np.isfinite(x)):
        return np.nan
    else:
        return mode(x).mode

PYART_NAMES_MAPPING = {'reflectivity':'ZH',
                       'differential_reflectivity':'ZDR',
                       'uncorrected_differential_phase':'PSIDP',
                       'spectrum_width':'SW',
                       'velocity':'RVEL',
                       'reflectivity_vv':'ZV',
                       'uncorrected_cross_correlation_ratio':'RHOHV'}
NUM_RADAR_PER_GAUGE = 2 # 2 5-min radar scans per 10-min gauge accumulation


AVG_BY_VAR = {'ZH': 1 , 'ZH_VISIB' : 1, 'ZV': 1, 'ZV_VISIB' : 1, 'ZDR': 1, 
              'ZV_CORR': 1, 'ZH_CORR':1 , 'ZDR_CORR':1,
              'NH': 1, 'NV': 1, 'TCOUNT':2}


AVG_METHODS = {}
AVG_METHODS[0] = lambda x, axis: np.nanmean(x, axis = axis)
AVG_METHODS[1] = lambda x, axis: 10 * np.log10(np.nanmean(10**(0.1 * x), axis = axis))
AVG_METHODS[2] = lambda x, axis: np.nansum(x, axis = axis)


WARNING_RAM = 512 # megabytes
SLURM_HEADER_R = '''#!/bin/sh
#SBATCH -N 1     # nodes requested
#SBATCH -c 1      # cores requested
#SBATCH -t 12:0:00  # time requested in hour:minute:second
#SBATCH --partition=postproc
#SBATCH --exclude=tsa-pp020,tsa-pp019,tsa-pp018
#SBATCH --output="db_gauges_%A.out"
#SBATCH --error="db_gauges_%A.err"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rebecca.gugerli@meteoswiss.ch

export MODULEPATH="/store/msclim/share/modulefiles/modules/all:${MODULEPATH}"
module load /store/msclim/share/modulefiles/modules/all/cat/tsa-R3.5.2

export RAINFOREST_DATAPATH=/store/msrad/radar/rainforest/rainforest_data/

source /scratch/rgugerli/miniconda3/etc/profile.d/conda.sh
conda activate rainforest_RandPython
'''


SLURM_HEADER_PY = '''#!/bin/sh
#SBATCH -N 1     # nodes requested
#SBATCH -c 1      # cores requested
#SBATCH --mem-per-cpu 64g # memory in mbytes  
#SBATCH -t 23:59:59  # time requested in hour:minute:second
#SBATCH --partition=postproc
#SBATCH --exclude=tsa-pp020,tsa-pp019,tsa-pp018
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rebecca.gugerli@meteoswiss.ch
'''

# Default (not indicated) is np.float32
COL_TYPES = {'TIMESTAMP':np.int32, 
         'RADAR':str,
         'SWEEP':np.dtype(np.int8),
         'NX':np.dtype(np.int8),
         'NY':np.dtype(np.int8),         
         'STATION':str,
         'HYDRO':np.int8,
         'VISIB':np.int8,
         'TCOUNT':np.int8}    


HYDRO_CENTROIDS = {}
HYDRO_CENTROIDS['A'] = [[13.5829, 0.4063, 0.0497, 0.9868, 1330.3],  # AG
         [02.8453, 0.2457, 0.0000, 0.9798, 0653.8],  # CR
         [07.6597, 0.2180, 0.0019, 0.9799, -1426.5],  # LR
         [31.6815, 0.3926, 0.0828, 0.9978, 0535.3],  # RP
         [39.4703, 1.0734, 0.4919, 0.9876, -1036.3],  # RN
         [04.8267, -0.5690, 0.0000, 0.9691, 0869.8],  # VI
         [30.8613, 0.9819, 0.1998, 0.9845, -0066.1],  # WS
         [52.3969, 2.1094, 2.4675, 0.9730, -1550.2],  # MH
         [50.6186, -0.0649, 0.0946, 0.9904, 1179.9]]  # IH/HDG
HYDRO_CENTROIDS['L'] = [[13.8231, 0.2514, 0.0644, 0.9861, 1380.6],  # AG
         [03.0239, 0.1971, 0.0000, 0.9661, 1464.1],  # CR
         [04.9447, 0.1142, 0.0000, 0.9787, -0974.7],  # LR
         [34.2450, 0.5540, 0.1459, 0.9937, 0945.3],  # RP
         [40.9432, 1.0110, 0.5141, 0.9928, -0993.5],  # RN
         [03.5202, -0.3498, 0.0000, 0.9746, 0843.2],  # VI
         [32.5287, 0.9751, 0.2640, 0.9804, -0055.5],  # WS
         [52.6547, 2.7054, 2.5101, 0.9765, -1114.6],  # MH
         [46.4998, 0.1978, 0.6431, 0.9845, 1010.1]]  # IH/HDG
HYDRO_CENTROIDS['D'] = [[12.567, 0.18934, 0.041193, 0.97693, 1328.1],  # AG
         [3.2115, 0.13379, 0.0000, 0.96918, 1406.3],  # CR
         [10.669, 0.18119, 0.0000, 0.97337, -1171.9],  # LR
         [34.941, 0.13301, 0.090056, 0.9979, 898.44],  # RP
         [39.653, 1.1432, 0.35013, 0.98501, -859.38],  # RN
         [2.8874, -0.46363, 0.0000, 0.95653, 1015.6],  # VI
         [34.122, 0.87987, 0.2281, 0.98003, -234.37],  # WS
         [53.134, 2.0888, 2.0055, 0.96927, -1054.7],  # MH
         [46.715, 0.030477, 0.16994, 0.9969, 976.56]]  # IH/HDG
HYDRO_CENTROIDS['P'] =  [[13.9882, 0.2470, 0.0690, 0.9939, 1418.1],  # AG
         [00.9834, 0.4830, 0.0043, 0.9834, 0950.6],  # CR
         [05.3962, 0.2689, 0.0000, 0.9831, -0479.5],  # LR
         [35.3411, 0.1502, 0.0940, 0.9974, 0920.9],  # RP
         [35.0114, 0.9681, 0.1106, 0.9785, -0374.0],  # RN
         [02.5897, -0.3879, 0.0282, 0.9876, 0985.5],  # VI
         [32.2914, 0.7789, 0.1443, 0.9075, -0153.5],  # WS
         [53.2413, 1.8723, 0.3857, 0.9454, -0470.8],  # MH
         [44.7896, 0.0015, 0.1349, 0.9968, 1116.7]]  # IH/HDG
HYDRO_CENTROIDS['W'] = [[16.7650, 0.3754, 0.0442, 0.9866, 1409.0],  # AG
         [01.4418, 0.3786, 0.0000, 0.9490, 1415.8],  # CR
         [16.0987, 0.3238, 0.0000, 0.9871, -0818.7],  # LR
         [36.5465, 0.2041, 0.0731, 0.9952, 0745.4],  # RP
         [43.4011, 0.6658, 0.3241, 0.9894, -0778.5],  # RN
         [00.9077, -0.4793, 0.0000, 0.9502, 1488.6],  # VI
         [36.8091, 0.7266, 0.1284, 0.9924, -0071.1],  # WS
         [53.8402, 0.8922, 0.5306, 0.9890, -1017.6],  # MH
         [45.9686, 0.0845, 0.0963, 0.9940, 0867.4]]  # IH/HDG



