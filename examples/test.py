#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:22:08 2020

@author: wolfensb
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rainforest.ml.utils import vert_aggregation
from rainforest.ml.rfdefinitions import RandomForestRegressorBC
import pickle

input_location = '/store/msrad/radar/radar_database/rf_input_data/'
radartab = pd.read_parquet(str(Path(input_location, 'radar_x0y0.parquet')))
refertab = pd.read_parquet(str(Path(input_location, 'reference_x0y0.parquet')))
gaugetab = pd.read_parquet(str(Path(input_location, 'gauge.parquet')))
grp = pickle.load(open(str(Path(input_location, 'grouping_idx_x0y0.p')),'rb'))
grp_vertical = grp['grp_vertical']
grp_hourly = grp['grp_hourly']

vweights = 10**(-0.5 *(radartab['HEIGHT']/1000.)) # vert. weights

features = ['RADAR', 'zh_VISIB_mean',
            'zv_VISIB_mean','KDP_mean','RHOHV_mean','T', 'HEIGHT','VISIB_mean']

for f in features:
    if 'zh' in f:
      
        radartab[f] = 10**(0.1 * radartab[f.replace('zh','ZH')])
    elif 'zv' in f:
     
        radartab[f] = 10**(0.1 * radartab[f.replace('zv','ZV')])        

            

ZH_agg = vert_aggregation(radartab[features],
                                  vweights,
                                  grp_vertical,
                                  True, radartab['VISIB_mean'])



reg = RandomForestRegressorBC(degree = 1, 
                         bctype = 'raw',
                         variables = features,
                         beta = -0.5,
                         n_estimators = 20,
                         max_depth = 20,
                         n_jobs = 12,
                         verbose = 10)

# remove nans
valid = np.all(np.isfinite(ZH_agg),
                 axis = 1)
  

ZH_agg = ZH_agg[valid]
  
gaugetab = gaugetab[valid]
refertab = refertab[valid]
grp_hourly = grp_hourly[valid]
# Get R, T and idx test/train
R = np.array(gaugetab['RRE150Z0'] * 6) # Reference precip in mm/h
R[np.isnan(R)] = 0
        
        

reg.fit(ZH_agg, R)
out = reg.predict(ZH_agg)
