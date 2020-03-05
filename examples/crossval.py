#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs cross-validation of 3 RF models, and CPC and RZC products with
reference to rain gauges
"""
from rainforest.ml.rf import RFTraining


rf = RFTraining('/store/msrad/radar/radar_database/')

# Create dictionnary of models to compare
models = {}
models['RF_dualpol'] =  ["RADAR", "zh_VISIB_mean", "zv_VISIB_mean",
                         "KDP_mean","RHOHV_mean","T",
                         "HEIGHT","VISIB_mean"]
models['RF_hpol'] =  ["RADAR", "zh_VISIB_mean","T",
                         "HEIGHT","VISIB_mean"]
models['RF_vpol'] =  ["RADAR", "zv_VISIB_mean","T",
                         "HEIGHT","VISIB_mean"]
# Path of configuration files
config = './intercomparison_config_example.yml'

output_folder = './output'

rf.model_intercomparison(models, config, output_folder, ['CPCH','RZC'])
