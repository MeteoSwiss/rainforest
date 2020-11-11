#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:50:09 2020

@author: wolfensb
"""

import os
from pathlib import Path
import datetime
import subprocess
from rainforest.common.constants import SLURM_HEADER_PY
from rainforest.common.retrieve_data import retrieve_prod
#  
days = ['20170725','20180122','20181027']
# days = ['20180122']
days.extend([        '20190806','20191015','20200129'])

models = '{"RF_dualpol_noS":"RF_dualpol_BETA_-0.5_BC_spline.p","RF_hpol_noS":"RF_hpol_BETA_-0.5_BC_spline.p"}'
# models = '{"RF_dualpol":"RF_dualpol_BETA_-0.5_BC_spline.p"}'
outputfolder = '/scratch/wolfensb/qpe_runs/'
gauge = "'/store/msrad/radar/radar_database/gauge/*.csv.gz'" # beeware of quotes!
config = '/store/msrad/radar/rainforest/tools/config.yml'

# from rainforest.qpe.qpe import QPEProcessor
# from rainforest.ml.rfdefinitions import read_rf
# models = {'RF_dualpol':read_rf('RF_dualpol_BETA_-0.5_BC_spline.p')}
# q = QPEProcessor(config, models)
# t0 = datetime.datetime(2017,7,25,0,0)
# t1 = datetime.datetime(2017,7,25,6,0)
# q.compute(outputfolder,t0,t1)
# %%
# Compute QPE

for d in days:
    folder = str(Path(outputfolder, d))
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open(d + '_job', 'w')
    f.write(SLURM_HEADER_PY) 
    start = d + '0000'
    end = datetime.datetime.strptime(d, '%Y%m%d') + datetime.timedelta(days = 1)
    end = datetime.datetime.strftime(end, '%Y%m%d%H%M')
    f.write("qpe_compute -s {:s} -e {:s} -m '{:s}' -o {:s} -c {:s}".format(start, end, models,folder, config))
    f.close()
    subprocess.call('sbatch {:s}'.format(d + '_job'), shell = True)
  
# %%
# Download RZC, CPC
for d in days:
    folder_rzc = str(Path(outputfolder, d, 'RZC'))
    if not os.path.exists(folder_rzc): 
        os.makedirs(folder_rzc) 
    folder_cpc = str(Path(outputfolder, d, 'CPCH'))
    if not os.path.exists(folder_cpc):
        os.makedirs(folder_cpc)
        
    start = datetime.datetime.strptime(d, '%Y%m%d')
    end = datetime.datetime.strptime(d, '%Y%m%d') + datetime.timedelta(days = 1)

    retrieve_prod(folder_rzc, start, end, 'RZC')
    retrieve_prod(folder_cpc, start, end, 'CPCH', pattern = '*5.801.gif') 
    
# %%
# Evaluate QPE
for d in days:
    folder = str(Path(outputfolder, d))
    output = str(Path(outputfolder, 'plots_allstaq'))
    if not os.path.exists(output):
        os.makedirs(output)
        
    f = open(d + '_job', 'w')
    f.write(SLURM_HEADER_PY)
    f.write("qpe_evaluation -q {:s} -g {:s} -o {:s} -m {:s} ".format(folder, gauge, output,
            'CPCH,CPC_RF'))
    f.close()
    subprocess.call('sbatch {:s}'.format(d + '_job'), shell = True)
        # %%
# Plot QPE
for d in days:
    folder = str(Path(outputfolder, d))
    output = str(Path(outputfolder, 'plots'))
    if not os.path.exists(output):
        os.makedirs(output)
    f = open(d + '_job', 'w')
    f.write(SLURM_HEADER_PY)
    f.write("qpe_plot -i {:s} -o {:s} -V 20.1 -t 1 -m {:s} -f 10,7 -d 2,2 -c 'vertical'".format(folder, output,'"RZC, CPC, RF_hpol","RF_dualpol"'))
    f.close()
    subprocess.call('sbatch {:s}'.format(d + '_job'), shell = True)
    