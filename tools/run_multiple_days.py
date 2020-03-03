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

days = ['20190610','20190806','20190922','20191015','20191124','20200129']
models = '{"RF_dualpol":"dualpol_model_BC_spline.p"}'
outputfolder = '/scratch/wolfensb/qpe_runs/'
gauge = "'/store/msrad/radar/radar_database/gauge/*.csv.gz'" # beeware of quotes!

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
    f.write("qpe_compute -s {:s} -e {:s} -m '{:s}' -o {:s}".format(start, end, models,folder))
    f.close()
#    subprocess.call('sbatch {:s}'.format(d + '_job'), shell = True)
    
    
# %%
# Download RZC, CPC
for d in days:
    folder_rzc = str(Path(outputfolder, d, 'RZC'))
    if not os.path.exists(folder_rzc):
        os.makedirs(folder_rzc)
    folder_cpc = str(Path(outputfolder, d, 'CPC'))
    if not os.path.exists(folder_cpc):
        os.makedirs(folder_cpc)
        
    start = datetime.datetime.strptime(d, '%Y%m%d')
    end = datetime.datetime.strptime(d, '%Y%m%d') + datetime.timedelta(days = 1)

    retrieve_prod(folder_rzc, start, end, 'RZC')
    retrieve_prod(folder_cpc, start, end, 'CPC', pattern = '*5.801.gif') 
    
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
            '"RZC, CPC, RF_dualpol, RF_hpol, RF_vpol"'))
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
    f.write("qpe_plot -i {:s} -o {:s} -V 80 -t 3".format(folder, output))
    f.close()
    subprocess.call('sbatch {:s}'.format(d + '_job'), shell = True)
    