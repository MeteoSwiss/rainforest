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
from rainforest.common.retrieve_data import retrieve_prod

##################
# User input
##################
days = ["26197"]  # either in yyjjj or yyyymmdd format
models = '{"RF_dualpol_":"RFO_BETA_-0.5_BC_spline.p"}'
outputfolder = "/scratch/mch/wolfensb/content/rfo_recompute/"
gauge = "'/store_new/mch/msrad/radar/radar_database_v2/gauge/*.csv.gz'"  # beeware of quotes!
config = "/users/wolfensb/github_repos/rainforest/rainforest/qpe/default_config.yml"
only_compute = True  # if set to false, the intercomparison of the computed rfo with gauge + CPC, RZC will be done

SLURM_HEADER_PY = """#!/bin/sh
#SBATCH -N 1     # nodes requested
#SBATCH -c 1      # cores requested
#SBATCH --mem-per-cpu 64g # memory in mbytes
#SBATCH -t 23:59:59  # time requested in hour:minute:second
#SBATCH --partition=postproc
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.wolfensberger@meteoswiss.ch
"""
##################

for d in days:
    folder = str(Path(outputfolder, d))
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open(d + "_job", "w")
    f.write(SLURM_HEADER_PY)
    dformat = "%y%j" if len(d) == 5 else "%Y%m%d"
    start = datetime.datetime.strptime(d, dformat).strftime("%Y%m%d%H%M")
    end = datetime.datetime.strptime(d, dformat) + datetime.timedelta(days=1)
    end = datetime.datetime.strftime(end, "%Y%m%d%H%M")
    f.write(
        "qpe_compute -s {:s} -e {:s} -m '{:s}' -o {:s} -c {:s}".format(
            start, end, models, folder, config
        )
    )
    f.close()
    subprocess.call("sbatch {:s}".format(d + "_job"), shell=True)

if not only_compute:
    # Download RZC, CPC
    for d in days:
        folder_rzc = str(Path(outputfolder, d, "RZC"))
        if not os.path.exists(folder_rzc):
            os.makedirs(folder_rzc)
        folder_cpc = str(Path(outputfolder, d, "CPCH"))
        if not os.path.exists(folder_cpc):
            os.makedirs(folder_cpc)

        start = datetime.datetime.strptime(d, "%Y%m%d")
        end = datetime.datetime.strptime(d, "%Y%m%d") + datetime.timedelta(days=1)

        retrieve_prod(folder_rzc, start, end, "RZC")
        retrieve_prod(folder_cpc, start, end, "CPCH", pattern="*5.801.gif")

    # Evaluate QPE
    for d in days:
        folder = str(Path(outputfolder, d))
        output = str(Path(outputfolder, "plots_allstaq"))
        if not os.path.exists(output):
            os.makedirs(output)

        f = open(d + "_job", "w")
        f.write(SLURM_HEADER_PY)
        f.write(
            "qpe_evaluation -q {:s} -g {:s} -o {:s} -m {:s} ".format(
                folder, gauge, output, "CPCH,CPC_RF"
            )
        )
        f.close()
        subprocess.call("sbatch {:s}".format(d + "_job"), shell=True)

    # Plot QPE
    for d in days:
        folder = str(Path(outputfolder, d))
        output = str(Path(outputfolder, "plots"))
        if not os.path.exists(output):
            os.makedirs(output)
        f = open(d + "_job", "w")
        f.write(SLURM_HEADER_PY)
        f.write(
            "qpe_plot -i {:s} -o {:s} -V 20.1 -t 1 -m {:s} -f 10,7 -d 2,2 -c 'vertical'".format(
                folder, output, '"RZC, CPC, RF_hpol","RF_dualpol"'
            )
        )
        f.close()
        subprocess.call("sbatch {:s}".format(d + "_job"), shell=True)
