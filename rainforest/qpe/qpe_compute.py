#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line script for the RandomForest QPE

see :ref:`qpe_compute`
"""

# Global imports
import json
import os
import datetime
from pathlib import Path
import argparse

# Local imports
from rainforest.qpe.qpe import QPEProcessor
from rainforest.ml.rfdefinitions import read_rf

# Suppress certain warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    parser = argparse.ArgumentParser(
        description="Run QPE model predictions for a given time range."
    )

    parser.add_argument(
        "-s",
        "--start",
        dest="start",
        type=str,
        help="Specify the start time in the format YYYYddmmHHMM",
        metavar="START",
    )

    parser.add_argument(
        "-e",
        "--end",
        dest="end",
        type=str,
        help="Specify the end time in the format YYYYddmmHHMM",
        metavar="END",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="outputfolder",
        type=str,
        default="./",
        help="Path of the output folder, default is current folder",
        metavar="OUTPUT",
    )

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path of the config file. Default is 'default_config.yml' in the qpe module",
        metavar="CONFIG",
    )

    parser.add_argument(
        "-m",
        "--models",
        dest="models",
        type=str,
        default='{"RF_dualpol":"RF_dualpol_BETA_-0.5_BC_spline.p"}',
        help=(
            "Specify which models you want to use in JSON format. "
            "Models must be in /ml/rf_models/ unless overridden by --modelpath. "
            'Example: \'{"RF_dualpol":"RF_dualpol_BETA_-0.5_BC_spline.p"}\'. '
            "Note: keep the double and single quotes as shown."
        ),
        metavar="MODELS",
    )

    parser.add_argument(
        "-M",
        "--mlflow_model",
        dest="mlflow_model",
        action="store_true",
        default=False,
        help=(
            "Defines whether the given model(s) is/are stored on MLflow. "
            "If true, the value passed to -m is/are the run ID(s) of MLflow."
        ),
    )

    parser.add_argument(
        "-p",
        "--modelpath",
        dest="modelpath",
        type=str,
        default=None,
        help="Specify where the models are stored if not in $RAINFOREST_DATAPATH/rf_models/",
        metavar="MODELPATHS",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "-r",
        "--rt",
        dest="rt",
        action="store_true",
        help="To be set if the code is to run on a server where the input data is available in real-time (uncompressed), as is usually the case on LOM/ZUE servers",
    )

    args = parser.parse_args()

    if args.config is None:
        script_path = os.path.dirname(os.path.realpath(__file__))
        args.config = str(Path(script_path, "default_config.yml"))

    args.models = json.loads(args.models)
    for k in args.models.keys():
        if args.mlflow_model:
            args.models[k] = read_rf(mlflow_runid=args.models[k])
        elif args.modelpath is None:
            args.models[k] = read_rf(args.models[k])
        else:
            args.models[k] = read_rf(args.models[k], filepath=args.modelpath)

    if not os.path.exists(args.outputfolder):
        os.makedirs(args.outputfolder)

    args.start = datetime.datetime.strptime(args.start, "%Y%m%d%H%M")
    args.end = datetime.datetime.strptime(args.end, "%Y%m%d%H%M")
    qpe = QPEProcessor(args.config, args.models, rt=args.rt, verbose=args.verbose)
    qpe.compute(args.outputfolder, args.start, args.end)


if __name__ == "__main__":
    main()
