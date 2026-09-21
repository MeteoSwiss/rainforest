#!/usr/bin/env python3
"""
Command line script to prepare input features and train RF models.

See :ref:`rf_train`
"""

# Global imports
import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

from rainforest.common import constants

# Local imports
from rainforest.ml.trainer import RFTraining


logging.basicConfig(level=logging.INFO)
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_path)


def parse_datetime(value):
    """Parse a datetime in YYYYMMDDHHMM format."""
    try:
        return datetime.datetime.strptime(value, "%Y%m%d%H%M")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Expected format: YYYYMMDDHHMM"
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare input features and train Random Forest models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--outputfolder",
        type=str,
        default=None,
        help=(
            "Path of the output folder. By default, the 'rf_models' "
            "folder in the current library is used."
        ),
        metavar="OUTPUT",
    )

    parser.add_argument(
        "-d",
        "--dbfolder",
        type=str,
        default=constants.FOLDER_DATABASE,
        help="Path of the database main folder.",
        metavar="DBFOLDER",
    )

    parser.add_argument(
        "-i",
        "--inputfolder",
        type=str,
        default=None,
        help=(
            "Path where the homogenized input files for the RF algorithm "
            "are stored. By default, 'rf_input_data' within the database "
            "folder is used."
        ),
        metavar="INPUTFOLDER",
    )

    parser.add_argument(
        "-s",
        "--start",
        type=parse_datetime,
        default=None,
        help=(
            "Start time in YYYYMMDDHHMM format. If not provided, the first "
            "timestamp in the database will be used."
        ),
        metavar="START",
    )

    parser.add_argument(
        "-e",
        "--end",
        type=parse_datetime,
        default=None,
        help=(
            "End time in YYYYMMDDHHMM format. If not provided, the last "
            "timestamp in the database will be used."
        ),
        metavar="END",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help=(
            "Path of the config file. By default, default_config.yml "
            "in the ml module is used."
        ),
        metavar="CONFIG",
    )

    parser.add_argument(
        "-r",
        "--rfmodels",
        type=str,
        default=None,
        help=(
            "Models to train. This can either be a JSON dictionary or a "
            "comma-separated list of predefined aliases. "
            "Available aliases: dualpol_default, vpol_default, hpol_default, "
            "RFO_default, RFO_default_vpol, RFO_default_hpol. "
            "If omitted, input data are generated but no model is trained. "
            "Example JSON: "
            '\'{"RF_dualpol": ["RADAR", "zh_VISIB_mean", "zv_VISIB_mean", '
            '"KDP_mean", "RHOHV_mean", "T", "HEIGHT", "VISIB_mean"]}\''
        ),
        metavar="MODELS",
    )

    parser.add_argument(
        "-g",
        "--generate-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate/recompute the homogenized parquet input files from "
            "the current database rows. Use --no-generate-inputs to disable."
        ),
    )

    parser.add_argument(
        "-l",
        "--logmlflow",
        choices=["none", "metrics", "all"],
        default="none",
        help=(
            "MLFlow logging mode: 'none' for no logging, 'metrics' for "
            "metrics only, or 'all' for metrics and model. For remote "
            "logging, MLFLOW_TRACKING_URI must be set."
        ),
    )

    parser.add_argument(
        "-C",
        "--cv",
        type=int,
        default=0,
        help=(
            "Number of folds for cross-validation when fitting models. "
            "Set to 0 to disable cross-validation."
        ),
        metavar="FOLDS",
    )

    return parser.parse_args()


def main():
    options = parse_args()

    if options.cv < 0:
        raise ValueError("--cv must be >= 0")

    if options.outputfolder is None:
        options.outputfolder = str(Path(script_path, "rf_models"))

    os.makedirs(options.outputfolder, exist_ok=True)

    logging.info("Output folder will be %s", options.outputfolder)

    if options.inputfolder is None:
        options.inputfolder = str(Path(options.dbfolder, "rf_input_data"))

    logging.info(
        "Folder with RF input data will be %s",
        options.inputfolder,
    )

    # Whether or not to only generate new input data (no training)
    only_regenerate = options.rfmodels is None

    dic_models = {}

    if not only_regenerate:
        if "default" in options.rfmodels:
            logging.info(
                'Found at least one "default" alias in models %s, '
                "assuming they are all aliases",
                options.rfmodels,
            )

            aliases = [x.strip() for x in options.rfmodels.split(",")]

            for opt in aliases:
                if opt == "dualpol_default":
                    dic_models["RF_dualpol"] = [
                        "RADAR",
                        "zh_VISIB_mean",
                        "zv_VISIB_mean",
                        "KDP_mean",
                        "RHOHV_mean",
                        "SW_mean",
                        "T",
                        "HEIGHT",
                        "VISIB_mean",
                    ]

                elif opt == "hpol_default":
                    dic_models["RF_hpol"] = [
                        "RADAR",
                        "zh_VISIB_mean",
                        "SW_mean",
                        "T",
                        "HEIGHT",
                        "VISIB_mean",
                    ]

                elif opt == "vpol_default":
                    dic_models["RF_vpol"] = [
                        "RADAR",
                        "zv_VISIB_mean",
                        "SW_mean",
                        "T",
                        "HEIGHT",
                        "VISIB_mean",
                    ]

                elif opt == "RFO_default":
                    dic_models["RFO"] = [
                        "RADAR",
                        "zh_VISIB_mean",
                        "zv_VISIB_mean",
                        "KDP_mean",
                        "RHOHV_mean",
                        "SW_mean",
                        "ISO0_HEIGHT",
                        "HEIGHT",
                        "VISIB_mean",
                    ]

                elif opt == "RFO_default_hpol":
                    dic_models["RFO_hpol"] = [
                        "RADAR",
                        "zh_VISIB_mean",
                        "SW_mean",
                        "ISO0_HEIGHT",
                        "HEIGHT",
                        "VISIB_mean",
                    ]

                elif opt == "RFO_default_vpol":
                    dic_models["RFO_vpol"] = [
                        "RADAR",
                        "zv_VISIB_mean",
                        "SW_mean",
                        "ISO0_HEIGHT",
                        "HEIGHT",
                        "VISIB_mean",
                    ]

                else:
                    raise ValueError(f"Unknown RF model alias: {opt}")

        else:
            try:
                dic_models = json.loads(options.rfmodels)
            except json.JSONDecodeError as exc:
                raise ValueError("Could not parse --rfmodels as JSON") from exc

    options.rfmodels = dic_models

    if only_regenerate:
        logging.info(
            "No model was given; only the input parquet files " "will be regenerated"
        )

    if options.config is None:
        options.config = str(Path(script_path, "default_config.yml"))

    logging.info("Starting Random Forest training, leave the script running!")

    logging.info(
        "Assuming that input data for RF training is in folder %s",
        options.inputfolder,
    )
    logging.info(
        "If not available in this folder, they will be computed " "and stored there"
    )

    rf = RFTraining(
        options.dbfolder,
        options.inputfolder,
        options.generate_inputs,
        options.logmlflow,
    )

    if not only_regenerate:
        rf.fit_models(
            options.config,
            options.rfmodels,
            options.start,
            options.end,
            cv=options.cv,
        )


if __name__ == "__main__":
    main()
