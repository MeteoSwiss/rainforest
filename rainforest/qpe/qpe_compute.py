#!/usr/bin/env python3
"""
Command line script for the RandomForest QPE.

See :ref:`qpe_compute`
"""

# Global imports
import argparse
import datetime
import json
import os
import warnings
from pathlib import Path

from rainforest.ml.rfdefinitions import read_rf

# Local imports
from rainforest.qpe.qpe import QPEProcessor


warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_datetime(value):
    """Parse a datetime in YYYYMMDDHHMM format."""
    try:
        return datetime.datetime.strptime(value, "%Y%m%d%H%M")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Expected format: YYYYMMDDHHMM"
        ) from exc


def parse_models(value):
    """Parse the model definition from a JSON string."""
    try:
        models = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid JSON model definition: {exc}"
        ) from exc

    if not isinstance(models, dict):
        raise argparse.ArgumentTypeError(
            "Models must be specified as a JSON dictionary."
        )

    return models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QPE model predictions for a given time range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--start",
        type=parse_datetime,
        required=True,
        help="Start time in YYYYMMDDHHMM format.",
        metavar="START",
    )

    parser.add_argument(
        "-e",
        "--end",
        type=parse_datetime,
        required=True,
        help="End time in YYYYMMDDHHMM format.",
        metavar="END",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="outputfolder",
        type=str,
        default="./",
        help="Path of the output folder.",
        metavar="OUTPUT",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help=(
            "Path of the config file. By default, default_config.yml "
            "in the qpe module is used."
        ),
        metavar="CONFIG",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=parse_models,
        default={"RF_dualpol": "RF_dualpol_BETA_-0.5_BC_spline.p"},
        help=(
            "Models to use as a JSON dictionary. Models must be in "
            "$RAINFOREST_DATAPATH/rf_models/ unless overridden by "
            "--modelpath. Example: "
            '\'{"RF_dualpol": "RF_dualpol_BETA_-0.5_BC_spline.p"}\''
        ),
        metavar="MODELS",
    )

    parser.add_argument(
        "-M",
        "--mlflow-model",
        dest="mlflow_model",
        action="store_true",
        help=(
            "Interpret the values passed to --models as MLflow run IDs "
            "instead of model filenames."
        ),
    )

    parser.add_argument(
        "-p",
        "--modelpath",
        type=str,
        default=None,
        help=(
            "Directory containing the models if they are not stored in "
            "$RAINFOREST_DATAPATH/rf_models/."
        ),
        metavar="MODELPATH",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    parser.add_argument(
        "-r",
        "--rt",
        action="store_true",
        help=(
            "Run in real-time mode, where input data are available "
            "uncompressed, as is usually the case on LOM/ZUE servers."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set default configuration file
    if args.config is None:
        script_path = os.path.dirname(os.path.realpath(__file__))
        args.config = str(Path(script_path, "default_config.yml"))

    # Load RF models
    models = {}

    for name, model in args.models.items():
        if args.mlflow_model:
            models[name] = read_rf(mlflow_runid=model)
        elif args.modelpath is None:
            models[name] = read_rf(model)
        else:
            models[name] = read_rf(
                model,
                filepath=args.modelpath,
            )

    # Create output directory
    os.makedirs(args.outputfolder, exist_ok=True)

    # Initialize and run QPE
    qpe = QPEProcessor(
        args.config,
        models,
        rt=args.rt,
        verbose=args.verbose,
    )

    qpe.compute(
        args.outputfolder,
        args.start,
        args.end,
    )


if __name__ == "__main__":
    main()
