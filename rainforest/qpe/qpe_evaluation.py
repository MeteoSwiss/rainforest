#!/usr/bin/env python3
"""
Command line script to evaluate a set of QPE realizations using gauge data
as reference.

See :ref:`qpe_evaluation`
"""

import argparse
import datetime
import os

from rainforest.qpe.evaluation import evaluation


def parse_datetime(value):
    """Parse a datetime in YYYYMMDDHHMM format."""
    try:
        return datetime.datetime.strptime(value, "%Y%m%d%H%M")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Expected format: YYYYMMDDHHMM"
        ) from exc


def parse_bounds(value):
    """Parse comma-separated precipitation bounds."""
    try:
        bounds = [int(u.strip()) for u in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid bounds '{value}'. Expected comma-separated integers, "
            "e.g. 0,2,10,200"
        ) from exc

    if len(bounds) < 2:
        raise argparse.ArgumentTypeError("At least two bounds must be specified.")

    if bounds != sorted(bounds):
        raise argparse.ArgumentTypeError("Bounds must be in increasing order.")

    return bounds


def parse_models(value):
    """Parse comma-separated model names."""
    return [model.strip() for model in value.split(",") if model.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a set of QPE realizations using gauge data " "as reference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-q",
        "--qpefolder",
        type=str,
        required=True,
        help="Path of the folder where QPE data are stored.",
        metavar="QPEFOLDER",
    )

    parser.add_argument(
        "-g",
        "--gaugepattern",
        type=str,
        default="/store/msrad/radar/radar_database_v2/gauge/*.csv.gz",
        help=(
            "Path pattern, including wildcards, of the gauge data from "
            "the database. Patterns containing wildcards should be quoted."
        ),
        metavar="GAUGEPATTERN",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="outputfolder",
        type=str,
        required=True,
        help="Path of the output folder.",
        metavar="OUTPUT",
    )

    parser.add_argument(
        "-s",
        "--start",
        type=parse_datetime,
        default=None,
        help=(
            "Start time in YYYYMMDDHHMM format. If omitted, the entire "
            "time range available in the QPE folder is used."
        ),
        metavar="START",
    )

    parser.add_argument(
        "-e",
        "--end",
        type=parse_datetime,
        default=None,
        help=(
            "End time in YYYYMMDDHHMM format. If omitted, the entire "
            "time range available in the QPE folder is used."
        ),
        metavar="END",
    )

    parser.add_argument(
        "-b",
        "--b10",
        type=parse_bounds,
        default=[0, 2, 10, 200],
        help=(
            "Precipitation bounds for the 10-minute resolution, given as "
            "comma-separated values. For example, 0,2,10,200 defines "
            "[0,2), [2,10), and [10,200)."
        ),
        metavar="BOUNDS",
    )

    parser.add_argument(
        "-B",
        "--b60",
        type=parse_bounds,
        default=[0, 2, 10, 200],
        help=(
            "Precipitation bounds for the 60-minute resolution, given as "
            "comma-separated values. For example, 0,2,10,200 defines "
            "[0,2), [2,10), and [10,200)."
        ),
        metavar="BOUNDS",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=parse_models,
        default=None,
        help=(
            "Comma-separated list of QPE models (subfolders in qpefolder) "
            "to evaluate. If omitted, all available models are used. "
            "Example: 'dualpol,hpol,RZC'."
        ),
        metavar="MODELS",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if necessary
    os.makedirs(args.outputfolder, exist_ok=True)

    # The evaluation routine expects a trailing slash
    if not args.outputfolder.endswith("/"):
        args.outputfolder += "/"

    evaluation(
        args.qpefolder,
        args.gaugepattern,
        list_models=args.models,
        outputfolder=args.outputfolder,
        t0=args.start,
        t1=args.end,
        bounds10=args.b10,
        bounds60=args.b60,
    )


if __name__ == "__main__":
    main()
