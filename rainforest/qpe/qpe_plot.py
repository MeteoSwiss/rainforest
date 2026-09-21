#!/usr/bin/env python3
"""
Command line script to display QPE realizations.

See :ref:`qpe_plot`
"""

# Global imports
import argparse
import datetime
import logging
import os
import sys

import matplotlib.pyplot as plt

# Local imports
from ..common.graphics import qpe_plot
from ..common.io_data import read_cart
from ..common.utils import get_qpe_files


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

logging.basicConfig(level=logging.INFO)


def parse_datetime(value):
    """Parse a datetime in YYYYMMDDHHMM format."""
    try:
        return datetime.datetime.strptime(value, "%Y%m%d%H%M")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Expected format: YYYYMMDDHHMM"
        ) from exc


def parse_float_pair(value):
    """Parse a comma-separated pair of floats."""
    try:
        values = [float(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid pair '{value}'. Expected two comma-separated numbers."
        ) from exc

    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected two comma-separated values, got '{value}'."
        )

    return values


def parse_int_pair(value):
    """Parse a comma-separated pair of integers."""
    try:
        values = [int(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid pair '{value}'. Expected two comma-separated integers."
        ) from exc

    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected two comma-separated values, got '{value}'."
        )

    return values


def parse_models(value):
    """Parse comma-separated model names."""
    models = [model.strip() for model in value.split(",") if model.strip()]

    if not models:
        raise argparse.ArgumentTypeError("At least one model must be specified.")

    return models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display QPE realizations as maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--inputfolder",
        type=str,
        required=True,
        help="Path of the input folder.",
        metavar="INPUT",
    )

    parser.add_argument(
        "-o",
        "--outputfolder",
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
            "time range in the input folder is plotted."
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
            "time range in the input folder is plotted."
        ),
        metavar="END",
    )

    parser.add_argument(
        "-S",
        "--shapefile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlay the shapefile of Swiss borders. " "Use --no-shapefile to disable."
        ),
    )

    parser.add_argument(
        "-f",
        "--figsize",
        type=parse_float_pair,
        default=None,
        help=(
            "Figure size as width,height in inches, e.g. 10,5. "
            "If omitted, the size is automatically chosen."
        ),
        metavar="WIDTH,HEIGHT",
    )

    parser.add_argument(
        "-x",
        "--xlim",
        type=parse_float_pair,
        default=[400.0, 900.0],
        help=(
            "Limits in the west-east direction in Swiss coordinates, " "e.g. 400,900."
        ),
        metavar="XMIN,XMAX",
    )

    parser.add_argument(
        "-y",
        "--ylim",
        type=parse_float_pair,
        default=[0.0, 350.0],
        help=(
            "Limits in the south-north direction in Swiss coordinates, " "e.g. 0,350."
        ),
        metavar="YMIN,YMAX",
    )

    parser.add_argument(
        "-c",
        "--cbar",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Orientation of the colorbar.",
    )

    parser.add_argument(
        "-d",
        "--display",
        type=parse_int_pair,
        default=None,
        help=(
            "Arrangement of QPE subplots as rows,columns. "
            "For example, 2,1 puts the fields on two rows and one column. "
            "If omitted, fields are placed in one row."
        ),
        metavar="ROWS,COLUMNS",
    )

    parser.add_argument(
        "-t",
        "--transition",
        type=float,
        default=3.0,
        help=(
            "Precipitation intensity at which the colormap changes "
            "from purple to rainbow."
        ),
        metavar="TRANSITION",
    )

    parser.add_argument(
        "-v",
        "--vmin",
        type=float,
        default=0.04,
        help="Minimum precipitation intensity to display.",
        metavar="VMIN",
    )

    parser.add_argument(
        "-V",
        "--vmax",
        type=float,
        default=120.0,
        help="Maximum precipitation intensity to display.",
        metavar="VMAX",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=parse_models,
        default=None,
        help=(
            "Comma-separated list of models (subfolders in the QPE folder) "
            "to plot. If omitted, all available models are used. "
            "Example: 'dualpol,hpol,RZC'."
        ),
        metavar="MODELS",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outputfolder, exist_ok=True)

    all_files = get_qpe_files(
        args.inputfolder,
        args.start,
        args.end,
        list_models=args.models,
    )

    for t in sorted(all_files):
        logging.info("Processing timestep: %s", t)

        labels = list(all_files[t].keys())

        fields = [read_cart(fname[0]) for fname in all_files[t].values()]

        fig, ax = qpe_plot(
            fields,
            figsize=args.figsize,
            transition=args.transition,
            vmax=args.vmax,
            vmin=args.vmin,
            subplots=args.display,
            cbar_orientation=args.cbar,
            xlim=args.xlim,
            ylim=args.ylim,
        )

        for axis, label in zip(ax, labels):
            axis.set_title(label)

        fig.suptitle(
            str(t),
            y=max(1 - 0.05 * len(fields), 0.9),
            fontsize=14,
            fontweight="bold",
        )

        tstr = t.strftime("%Y%j%H%M")

        output_file = os.path.join(
            args.outputfolder,
            f"qpeplot{tstr}.png",
        )

        plt.savefig(
            output_file,
            bbox_inches="tight",
            dpi=300,
        )

        plt.close(fig)


if __name__ == "__main__":
    main()
