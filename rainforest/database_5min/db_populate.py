#!/usr/bin/env python3
"""
Command line script to add new data to the database.

See :doc:`db_cmd`
"""

# Global imports
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

from rainforest.common import constants

# Local imports
from rainforest.database_5min.database import Database


logging.basicConfig(level=logging.INFO)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)


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
        description="Add new data to the 5-minute radar database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--type",
        dest="type",
        choices=["gauge", "reference", "radar"],
        required=True,
        help="Type of table to populate",
        metavar="TYPE",
    )

    parser.add_argument(
        "-o",
        "--outputfolder",
        type=str,
        default=None,
        help=(
            "Path of the output folder. If not specified, "
            "FOLDER_DATABASE_5min/<type> is used."
        ),
        metavar="OUTPUT",
    )

    parser.add_argument(
        "-s",
        "--start",
        type=parse_datetime,
        default=None,
        help=(
            "Start time in YYYYMMDDHHMM format. Mandatory for type 'gauge'. "
            "For other types it can be inferred from gauge data."
        ),
        metavar="START",
    )

    parser.add_argument(
        "-e",
        "--end",
        type=parse_datetime,
        default=None,
        help=(
            "End time in YYYYMMDDHHMM format. Mandatory for type 'gauge'. "
            "For other types it can be inferred from gauge data."
        ),
        metavar="END",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help=(
            "Path of the config file. If not specified, "
            "default_config.yml in the database module is used."
        ),
        metavar="CONFIG",
    )

    parser.add_argument(
        "-g",
        "--gauge",
        type=str,
        default="/store/msrad/radar/radar_database_5min/gauge/*.csv.gz",
        help=(
            "Gauge data path pattern. Needed only for type 'reference' or "
            "'radar'. Put patterns containing wildcards in quotes."
        ),
        metavar="PATTERN",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # start/end are mandatory when updating gauge data
    if args.type == "gauge" and (args.start is None or args.end is None):
        raise ValueError("Please specify both --start and --end when --type=gauge")

    # Set default output directory
    if args.outputfolder is None:
        args.outputfolder = str(Path(constants.FOLDER_DATABASE_5min, args.type))

    os.makedirs(args.outputfolder, exist_ok=True)

    # Set default config file
    if args.config is None:
        script_path = os.path.dirname(os.path.realpath(__file__))
        args.config = str(Path(script_path, "default_config.yml"))

    dbase = Database(config_file=args.config)

    if args.type != "gauge":
        logging.info("Trying to read gauge table...")
        try:
            dbase.add_tables({"gauge": args.gauge})
        except Exception:
            logging.error("Could not read gauge table, please check --gauge/-g pattern")
            raise

    logging.info("Starting database update, leave the script running!")

    if args.type == "gauge":
        dbase.update_station_data(
            args.start,
            args.end,
            args.outputfolder,
        )

    elif args.type == "reference":
        dbase.update_reference_data(
            "gauge",
            args.outputfolder,
            args.start,
            args.end,
        )

    elif args.type == "radar":
        dbase.update_radar_data(
            "gauge",
            args.outputfolder,
            args.start,
            args.end,
        )


if __name__ == "__main__":
    main()
