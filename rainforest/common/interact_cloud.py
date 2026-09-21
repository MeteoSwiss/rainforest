#!/usr/bin/env python3
"""
Command line interface for Rainforest S3 object storage.
"""

# Global imports
import argparse
import glob
import logging

# Local imports
from rainforest.common.object_storage import ObjectStorage


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manage files on the Rainforest S3 object storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        default="rainforest",
        help="Bucket on the S3 storage.",
        metavar="BUCKET",
    )

    parser.add_argument(
        "-a",
        "--action",
        choices=["list", "upload", "download", "delete", "rsync"],
        required=True,
        help="Action to perform.",
        metavar="ACTION",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help=(
            "Name of the file to download, upload, or delete. "
            "Wildcards are accepted for uploads."
        ),
        metavar="NAME",
    )

    parser.add_argument(
        "-o",
        "--outputfolder",
        type=str,
        default="./",
        help=(
            "Directory where downloaded files are stored. "
            "Only used with --action download."
        ),
        metavar="OUTPUT",
    )

    parser.add_argument(
        "-k",
        "--key",
        type=str,
        default=None,
        help=(
            "AWS API key used to access the S3 storage. "
            "Only needed if the AWS_KEY environment variable is not defined."
        ),
        metavar="KEY",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --name is required for these actions
    if args.action in ["download", "upload", "delete"] and args.name is None:
        raise ValueError(f"--name/-n is required when action is '{args.action}'")

    objstorage = ObjectStorage(aws_key=args.key)

    if args.action == "download":
        logging.info(
            "Downloading file %s to directory %s",
            args.name,
            args.outputfolder,
        )

        objstorage.download_file(
            args.name,
            args.outputfolder,
            args.bucket,
        )

    elif args.action == "upload":
        files = glob.glob(args.name)

        if not files:
            logging.warning(
                "No files found matching pattern: %s",
                args.name,
            )

        for filename in files:
            logging.info("Uploading file %s", filename)

            objstorage.upload_file(
                filename,
                args.bucket,
            )

    elif args.action == "delete":
        logging.info("Deleting file %s", args.name)

        objstorage.delete_file(
            args.name,
            args.bucket,
        )

    elif args.action == "list":
        logging.info(
            "Listing files from bucket %s",
            args.bucket,
        )

        files = objstorage.list_files(args.bucket)

        for filename in files:
            print(filename)

    elif args.action == "rsync":
        logging.info(
            "Rsync of local files with bucket %s",
            args.bucket,
        )

        objstorage.rsync_cloud(
            bucket=args.bucket,
        )


if __name__ == "__main__":
    main()
