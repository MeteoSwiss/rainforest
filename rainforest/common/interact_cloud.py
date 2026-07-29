# Global imports
import glob
import logging
from optparse import OptionParser

# Local imports
from rainforest.common.object_storage import ObjectStorage


def main():
    parser = OptionParser()

    parser.add_option(
        "-b",
        "--bucket",
        dest="bucket",
        type=str,
        default="rainforest",
        help="Bucket on the s3 storage, should be 'rainforest'",
        metavar="BUCKET",
    )

    parser.add_option(
        "-a",
        "--action",
        dest="action",
        type=str,
        help="Action to perform: either 'list', upload', 'download', 'rsync', or 'delete'",
        metavar="NAME",
    )

    parser.add_option(
        "-n",
        "--name",
        dest="name",
        type=str,
        help="Name of the file to download, upload or delete, wildcards are accepted only if action == 'upload'",
        metavar="NAME",
    )

    parser.add_option(
        "-o",
        "--outputfolder",
        dest="outputfolder",
        type=str,
        default="./",
        help="Only used if action = 'download', defines the directory where the downloaded file is stored, default is current directory.",
        metavar="OUTPUT",
    )

    parser.add_option(
        "-k",
        "--key",
        dest="key",
        type=str,
        default=None,
        help="The AWS API key to use to access the S3 cloud, only needed if the env variable AWS_KEY is not defined.",
        metavar="KEY",
    )

    (options, _args) = parser.parse_args()

    if options.action not in ["delete", "download", "upload", "list"]:
        raise ValueError(
            "Invalid action, should be either 'list', 'upload', 'download' or 'delete'"
        )

    if options.action == "download":
        logging.info(f"Output folder will be {options.outputfolder:s}")

    objstorage = ObjectStorage(aws_key=options.key)

    if options.action == "download":
        logging.info(
            f"Downloading file {options.name:s} to directory {options.outputfolder:s}"
        )
        objstorage.download_file(options.name, options.outputfolder, options.bucket)
    elif options.action == "upload":
        files = glob.glob(options.name)
        for f in files:
            logging.info(f"Uploading file {f:s}")
            objstorage.upload_file(f, options.bucket)
    elif options.action == "delete":
        logging.info(f"Deleting file {options.name:s}")
        objstorage.delete_file(options.name, options.bucket)
    elif options.action == "list":
        logging.info(f"Listing files from bucket {options.bucket:s}")
        listfiles = objstorage.list_files(options.bucket)
        print(listfiles)
    elif options.action == "rsync":
        logging.info(f"Rsync of local files with bucket {options.bucket:s}")
        objstorage.rsync_cloud(bucket=options.bucket)
