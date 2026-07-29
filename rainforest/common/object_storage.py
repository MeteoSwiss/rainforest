# Global imports
import glob
import os
import socket
from pathlib import Path

import boto3
from botocore.config import Config

# These two env variables need to be defined to avoid MissingContentLength error
os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"

###############
# S3 ACCESS
###############
ENDPOINT_URL = "https://s3.pub1.infomaniak.cloud"


class ObjectStorage:
    def __init__(self, aws_key=None, aws_secret=None):
        """
        Creates an ObjectStorage instance
        """
        self.aws_defined = True
        if aws_key is None:
            aws_key = os.getenv("AWS_KEY")
            if aws_key is None:
                print(
                    "No AWS_KEY environment variable was found, you will not be able to download/upload additional data from the cloud!"
                )
                self.aws_defined = False
        if aws_secret is None:
            aws_secret = os.getenv("AWS_SECRET")
            if aws_secret is None:
                print(
                    "No AWS_SECRET environment variable was found, you will not be able to download/upload additional data from the cloud!"
                )
                self.aws_defined = False

        hostname = socket.gethostname()
        if "zuerh" in hostname or "lomrh" in hostname:
            verify = "/etc/ssl/certs/ca-bundle.crt"
        elif "zueub" in hostname or "lomub" in hostname:
            verify = "/etc/ssl/certs/ca-certificates.crt"
        else:
            verify = None

        if self.aws_defined:
            self.linode_obj_config = {
                "aws_access_key_id": aws_key,
                "endpoint_url": ENDPOINT_URL,
                "aws_secret_access_key": aws_secret,
                "config": Config(s3={"use_expect_header": False}),
            }
            self.client = boto3.client("s3", **self.linode_obj_config, verify=verify)

    def check_file(self, filename):
        """
        Checks if a file exists and if not tries to download it from the cloud

        Parameters
        ----------
        filename : str
            Name of the file to retrieve
        """

        if not os.path.exists(filename):
            if not self.aws_defined:
                raise FileNotFoundError(
                    f"File {filename:s} not found and AWS_KEY env variable not defined: retrieval from cloud IMPOSSIBLE."
                )
            print("File was not found, retrieving it from Object Storage")
            key = os.path.basename(filename)
            bpath = os.path.dirname(filename)
            self.download_file(key, bpath)
        return filename

    def list_files(self, bucket="rainforest-ci"):
        """
        Lists all files in a given bucket

        Parameters
        ----------
        bucket : str
            Name of the bucket to clean
        """
        paginator = self.client.get_paginator("list_objects_v2")

        all_keys = []
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                all_keys.append(obj["Key"])

        return all_keys

    def clean_bucket(self, bucket="rainforest-ci"):
        """
        Cleans a bucket on the cloud S3, will DELETE all data!

        Parameters
        ----------
        bucket : str
            Name of the bucket to clean
        """
        objects = self.client.list_objects_v2(Bucket=bucket)
        print(f"Bucket contains {len(objects):d} objects")
        userinput = input(
            "Are you sure wou want to delete all content from the bucket y/n? "
        )
        if userinput == "y":
            for object in objects["Contents"]:
                self.client.delete_object(Bucket=bucket, Key=object["Key"])

    def delete_file(self, key, bucket="rainforest-ci"):
        """
        Deletes a file from a bucket

        Parameters
        ----------
        key : str
            Name of the object in the S3 storage
        bucket: str
            Name of the bucket where the file is stored
        """
        self.client.delete_object(Bucket=bucket, Key=key)

    def download_file(self, key, bpath, bucket="rainforest-ci"):
        """
        Downloads a given file from the cloud S3

        Parameters
        ----------
        key : str
            Name of the file on the cloud
        bpath : str
            Directory where to store the file, its name will be bpath/key
        bucket : str
            Name of the bucket from where to download the file

        """

        if not os.path.exists(bpath):
            os.makedirs(bpath)
        self.client.download_file(
            Bucket=bucket, Key=key, Filename=str(Path(bpath, key))
        )

    def upload_file(self, filename, bucket="rainforest-ci"):
        """
        Uploads a given file to the cloud S3

        Parameters
        ----------
        filename : str
            Full path of the file to upload. Its name on the cloud will be the basename of that path
        bucket : str
            Name of the bucket where to upload the file

        """
        key = os.path.basename(filename)
        self.client.upload_file(
            Bucket=bucket,
            Key=key,
            Filename=filename,
            Config=boto3.s3.transfer.TransferConfig(use_threads=False),
        )

    def rsync_cloud(
        self, rainforest_data_folder=None, bucket="rainforest-ci", overwrite=False
    ):
        """
        Uploads all files within a given folder to the rainforest-ci S3 storage

        Parameters
        ----------
        data_folder : str
            Directory from where to upload all content. The names of the files on the cloud will be their basename (no data structure is kept)
        bucket : str
            Name of the bucket where to upload the files
        overwrite: bool
            If set to true will overwrite files that are already on the cloud
        """

        print(
            "This script will upload the data and rf_models directories of rainforest to a cloud storage service"
        )
        print(
            "These directories contain large files that cannot be packaged with Pypi or conda"
        )

        if not rainforest_data_folder:
            print("Using default data folder:")
            print(os.environ["RAINFOREST_DATAPATH"])
            rainforest_data_folder = os.environ["RAINFOREST_DATAPATH"]

        all_files = glob.glob(rainforest_data_folder + "/**/*.*", recursive=True)
        cloud_files = self.list_files(bucket=bucket)

        all_files_to_upload = []
        for f in all_files:
            if os.path.basename(f) not in cloud_files:
                all_files_to_upload.append(f)

        if not len(all_files_to_upload):
            print("No files need to be sync with cloud, terminating...")
        else:
            for i, f in enumerate(all_files_to_upload):
                print(f"Uploading file {f:s}")
                print(f"Progress: {i:d}/{len(all_files_to_upload):d}")
                self.upload_file(f)

        print("Done")
