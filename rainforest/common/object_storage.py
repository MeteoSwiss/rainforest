# Global imports
import boto3 
import readline
import glob
from pathlib import Path
import os

###############
# S3 ACCESS
###############
AWS_ACCESS_KEY_ID = "78FXN2YBOXTRQS8938Z7"
ENDPOINT_URL = f'https://eu-central-1.linodeobjects.com'

class ObjectStorage(object):
    def __init__(self, aws_key = None):
        """
        Creates an ObjectStorage instance
        """
        self.aws_defined = True
        if aws_key == None:
            if 'AWS_KEY' in os.environ:
                aws_key = os.environ['AWS_KEY']
            else:
                print('No AWS_KEY environment variable was found, you will not be able to download/upload additional data from the cloud!"')
                self.aws_defined = False
        
        if self.aws_defined:
            self.linode_obj_config = {
            "aws_access_key_id": AWS_ACCESS_KEY_ID,
            "endpoint_url": ENDPOINT_URL,
            'aws_secret_access_key': aws_key}

            self.client = boto3.client("s3", **self.linode_obj_config)
        
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
                raise FileNotFoundError('File {:s} not found and AWS_KEY env variable not defined: retrieval from cloud IMPOSSIBLE.'.format(filename))
            print("File was not found, retrieving it from Object Storage")
            key = os.path.basename(filename)
            bpath = os.path.dirname(filename)
            self.download_file(key, bpath)
        return filename

    def list_files(self, bucket = 'rainforest'):
        """
        Lists all files in a given bucket

        Parameters
        ----------
        bucket : str
            Name of the bucket to clean
        """
        objects = self.client.list_objects_v2(Bucket = bucket)

        for object in objects['Contents']:
            print(object['Key'])

    def clean_bucket(self, bucket = 'rainforest'):
        """
        Cleans a bucket on the cloud S3, will DELETE all data!

        Parameters
        ----------
        bucket : str
            Name of the bucket to clean
        """
        objects = self.client.list_objects_v2(Bucket = bucket)
        print('Bucket contains {:d} objects'.format(len(objects)))
        userinput = input("Are you sure wou want to delete all content from the bucket y/n? ")
        if userinput == 'y':
            for object in objects['Contents']:
                self.client.delete_object(Bucket = bucket, key = object['Key'])
    
    def delete_file(self, key, bucket = 'rainforest'):
        """
        Deletes a file from a bucket

        Parameters
        ----------
        key : str
            Name of the object in the S3 storage
        bucket: str
            Name of the bucket where the file is stored
        """
        self.client.delete_object(Bucket = bucket, Key = key)
 
    def download_file(self, key, bpath, bucket = 'rainforest'):
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
            Bucket=bucket,
            Key = key,
            Filename = str(Path(bpath, key)))
        
    def upload_file(self, filename, bucket = 'rainforest'):
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
                Filename=filename)

    def upload_all_files(self, data_folder = None, bucket = 'rainforest'):
        """
        Uploads all files within a given folder

        Parameters
        ----------
        data_folder : str
            Directory from where to upload all content. The names of the files on the cloud will be their basename (no data structure is kept)
        bucket : str
            Name of the bucket where to upload the files

        """

        print('This script will upload the data and rf_models directories of rainforest to a cloud storage service')
        print('These directories contain large files that cannot be packaged with Pypi or conda')

        if not data_folder:
            print('Using default data folder:')
            print(os.environ['RAINFOREST_DATAPATH'])
            data_folder = os.environ['RAINFOREST_DATAPATH']

        datadir = Path(data_folder, 'references')
        rfdir = Path(data_folder, 'rf_models')

        self.clean_bucket(bucket = bucket)

        print('Uploading reference storage...')
        data_storage_files = glob.glob(str(Path(datadir, '**', '*.*')), recursive=True)
        rfmodels_files = glob.glob(str(Path(rfdir, '**', '*.*')), recursive=True)

        for i, f in enumerate(data_storage_files):
            print("{:d}/{:d} files".format(i, len(data_storage_files)))
            self.upload_file(filename)            

        print('Uploading rf_models storage...')
        rfmodels_files = glob.glob(str(Path(rfdir, '**', '*.*')), recursive=True)
        for i, f in enumerate(rfmodels_files):
            print("{:d}/{:d} files".format(i, len(rfmodels_files)))
            self.upload_file(filename)
        print('Done')
