# Global imports
import boto3
import logging
import os
import glob
from optparse import OptionParser

# Local imports
from rainforest.common.object_storage import ObjectStorage

def main():
    parser = OptionParser()


    parser.add_option("-b", "--bucket", dest = "bucket", type = str,
                      default = "rainforest",
                      help="Bucket on the s3 storage, should be 'rainforest'",
                      metavar="BUCKET")

    parser.add_option("-a", "--action", dest = "action", type = str,
                      help="Action to perform: either 'list', upload', 'download', 'rsync', or 'delete'",
                      metavar="NAME")

    parser.add_option("-n", "--name", dest = "name", type = str,
                      help="Name of the file to download, upload or delete, wildcards are accepted only if action == 'upload'",
                      metavar="NAME")

    parser.add_option("-o", "--outputfolder", dest = "outputfolder", type = str,
                      default = './',
                      help="Only used if action = 'download', defines the directory where the downloaded file is stored, default is current directory.",
                      metavar="OUTPUT")

    parser.add_option("-k", "--key", dest = "key", type = str,
                      default = None,
                      help="The AWS API key to use to access the S3 cloud, only needed if the env variable AWS_KEY is not defined.",
                      metavar="KEY")

    (options, args) = parser.parse_args()

    if options.action not in ['delete', 'download', 'upload', 'list']:
        raise ValueError("Invalid action, should be either 'list', 'upload', 'download' or 'delete'")

    if options.action == 'download':
        logging.info('Output folder will be {:s}'.format(options.outputfolder))
                
    objstorage = ObjectStorage(aws_key = options.key)
    
    if options.action == 'download':
        logging.info('Downloading file {:s} to directory {:s}'.format(options.name, options.outputfolder))
        objstorage.download_file(options.name, options.outputfolder, options.bucket)
    elif options.action == 'upload':
        files = glob.glob(options.name)
        for f in files:
            logging.info('Uploading file {:s}'.format(f))
            objstorage.upload_file(f, options.bucket)
    elif options.action == 'delete':
        logging.info('Deleting file {:s}'.format(options.name))
        objstorage.delete_file(options.name, options.bucket)
    elif options.action == 'list':
        logging.info('Listing files from bucket {:s}'.format(options.bucket))
        listfiles = objstorage.list_files(options.bucket)
        print(listfiles)
    elif options.action == 'rsync':
        logging.info('Rsync of local files with bucket {:s}'.format(options.bucket))
        objstorage.rsync_cloud(bucket = bucket)
