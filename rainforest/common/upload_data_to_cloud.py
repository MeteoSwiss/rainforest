import boto3 
import readline
import glob
from pathlib import Path

def rlinput(prompt, prefill=''):
   readline.set_startup_hook(lambda: readline.insert_text(prefill))
   try:
      return input(prompt)  # or raw_input in Python 2
   finally:
      readline.set_startup_hook()

def ask_credentials():
    # Ask credentials
    linode_obj_config = {}
    linode_obj_config["endpoint_url"] = f'https://eu-central-1.linodeobjects.com'
    linode_obj_config["aws_access_key_id"] = input('AWS Access key ID: ')
    linode_obj_config["aws_secret_access_key"] = input('AWS Secret Access key: ')

    print(linode_obj_config)
    return linode_obj_config

def clean_bucket(linode_obj_config):
    s3 = boto3.resource('s3',
                            region_name = 'eu-central-1',
                            endpoint_url = linode_obj_config["endpoint_url"],
                            aws_access_key_id = linode_obj_config["aws_access_key_id"],
                            aws_secret_access_key = linode_obj_config["aws_secret_access_key"])
    bucket = s3.Bucket('rainforest')
    bucket.objects.all().delete()

if __name__ == '__main__':
    print('This script will upload the data and rf_models directories of rainforest to a cloud storage service')
    print('These directories contain large files that cannot be packaged with Pypi or conda')

    linode_obj_config = ask_credentials()

    datadir = rlinput("Enter directory of rainforest data storage: ", '/store/msrad/radar/rainforest/rainforest/common/data')
    rfdir = rlinput("Enter directory of rainforest rf_models storage: ", '/store/msrad/radar/rainforest/rainforest/ml/rf_models')

    # Create client
    client = boto3.client("s3", **linode_obj_config)

    clean_bucket(linode_obj_config)

    print('Uploading data storage...')
    data_storage_files = glob.glob(str(Path(datadir, '**', '*.*')), recursive=True)
    for i, f in enumerate(data_storage_files):
        print("{:d}/{:d} files".format(i, len(data_storage_files)))
        key = 'data/' + f[len(datadir)+1:]
        client.upload_file(
            Bucket='rainforest',
            Key=key,
            Filename=f)

    print('Uploading rf_models storage...')
    rfmodels_files = glob.glob(str(Path(rfdir, '**', '*.*')), recursive=True)
    for i, f in enumerate(rfmodels_files):
        print("{:d}/{:d} files".format(i, len(rfmodels_files)))
        key = 'rfmodels/' + f[len(rfdir)+1:]
        client.upload_file(
            Bucket='rainforest',
            Key=key,
            Filename=f)
    print('Done')
