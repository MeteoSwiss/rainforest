import os
from pathlib import Path

from rainforest.common.object_storage import ObjectStorage


def test_upload_download():
    cwd = os.path.dirname(os.getenv('PYTEST_CURRENT_TEST')) + '/'

    objSto = ObjectStorage()
    # Create test object
    fname = str(Path(cwd, 'test.txt'))
    with open(fname, 'w') as fh:
        fh.write('Hello World')
    objSto.upload_file(fname)
    os.remove(fname)
    objSto.download_file('test.txt', cwd)
    objSto.delete_file(fname)

    assert os.path.exists(fname)
    assert open(fname,'r').readlines() == ['Hello World']

    os.remove(fname)
