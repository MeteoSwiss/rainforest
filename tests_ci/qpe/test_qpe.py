
import os
from pathlib import Path
import datetime
import numpy as np
import shutil

from rainforest.common.object_storage import ObjectStorage
from rainforest.qpe import QPEProcessor
from rainforest.ml.rfdefinitions import read_rf
from rainforest.common.io_data import read_cart

def test_qpe():
    cwd = os.path.dirname(os.getenv('PYTEST_CURRENT_TEST')) + '/'

    # Get RF model
    models = {}
    models['RF_dualpol'] = read_rf('RF_dualpol_BETA_-0.5_BC_spline.p')

    qpeproc = QPEProcessor(str(Path(cwd, 'test_config.yml')), models)

    t0 = datetime.datetime(2022,9,28,5,10 )
    t1 = datetime.datetime(2022,9,28,5,10 )

    qpeproc.compute('/users/wolfensb/rainforest/tests_ci/qpe/', t0,t1, test_mode = True)
    qpe = read_cart(str(Path(cwd, 'RF_dualpol', 'RFQ222710510.h5')))
    qpe_field = qpe.data

    # Assertions
    assert qpe_field.shape == (640, 710)
    assert len(np.unique(qpe_field)) > 2

    shutil.rmtree(str(Path(cwd, 'RF_dualpol')))