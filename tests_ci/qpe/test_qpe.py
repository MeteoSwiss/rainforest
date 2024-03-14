
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
    filename = {}
    filename['RF_dualpol'] = 'RF_dualpol_BETA_-0.5_BC_spline.p'
    filename['RFO'] = 'RFO_BETA_-0.5_BC_spline_trained_2016_2019.p'

    names = {}
    names['RF_dualpol'] = 'RFQ'
    names['RFO'] = 'RFO'

    t0 = datetime.datetime(2022,9,28,5,10)
    t1 = datetime.datetime(2022,9,28,5,10)
    tstr = '%y%j%H%M'

    for model in filename.keys():
        
        models = {}
        models[model] = read_rf(filename[model])
        print(str(Path(cwd, 'test_config.yml')))        
        qpeproc = QPEProcessor(str(Path(cwd, 'test_config.yml')), models)
        qpeproc.compute(cwd, t0,t1, basename = '{}{}'.format(names[model], tstr), test_mode = True)
        qpe = read_cart(str(Path(cwd, model, datetime.datetime.strftime(t1, '{}{}.h5'.format(names[model], tstr)))))
        qpe_field = qpe.data

        # Assertions
        # Data
        assert 'radar_estimated_rain_rate' in qpe.fields
        assert qpe.fields['radar_estimated_rain_rate']['data'].shape == (1,640, 710)
        assert len(np.unique(qpe.fields['radar_estimated_rain_rate']['data'])) > 2

        # Time
        assert qpe.time['units'] == 'seconds since 2022-09-28T05:05:00Z'
        assert qpe.time['data'] == [0, 300]

        shutil.rmtree(str(Path(cwd, model)))
