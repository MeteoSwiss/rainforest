from rainforest.common.radarprocessing import Radar
from rainforest.common.utils import split_by_time
from rainforest.common.object_storage import ObjectStorage
from pathlib import Path
import pyart
import os

def test_radarprocessing():
    RADAR_SAMPLES_FOLDER = Path(os.environ['RAINFOREST_DATAPATH'], 'references', 'radar_samples')
    objsto = ObjectStorage()

    bname = 'MLL2005500000U.00'
    sweeps = range(1,6)
    # get all radar files
    radfiles = []
    for sweep in sweeps:
        fname = str(Path(RADAR_SAMPLES_FOLDER, bname + str(sweep)))
        radfiles.append(objsto.check_file(fname))
    statfile = objsto.check_file(str(Path(RADAR_SAMPLES_FOLDER, 'STL2005500000U.xml')))
    hztfile = objsto.check_file(str(Path(RADAR_SAMPLES_FOLDER, 'HZT2005500000L.nc')))
    hztgrid = pyart.io.read_grid(hztfile)

    radar = Radar('L', radfiles, statusfile = statfile,  metranet_reader = 'python')
    radar.snr_mask(3)
    radar.visib_mask(50, 2)
    radar.add_hzt_data(hztgrid.fields['iso0_height']['data'][0])
    dscfg = {}
    dscfg['RMIN']  = 1000.
    dscfg['RMAX']  = 50000.
    dscfg['RCELL'] = 1000.
    dscfg['ZMIN']  = 20.
    dscfg['ZMAX']  = 40.
    dscfg['RWIND'] = 6000.
    radar.compute_kdp(dscfg)

    assert list(radar.radsweeps.keys()) == list(range(1,6))
    assert 'KDP' in radar.radsweeps[1].fields.keys()
    assert 'NH' in radar.radsweeps[1].fields.keys()
    assert 'ZH_VISIB' in radar.radsweeps[1].fields.keys()
    assert 'ISO0_HEIGHT' in radar.radsweeps[1].fields.keys()

