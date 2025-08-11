#%%
import os
import shutil
from pathlib import Path

from rainforest.common.lookup import get_lookup, calc_lookup
from rainforest.common.constants import METSTATIONS

LOOKUP_FOLDER = Path(os.environ['RAINFOREST_DATAPATH'], 'references', 'lookup_data')

def test_read_lookup():
    lut_station = get_lookup('station_to_rad','A')
    assert 'OTL' in lut_station[2].keys()

def test_calc_lookup():
    # Make a backup of lut if exists
    lut_name = str(Path(LOOKUP_FOLDER, 'lut_station_to_qpegrid.p'))
    fail = True
    if os.path.exists(lut_name):
        os.rename(lut_name, lut_name.replace('.p','.bkp'))
    try:
        calc_lookup('station_to_qpegrid')
        if os.path.exists(lut_name):
            fail = False
    except:
        os.rename(lut_name.replace('.p','.bkp'), lut_name)
    
    assert fail == False


# %%
