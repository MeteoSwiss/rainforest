import pandas as pd
import os
from pathlib import Path

from rainforest.database.retrieve_radar_data import Updater

def test_retrieve_radar_data():
    cwd = os.path.dirname(os.getenv('PYTEST_CURRENT_TEST')) + '/'

    cf = str(Path(cwd, 'test_config.yml'))
    tf = str(Path(cwd,'test_task_file.txt'))
    u = Updater(tf, cf, cwd)

    u.process_all_timesteps()
    u.final_cleanup()
  
    ref_table = str(Path(cwd, 'reference_test_output.parquet'))
    new_table = str(Path(cwd, '20191019.parquet'))

    ref_table_df = pd.read_parquet(ref_table)
    new_table_df = pd.read_parquet(new_table)
    os.remove(new_table)

    pd.testing.assert_frame_equal(ref_table_df, new_table_df, check_dtype = False,
    check_less_precise = 3)

