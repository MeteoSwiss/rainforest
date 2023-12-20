import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from pytz import utc

import logging

from rainforest.common.jretrievedwh import jretrievedwh_to_pandas
from rainforest.common.object_storage import ObjectStorage
ObjStorage = ObjectStorage()

# This function adds new data to a file, with/without overwriting already present data
def append_to_file(fname, df, overwrite=False):
    if not os.path.exists(fname):
        df_merged = df
    else:
        df_old = pd.read_csv(fname)
        if not overwrite:
            not_in_old = ~df['TIMESTAMP'].isin(df_old['TIMESTAMP'])
            df_merged = pd.concat([df_old, df.loc[not_in_old]], ignore_index=True)
        else:
            not_in_new = ~df_old['TIMESTAMP'].isin(df['TIMESTAMP'])
            df_merged = pd.concat([df_old.loc[not_in_new], df], ignore_index=True)
        df_merged = df_merged.sort_values(by='TIMESTAMP')
    return df_merged

# Parsing command line arguments

args = sys.argv[1:]

t0_str = args[0] # t0
tend_str = args[1] # t1
threshold = float(args[2]) # precip threshold
stations = args[3].split(',') # station list
variables = args[4].split(',') # variable list
output_folder = args[5] 
missing_value = float(args[6]) # how to replace missing value
overwrite = bool(int(args[7])) # whether to replace existing data


# Get directory of data_stations.csv, i.e rainforest/common/constants
path = os.environ.get("RAINFOREST_DATAPATH")
station_info_path = ObjStorage.check_file(os.path.join(path, 'references', 'metadata', 'data_stations.csv'))
station_info = pd.read_csv(station_info_path, sep=';', encoding='latin-1')

for station in stations:
    print(f"Retrieving station {station}")
    is_valid = True
    variables_var = variables.copy()
    if not is_valid:
        continue

    try:
        data = jretrievedwh_to_pandas(['-s', 'surface', '-i', f'nat_abbr,{station}',
            '-t', f'{t0_str},{tend_str}', '-n', f"{','.join(variables_var)}"])

        # Assume transfer function (catch efficiency) is possible
        CE = True
    
        # Fill up missing columns
        for variable in variables:
            if variable not in data.columns:
                data[variable] = missing_value
                # Add a tag for CatchEfficiency
                if variable == 'fkl010z0':
                    CE = False
                    data['rre150z0_adj'] = missing_value
                    logging.info(f"Transfer function not applicable for {station}")
                if variable == 'rre150z0':
                    CE = False

        if CE:
            data['rre150z0_adj'] = data['rre150z0']
            try:
                logging.info(f"Adjustment of gauge measurement {station}")
                # Add Kochendorfer Equation (see https://hess.copernicus.org/articles/21/3525/2017/hess-21-3525-2017.pdf)
                wind = data['fkl010z0']
                wind[wind > 9] = 9.
                # Mixed precipitation (-2<=tair<=2)
                a_mixed = 0.624
                b_mixed = 0.185
                c_mixed = 0.364
                CEmixed_KD4 = a_mixed * np.exp(-b_mixed * wind) + c_mixed
                index_mixed = np.logical_and((data['tre200s0'] >= -2), (data['tre200s0'] <= 2)) 
                index_mixed = np.logical_and(index_mixed, ~data['tre200s0'].isna())
                data.loc[index_mixed, 'rre150z0_adj'] = data.loc[index_mixed, 'rre150z0'] / CEmixed_KD4[index_mixed]
                # solid precipitation (<-2)
                a_solid = 0.865
                b_solid = 0.298
                c_solid = 0.225
                CEsolid_KD4 = a_solid * np.exp(-b_solid * wind) + c_solid
                index_solid = np.logical_and(data['tre200s0'] < -2, ~data['tre200s0'].isna())
                data.loc[index_solid, 'rre150z0_adj'] = data.loc[index_solid, 'rre150z0'] / CEsolid_KD4[index_solid]
                data['rre150z0_adj'] = np.round(data['rre150z0_adj'], decimals=3)
            except Exception as e:
                logging.info(e)
                logging.info(f"Catch efficiency failed for {station}")
        else:
            data['rre150z0_adj'] = missing_value

        # Time of beginning of measurement
        tstamps = [(t -  pd.Timedelta(minutes=5)) for t in data['datetime']]
        stahours_all = [str(d) + '_' + t.strftime('%Y%m%d%H') for d,t in zip(data['station'], tstamps)]
        data['stahours'] = stahours_all

        # Get all wet hours
        rain_h = data.groupby('stahours')['rre150z0'].sum()
        hours_wet = rain_h >= threshold
        stahours_wet = rain_h[hours_wet].index

        zhours_wet = np.isin(stahours_all, stahours_wet)
        zdata_wet = data.loc[zhours_wet, ['datetime'] + variables]
        
        name_out = os.path.join(output_folder, f"{station}.csv.gz")
        # date format must be changed to Linux timestamp
        zdata_wet['datetime'] = [int(t.timestamp()) for t in zdata_wet['datetime']]
        zdata_wet.rename(columns={'datetime': 'timestamp'}, inplace=True)

        # Finally since the R format of hdf5 is not compatible with dask and pandas
        # I choose to stay with csv
        zdata_wet = zdata_wet.fillna(missing_value)

        # Add station info
        zdata_wet.insert(0, 'station', station)

        # Reorder if needed to always get same column order
        right_order = ['station', 'timestamp'] + variables
        zdata_wet = zdata_wet[right_order]
        zdata_wet.columns = map(str.upper, zdata_wet.columns)

        zdata_wet = append_to_file(name_out, zdata_wet, overwrite)
        zdata_wet.to_csv(name_out, sep=',', compression='gzip', index=False)
    except Exception as e:
        logging.info(e)
        logging.info(f"Data not available for {station}")

