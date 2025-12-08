import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import logging

from rainforest.common.jretrievedwh import jretrievedwh_to_pandas
from rainforest.common.object_storage import ObjectStorage

ObjStorage = ObjectStorage()


def get_dwh_data(station, tstart_str, tend_str, variables, assign_even_to_odd=True):
    """
    Retrieve and preprocess DWH data for a given station and time range.

    This function queries the DWH (Data Warehouse) surface database for a specific station,
    time interval, and list of variables. It returns a DataFrame with datetime indexing and
    optionally fills in NaN values at uneven 5-minute intervals (e.g., 00:05, 00:15, ...)
    using values from the subsequent even timestep (e.g., 00:10, 00:20, ...).

    Parameters
    ----------
    station : str
        Abbreviated station name to query (e.g., 'LUG', 'BER').
    tstart_str : str
        Start datetime in ISO format (e.g., '2025-03-01T00:00').
    tend_str : str
        End datetime in ISO format (e.g., '2025-03-01T12:00').
    variables : list of str
        List of variable codes to retrieve (e.g., ['tre200s0', 'rre005r0']).
    assign_even_to_odd : bool, optional
        Whether to fill NaNs at uneven 5-minute intervals with the value from
        5 minutes later (default is True).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the requested variables, indexed by datetime.
        If `assign_even_to_odd` is True, certain NaNs at odd timestamps are filled.
    """

    data = jretrievedwh_to_pandas(
        [
            "-s",
            "surface",
            "-i",
            f"nat_abbr,{station}",
            "-t",
            f"{tstart_str},{tend_str}",
            "-n",
            f"{','.join(variables)}",
        ]
    )
    data["datetime"] = pd.to_datetime(data["datetime"])

    # Set datetime as index for convenience
    data = data.set_index("datetime")

    if assign_even_to_odd:
        # Define the columns to fill (e.g., all except 'station', 'stahours')
        data_columns = data.columns.difference(["station", "stahours", "rre005r0"])

        # Find rows where the minute % 10 == 5 (i.e., 00:05, 00:15, ...)
        odd_minutes_1 = data.index.minute % 10 == 5

        # Create a DataFrame shifted by -5 minutes (i.e., next row)
        shifted = data[data_columns].shift(-1, freq="5min")

        odd_minutes_2 = shifted.index.minute % 10 == 5
        # Only update values if current value is NaN
        data.loc[odd_minutes_1, data_columns] = data.loc[
            odd_minutes_1, data_columns
        ].combine_first(shifted.loc[odd_minutes_2])

        # Reset index if needed
        data = data.reset_index()
    return data


# This function adds new data to a file, with/without overwriting already present data
def append_to_file(fname, df, overwrite=False):
    if not os.path.exists(fname):
        df_merged = df
    else:
        df_old = pd.read_csv(fname)
        if not overwrite:
            not_in_old = ~df["TIMESTAMP"].isin(df_old["TIMESTAMP"])
            df_merged = pd.concat([df_old, df.loc[not_in_old]], ignore_index=True)
        else:
            not_in_new = ~df_old["TIMESTAMP"].isin(df["TIMESTAMP"])
            df_merged = pd.concat([df_old.loc[not_in_new], df], ignore_index=True)
        df_merged = df_merged.sort_values(by="TIMESTAMP")
    return df_merged


# Parsing command line arguments

args = sys.argv[1:]

tstart_str = args[0]  # t0
tend_str = args[1]  # t1
threshold = float(args[2])  # precip threshold
stations = args[3].split(",")  # station list
variables = args[4].split(",")  # variable list
output_folder = args[5]
missing_value = float(args[6])  # how to replace missing value
overwrite = bool(int(args[7]))  # whether to replace existing data

# Get directory of data_stations.csv, i.e rainforest/common/constants
path = os.environ.get("RAINFOREST_DATAPATH")
station_info_path = ObjStorage.check_file(
    os.path.join(path, "references", "metadata", "data_stations.csv")
)
station_info = pd.read_csv(station_info_path, sep=";", encoding="latin-1")

tstart = datetime.strptime(tstart_str, "%Y%m%d%H%M")
tend = datetime.strptime(tend_str, "%Y%m%d%H%M")

# Create periods of time of at most 6 months
# otherwise queries are too big for jretrieve
periods = []
t0 = tstart

while t0 < tend:
    t1 = min(tend, t0 + timedelta(days=6 * 30))
    periods.append([t0.strftime("%Y%m%d%H%M"), t1.strftime("%Y%m%d%H%M")])
    t0 = t1

for station in stations:
    name_out = os.path.join(output_folder, f"{station}.csv.gz")
    logging.info(f"Retrieving station {station}")
    for t0, t1 in periods:
        logging.info(f"Period {t0} to {t1}")
        is_valid = True
        variables_var = variables.copy()
        if not is_valid:
            continue

        try:
            data = get_dwh_data(station, t0, t1, variables_var)

            # Assume transfer function (catch efficiency) is possible
            CE = True

            # Fill up missing columns
            for variable in variables:
                if variable not in data.columns:
                    data[variable] = missing_value
                    # Add a tag for CatchEfficiency
                    if variable == "fkl010z0":
                        CE = False
                        data["rre005r0_adj"] = missing_value
                        logging.info(f"Transfer function not applicable for {station}")
                    if variable == "rre005r0":
                        CE = False

            if CE:
                data["rre005r0_adj"] = data["rre005r0"]
                try:
                    logging.info(f"Adjustment of gauge measurement {station}")
                    # Add Kochendorfer Equation (see https://hess.copernicus.org/articles/21/3525/2017/hess-21-3525-2017.pdf)
                    wind = data["fkl010z0"]
                    wind[wind > 9] = 9.0
                    # Mixed precipitation (-2<=tair<=2)
                    a_mixed = 0.624
                    b_mixed = 0.185
                    c_mixed = 0.364
                    CEmixed_KD4 = a_mixed * np.exp(-b_mixed * wind) + c_mixed
                    index_mixed = np.logical_and(
                        (data["tre200s0"] >= -2), (data["tre200s0"] <= 2)
                    )
                    index_mixed = np.logical_and(index_mixed, ~data["tre200s0"].isna())
                    data.loc[index_mixed, "rre005r0_adj"] = (
                        data.loc[index_mixed, "rre005r0"] / CEmixed_KD4[index_mixed]
                    )
                    # solid precipitation (<-2)
                    a_solid = 0.865
                    b_solid = 0.298
                    c_solid = 0.225
                    CEsolid_KD4 = a_solid * np.exp(-b_solid * wind) + c_solid
                    index_solid = np.logical_and(
                        data["tre200s0"] < -2, ~data["tre200s0"].isna()
                    )
                    data.loc[index_solid, "rre005r0_adj"] = (
                        data.loc[index_solid, "rre005r0"] / CEsolid_KD4[index_solid]
                    )
                    data["rre005r0_adj"] = np.round(data["rre005r0_adj"], decimals=3)
                except Exception as e:
                    logging.info(e)
                    logging.info(f"Catch efficiency failed for {station}")
            else:
                data["rre005r0_adj"] = missing_value

            # Time of beginning of measurement
            tstamps = [(t - pd.Timedelta(minutes=5)) for t in data["datetime"]]
            stahours_all = [
                str(d) + "_" + t.strftime("%Y%m%d%H")
                for d, t in zip(data["station"], tstamps)
            ]
            data["stahours"] = stahours_all

            # Get all wet hours
            rain_h = data.groupby("stahours")["rre005r0"].sum()
            hours_wet = rain_h >= threshold
            stahours_wet = rain_h[hours_wet].index

            zhours_wet = np.isin(stahours_all, stahours_wet)
            zdata_wet = data.loc[zhours_wet, ["datetime"] + variables]

            # date format must be changed to Linux timestamp
            zdata_wet["datetime"] = [int(t.timestamp()) for t in zdata_wet["datetime"]]
            zdata_wet.rename(columns={"datetime": "timestamp"}, inplace=True)

            zdata_wet = zdata_wet.fillna(missing_value)

            # Add station info
            zdata_wet.insert(0, "station", station)

            # Reorder if needed to always get same column order
            right_order = ["station", "timestamp"] + variables
            zdata_wet = zdata_wet[right_order]
            zdata_wet.columns = map(str.upper, zdata_wet.columns)
            zdata_wet = append_to_file(name_out, zdata_wet, overwrite)
            zdata_wet.to_csv(name_out, sep=",", compression="gzip", index=False)
        except Exception as e:
            logging.info(e)
            logging.info(f"Data not available for {station}")
