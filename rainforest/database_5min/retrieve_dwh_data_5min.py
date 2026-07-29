import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from rainforest.common.jretrievedwh import jretrievedwh_to_pandas
from rainforest.common.object_storage import ObjectStorage

ObjStorage = ObjectStorage()


def month_chunks(t0: datetime, t1: datetime):
    """Split [t0, t1] into consecutive <=1-month chunks."""
    chunks = []
    cur = t0
    while cur < t1:
        nxt = min(cur + relativedelta(months=1), t1)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def get_dwh_data(station, tstart_str, tend_str, variables, assign_even_to_odd=True):
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
    data = data.set_index("datetime")

    if assign_even_to_odd:
        data_columns = data.columns.difference(["station", "stahours", "rre005r0"])
        odd_minutes_1 = data.index.minute % 10 == 5
        shifted = data[data_columns].shift(-1, freq="5min")
        odd_minutes_2 = shifted.index.minute % 10 == 5
        data.loc[odd_minutes_1, data_columns] = data.loc[
            odd_minutes_1, data_columns
        ].combine_first(shifted.loc[odd_minutes_2])
        data = data.reset_index()

    return data


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


# ---- Parsing command line arguments ----
args = sys.argv[1:]

tstart_str = args[0]  # t0 (YYYYMMDDHHMM)
tend_str = args[1]  # t1 (YYYYMMDDHHMM)
threshold = float(args[2])  # precip threshold
stations = args[3].split(",")
variables = args[4].split(",")
output_folder = args[5]
missing_value = float(args[6])
overwrite = bool(int(args[7]))

# ---- Station metadata ----
path = os.environ.get("RAINFOREST_DATAPATH")
station_info_path = ObjStorage.check_file(
    os.path.join(path, "references", "metadata", "data_stations.csv")
)
station_info = pd.read_csv(station_info_path, sep=";", encoding="latin-1")

tstart = datetime.strptime(tstart_str, "%Y%m%d%H%M")
tend = datetime.strptime(tend_str, "%Y%m%d%H%M")

# ---- NEW: Create 1-month periods (instead of 6 months) ----
periods = [
    (t0.strftime("%Y%m%d%H%M"), t1.strftime("%Y%m%d%H%M"))
    for (t0, t1) in month_chunks(tstart, tend)
]

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

            CE = True

            # Fill up missing columns
            for variable in variables:
                if variable not in data.columns:
                    data[variable] = missing_value
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
                    wind = data["fkl010z0"].copy()
                    wind[wind > 9] = 9.0

                    # Mixed precipitation (-2<=tair<=2)
                    a_mixed, b_mixed, c_mixed = 0.624, 0.185, 0.364
                    CEmixed_KD4 = a_mixed * np.exp(-b_mixed * wind) + c_mixed
                    index_mixed = (data["tre200s0"] >= -2) & (data["tre200s0"] <= 2)
                    index_mixed = index_mixed & (~data["tre200s0"].isna())
                    data.loc[index_mixed, "rre005r0_adj"] = (
                        data.loc[index_mixed, "rre005r0"] / CEmixed_KD4[index_mixed]
                    )

                    # Solid precipitation (<-2)
                    a_solid, b_solid, c_solid = 0.865, 0.298, 0.225
                    CEsolid_KD4 = a_solid * np.exp(-b_solid * wind) + c_solid
                    index_solid = (data["tre200s0"] < -2) & (~data["tre200s0"].isna())
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
            zdata_wet = data.loc[zhours_wet, ["datetime"] + variables].copy()

            # date format must be changed to Linux timestamp
            zdata_wet["datetime"] = [int(t.timestamp()) for t in zdata_wet["datetime"]]
            zdata_wet.rename(columns={"datetime": "timestamp"}, inplace=True)

            zdata_wet = zdata_wet.fillna(missing_value)

            # Add station info
            zdata_wet.insert(0, "station", station)

            # Reorder and uppercase
            right_order = ["station", "timestamp"] + variables
            zdata_wet = zdata_wet[right_order]
            zdata_wet.columns = map(str.upper, zdata_wet.columns)

            zdata_wet = append_to_file(name_out, zdata_wet, overwrite)
            zdata_wet.to_csv(name_out, sep=",", compression="gzip", index=False)

        except Exception as e:
            logging.info(e)
            logging.info(f"Data not available for {station}")
