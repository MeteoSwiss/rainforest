import logging
import os
import sys

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from rainforest.common.jretrievedwh import jretrievedwh_to_pandas
from rainforest.common.object_storage import ObjectStorage

ObjStorage = ObjectStorage()


def month_chunks(t0, t1):
    """
    Split [t0, t1] into consecutive 1-month chunks.
    Returns list of (start, end) with end being the final bound.
    """
    chunks = []
    cur = t0
    while cur < t1:
        nxt = min(cur + relativedelta(months=1), t1)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


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

t0_str = args[0]  # t0
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

t0 = pd.to_datetime(t0_str)
t1 = pd.to_datetime(tend_str)

for station in stations:
    print(f"Retrieving station {station}")
    is_valid = True
    variables_var = variables.copy()
    if not is_valid:
        continue

    # collect per-month results here
    zdata_wet_all = []

    try:
        for tt0, tt1 in month_chunks(t0, t1):
            tt0_str = tt0.strftime("%Y%m%d%H%M")
            tt1_str = tt1.strftime("%Y%m%d%H%M")

            data = jretrievedwh_to_pandas(
                [
                    "-s",
                    "surface",
                    "-i",
                    f"nat_abbr,{station}",
                    "-t",
                    f"{tt0_str},{tt1_str}",
                    "-n",
                    f"{','.join(variables_var)}",
                ]
            )
            # Assume transfer function (catch efficiency) is possible
            CE = True

            # Fill up missing columns
            for variable in variables:
                if variable not in data.columns:
                    data[variable] = missing_value
                    # Add a tag for CatchEfficiency
                    if variable == "fkl010z0":
                        CE = False
                        data["rre150z0_adj"] = missing_value
                        logging.info(f"Transfer function not applicable for {station}")
                    if variable == "rre150z0":
                        CE = False

            if CE:
                data["rre150z0_adj"] = data["rre150z0"]
                try:
                    logging.info(f"Adjustment of gauge measurement {station}")
                    wind = data["fkl010z0"].copy()
                    wind[wind > 9] = 9.0

                    # Mixed precipitation (-2<=tair<=2)
                    a_mixed, b_mixed, c_mixed = 0.624, 0.185, 0.364
                    CEmixed_KD4 = a_mixed * np.exp(-b_mixed * wind) + c_mixed
                    index_mixed = (data["tre200s0"] >= -2) & (data["tre200s0"] <= 2)
                    index_mixed = index_mixed & (~data["tre200s0"].isna())
                    data.loc[index_mixed, "rre150z0_adj"] = (
                        data.loc[index_mixed, "rre150z0"] / CEmixed_KD4[index_mixed]
                    )

                    # Solid precipitation (<-2)
                    a_solid, b_solid, c_solid = 0.865, 0.298, 0.225
                    CEsolid_KD4 = a_solid * np.exp(-b_solid * wind) + c_solid
                    index_solid = (data["tre200s0"] < -2) & (~data["tre200s0"].isna())
                    data.loc[index_solid, "rre150z0_adj"] = (
                        data.loc[index_solid, "rre150z0"] / CEsolid_KD4[index_solid]
                    )

                    data["rre150z0_adj"] = np.round(data["rre150z0_adj"], decimals=3)
                except Exception as e:
                    logging.info(e)
                    logging.info(f"Catch efficiency failed for {station}")
            else:
                data["rre150z0_adj"] = missing_value

            # Time of beginning of measurement
            tstamps = [(t - pd.Timedelta(minutes=5)) for t in data["datetime"]]
            stahours_all = [
                str(d) + "_" + t.strftime("%Y%m%d%H")
                for d, t in zip(data["station"], tstamps)
            ]
            data["stahours"] = stahours_all

            # Get all wet hours
            rain_h = data.groupby("stahours")["rre150z0"].sum()
            hours_wet = rain_h >= threshold
            stahours_wet = rain_h[hours_wet].index

            zhours_wet = np.isin(stahours_all, stahours_wet)
            zdata_wet = data.loc[zhours_wet, ["datetime"] + variables].copy()

            # date format must be changed to Linux timestamp
            zdata_wet["datetime"] = [int(t.timestamp()) for t in zdata_wet["datetime"]]
            zdata_wet.rename(columns={"datetime": "timestamp"}, inplace=True)

            zdata_wet = zdata_wet.fillna(missing_value)
            zdata_wet.insert(0, "station", station)

            right_order = ["station", "timestamp"] + variables
            zdata_wet = zdata_wet[right_order]
            zdata_wet.columns = map(str.upper, zdata_wet.columns)

            zdata_wet_all.append(zdata_wet)

        # After looping over months: merge, drop duplicates, sort, write once
        if len(zdata_wet_all) == 0:
            raise RuntimeError("No data returned across monthly chunks.")

        zdata_wet_all = pd.concat(zdata_wet_all, ignore_index=True)
        # protect against boundary overlap / duplicates
        zdata_wet_all = zdata_wet_all.drop_duplicates(subset=["STATION", "TIMESTAMP"])
        zdata_wet_all = zdata_wet_all.sort_values(["STATION", "TIMESTAMP"])

        name_out = os.path.join(output_folder, f"{station}.csv.gz")
        zdata_wet_all = append_to_file(name_out, zdata_wet_all, overwrite)
        zdata_wet_all.to_csv(name_out, sep=",", compression="gzip", index=False)

    except Exception as e:
        logging.info(e)
        logging.info(f"Data not available for {station}")
