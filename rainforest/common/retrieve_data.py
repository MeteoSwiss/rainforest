#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to retrieve MeteoSwiss products from the archives

Daniel Wolfensberger, Rebecca Gugerli
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch, rebecca.gugerli@epfl.ch
December 2019, July 2022
"""


import numpy as np
import os
import zipfile
import datetime
import glob
import subprocess
from io import BytesIO
import fnmatch
import re
from warnings import warn

import pandas as pd  # function used in retrieve_hzt_prod
from . import constants
from . import io_data as io  # avoid circular


def _make_timezone_aware(dt, tz=datetime.timezone.utc):
    """
    Makes a naive datetime timezone-aware by setting it to the provided timezone.
    If the datetime is already timezone-aware, it is returned unchanged.

    Parameters:
    dt (datetime): The datetime object to check.
    tz (timezone): The timezone to set if the datetime is naive (default is UTC).

    Returns:
    datetime: The timezone-aware datetime object.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt


def retrieve_hzt_prod(folder_out, start_time, end_time, pattern_type="shell"):
    """Retrieves the preprocessed HZT products from the CSCS repository for a specified
    time range, unzips them and places them in a specified folder

    Parameters
    ----------

    folder_out: str
        directory where to store the unzipped files
    start_time : datetime.datetime instance
        starting time of the time range
    end_time : datetime.datetime instance
        end time of the time range
    pattern_type: either 'shell' or 'regex' (optional)
        use 'shell' for standard shell patterns, which use * as wildcard
        use 'regex' for more advanced regex patterns

    Returns
    -------
    A list containing all the filepaths of the retrieved files

    """
    dt = datetime.timedelta(hours=1)
    delta = end_time - start_time
    if delta.total_seconds() == 0:
        times = [start_time]
    else:
        times = start_time + np.arange(int(delta.total_seconds() / (60 * 60)) + 2) * dt
    dates = []
    for t in times:
        dates.append(datetime.datetime(year=t.year, month=t.month, day=t.day))
    if len(times) == 1 and start_time.hour == 23:  # handle special case of time 23
        dates.append(dates[0] + datetime.timedelta(days=1))  # add tomorrow
    dates = np.unique(dates)

    t0 = _make_timezone_aware(start_time)
    t1 = _make_timezone_aware(end_time)
    all_files = []
    for i, d in enumerate(dates):
        if i == 0:
            start_time = datetime.datetime(
                year=t0.year, month=t0.month, day=t0.day, hour=t0.hour
            )
            # print('*first start time: ', start_time)
        else:
            start_time = datetime.datetime(year=d.year, month=d.month, day=d.day)
            # print('*all other start times', start_time)
        if i == len(dates) - 1:
            end_time = datetime.datetime(
                year=t1.year,
                month=t1.month,
                day=t1.day,
                hour=t1.hour,
            ) + datetime.timedelta(hours=1)
        else:
            end_time = datetime.datetime(year=d.year, month=d.month, day=d.day, hour=23)
            # print('*end_time: ', end_time)
        files = _retrieve_hzt_prod_daily(folder_out, start_time, end_time, pattern_type)

        if files is not None:
            all_files.extend(files)

    return all_files


def retrieve_hzt_RT(tstep):
    """Retrieves the preprocessed HZT products
        A version adapted to real time implementation
        Only used in for the function retrieve_hzt_prod

    Parameters
    ----------

    tstep: datetime
        directory where to store the unzipped files

    Returns
    -------
    A list containing all the filepaths of the retrieved files

    """

    # Get list of available files
    folder_in = constants.FOLDER_ISO0
    content_zip = np.array(
        [
            c
            for c in os.listdir(folder_in)
            if (len(c.split(".")) == 2) and (int(c.split(".")[-1]) >= 800)
        ]
    )

    # HZT files are produced once an hour
    start_time = tstep.replace(minute=0)
    end_time = start_time + datetime.timedelta(hours=1)
    start_time = _make_timezone_aware(start_time)
    end_time = _make_timezone_aware(end_time)

    try:
        # Sort filelist to most recent prediction
        content_filt = np.array([c for c in content_zip if c.endswith("800")])
        times_filt = np.array(
            [
                datetime.datetime.strptime(c[3:12], "%y%j%H%M").replace(
                    tzinfo=datetime.timezone.utc
                )
                + datetime.timedelta(hours=int(c[-2::]))
                for c in content_filt
            ]
        )
        conditions = np.array(
            [np.logical_and((t >= start_time), (t <= end_time)) for t in times_filt]
        )

        content_filt = content_filt[conditions]
        times_filt = times_filt[conditions]
    except (ValueError, TypeError, IndexError):
        print("HZT data does not exist for " + start_time.strftime("%d-%b-%y"))
        files = None
        return

    # Check that an hourly estimate is available
    all_hours = pd.date_range(start=start_time, end=end_time, freq="H")

    if len(all_hours) != len(times_filt):
        content_times = np.array(
            [
                datetime.datetime.strptime(c[3:12], "%y%j%H%M").replace(
                    tzinfo=datetime.timezone.utc
                )
                + datetime.timedelta(hours=int(c[-2::]))
                for c in content_zip
            ]
        )
        # Find time that is missing:
        for hh in all_hours:
            if hh not in times_filt:
                hh_last = np.where(hh == content_times)
                times_filt = np.sort(np.append(times_filt, content_times[hh_last][-1]))
                content_filt = np.sort(
                    np.append(content_filt, content_zip[hh_last][-1])
                )

    # Get a list of all files to retrieve
    conditions = np.array(
        [np.logical_and(t >= start_time, t <= end_time) for t in times_filt]
    )

    if not np.any(conditions):
        msg = """
        No file was found corresponding to this format, verify pattern and product_name
        """
        raise ValueError(msg)

    files = sorted(
        np.array([folder_in + c for c in np.array(content_filt)[conditions]])
    )

    return files


def _retrieve_hzt_prod_daily(folder_out, start_time, end_time, pattern_type="shell"):
    """Retrieves the preprocessed HZT products from the CSCS repository for a day,
        Only used in for the function retrieve_hzt_prod

    Parameters
    ----------

    folder_out: str
        directory where to store the unzipped files
    start_time : datetime.datetime instance
        starting time of the time range
    end_time : datetime.datetime instance
        end time of the time range
    pattern_type: either 'shell' or 'regex' (optional)
        use 'shell' for standard shell patterns, which use * as wildcard
        use 'regex' for more advanced regex patterns

    Returns
    -------
    A list containing all the filepaths of the retrieved files

    """

    folder_out += "/"
    start_time = _make_timezone_aware(start_time)
    end_time = _make_timezone_aware(end_time)

    suffix = str(start_time.year)[-2:] + str(start_time.timetuple().tm_yday).zfill(3)
    folder_in = constants.FOLDER_ISO0 + str(start_time.year) + "/" + suffix + "/"
    name_zipfile = "HZT" + suffix + ".zip"
    try:
        # Get list of files in zipfile
        zipp = zipfile.ZipFile(folder_in + name_zipfile)
        content_zip = np.sort(np.array(zipp.namelist()))

        # Sort filelist to most recent prediction
        content_filt = np.array([c for c in content_zip if c.endswith("800")])
        times_filt = np.array(
            [
                _make_timezone_aware(
                    datetime.datetime.strptime(c[3:12], "%y%j%H%M")
                    + datetime.timedelta(hours=int(c[-2::]))
                )
                for c in content_filt
            ]
        )
        content_filt = content_filt[
            np.where((times_filt >= start_time) & (times_filt <= end_time))
        ]
        times_filt = times_filt[
            np.where((times_filt >= start_time) & (times_filt <= end_time))
        ]
    except (ValueError, TypeError, IndexError):
        print(
            "Zip file with HZT data does not exist for "
            + start_time.strftime("%d-%b-%y")
        )
        files = None
        return
    # Check that an hourly estimate is available
    all_hours = pd.date_range(start=start_time, end=end_time, freq="H")

    if len(all_hours) != len(times_filt):
        content_times = np.array(
            [
                datetime.datetime.strptime(c[3:12], "%y%j%H%M")
                + datetime.timedelta(hours=int(c[-2::]))
                for c in content_zip
            ]
        )
        # Find time that is missing:
        for hh in all_hours:
            if hh not in times_filt:
                hh_last = np.where(hh == content_times)
                times_filt = np.sort(np.append(times_filt, content_times[hh_last][-1]))
                content_filt = np.sort(
                    np.append(content_filt, content_zip[hh_last][-1])
                )

    # Get a list of all files to retrieve
    conditions = np.array(
        [np.logical_and(t >= start_time, t <= end_time) for t in times_filt]
    )

    if not np.any(conditions):
        msg = """
        No file was found corresponding to this format, verify pattern and product_name
        """
        raise ValueError(msg)

    files_to_retrieve = " ".join(content_filt[conditions])

    # Check if files are already unzipped (saves time if they already exist)
    for fi in content_filt[conditions]:
        if os.path.exists(folder_out + fi):
            files_to_retrieve = files_to_retrieve.replace(fi, "")

    # Only unzip if at least one file does not exist
    if len(files_to_retrieve.strip()) > 0:
        print("Unzippping: " + files_to_retrieve)
        cmd = 'unzip -j -o -qq "{:s}" {:s} -d {:s}'.format(
            folder_in + name_zipfile, files_to_retrieve, folder_out
        )
        subprocess.call(cmd, shell=True)

    files = sorted(
        np.array([os.path.join(folder_out, c) for c in content_filt[conditions]])
    )

    return files


def retrieve_prod(
    start_time,
    end_time,
    product_name,
    folder_out=None,
    pattern=None,
    pattern_type="shell",
    sweeps=None,
    hdf5=False,
):
    """Retrieves radar data from the CSCS repository for a specified
    time range, unzips them and places them in a specified folder

    Parameters
    ----------

    start_time : datetime.datetime instance
        starting time of the time range
    end_time : datetime.datetime instance
        end time of the time range
    product_name: str
        name of the product, as stored on CSCS, e.g. RZC, CPCH, MZC, BZC...
    folder_out: str
        directory where to store the unzipped files, if set to None
        will read the file to memory
    pattern: str
        pattern constraint on file names, can be used for products which contain
        multiple filetypes, f.ex CPCH folders contain both rda and gif files,
        if only gifs are wanted : file_type = '*.gif'
    pattern_type: either 'shell' or 'regex' (optional)
        use 'shell' for standard shell patterns, which use * as wildcard
        use 'regex' for more advanced regex patterns
    sweeps: list of int (optional)
        For polar products, specifies which sweeps (elevations) must be
        retrieved, if not specified all available sweeps will be retrieved
    hdf5: bool
        If True will retrieve the hdf5 files for the given product (beware
        hdf5 is not available for all products)

    Returns
    -------
    A list containing all the filepaths of the retrieved files

    """

    if product_name == "ZZW" or product_name == "ZZP":  # no vpr for PPM and WEI
        product_name = "ZZA"

    if folder_out:
        if product_name == "CPC":
            folder_out = os.path.join(folder_out, "CPC")
        if product_name == "CPCH":
            folder_out = os.path.join(folder_out, "CPCH")
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)

    # Check if times are aware or naive
    start_time = _make_timezone_aware(start_time)
    end_time = _make_timezone_aware(end_time)

    dt = datetime.timedelta(minutes=1)
    delta = end_time - start_time
    if delta.total_seconds() == 0:
        times = [start_time]
    else:
        times = start_time + np.arange(int(delta.total_seconds() / (60)) + 1) * dt
    dates = [datetime.datetime(year=t.year, month=t.month, day=t.day) for t in times]
    dates = np.unique(dates)

    t0 = start_time
    t1 = end_time
    all_files = []
    for i, d in enumerate(dates):
        if i == 0:
            start_time = t0
        else:
            start_time = datetime.datetime(
                year=d.year, month=d.month, day=d.day, tzinfo=datetime.timezone.utc
            )
        if i == len(dates) - 1:
            end_time = t1
        else:
            end_time = datetime.datetime(
                year=d.year,
                month=d.month,
                day=d.day,
                hour=23,
                minute=59,
                tzinfo=datetime.timezone.utc,
            )
        files = _retrieve_prod_daily(
            start_time,
            end_time,
            product_name,
            folder_out,
            pattern,
            pattern_type,
            sweeps,
            hdf5,
        )

        all_files.extend(files)
    return all_files


def retrieve_prod_RT(
    time, product_name, pattern=None, pattern_type="shell", sweeps=None
):
    """Adapted function from rainforest.common.retrieve_data
        Here, it reads the data per timestep, and in the real-time
        operation, the radar data is not zipped

    Args:
        time (datetime object): timestamp to extract
        product_name (string): Name of the product to be extracted
        sweeps (list): List of sweeps if not all want to be extracted. Defaults to None.

    Raises:
        ValueError: If no data is found

    Returns:
        dict: dictionary containing with the the file list
    """
    time = _make_timezone_aware(time)

    # Get all files
    folder_radar = constants.FOLDER_RADAR
    folder_in = folder_radar + product_name + "/"

    # Get list of available files
    content_zip = np.array(os.listdir(folder_in))

    if pattern is not None:
        if pattern_type == "shell":
            content_zip = [
                c for c in content_zip if fnmatch.fnmatch(os.path.basename(c), pattern)
            ]
        elif pattern_type == "regex":
            content_zip = [
                c
                for c in content_zip
                if re.match(pattern, os.path.basename(c)) is not None
            ]
        else:
            raise ValueError('Unknown pattern_type, must be either "shell" or "regex".')

    # Derive datetime of each file
    times_zip = np.array(
        [
            datetime.datetime.strptime(c[3:12], "%y%j%H%M").replace(
                tzinfo=datetime.timezone.utc
            )
            for c in content_zip
        ]
    )

    # Get a list of all files to retrieve
    conditions = times_zip == time

    # Filter on sweeps:
    if sweeps is not None:
        sweeps_zip = np.array([int(c[-3:]) for c in content_zip])
        # Get a list of all files to retrieve
        conditions_sweep = np.array([s in sweeps for s in sweeps_zip])
        conditions = np.logical_and(conditions, conditions_sweep)

    if not np.any(conditions):
        msg = """
        No file was found corresponding to this format, verify pattern and product_name
        """
        warn(msg)
        return []

    files = sorted(np.array([folder_in + c for c in np.array(content_zip)[conditions]]))

    return files


def _retrieve_prod_daily(
    start_time,
    end_time,
    product_name,
    folder_out=None,
    pattern=None,
    pattern_type="shell",
    sweeps=None,
    hdf5=False,
):
    """Retrieve radar product files for a given day, with an option to store them in RAM."""

    if hdf5:
        folder_radar = constants.FOLDER_RADAR_HDF5
        product_suffix = "hdf5" if product_name in ("CPCH", "CPC") else ""
    else:
        folder_radar = (
            constants.FOLDER_RADARH
            if product_name.startswith("MH")
            else constants.FOLDER_RADAR
        )
        product_suffix = ""

    suffix = f"{start_time.year % 100:02d}{start_time.timetuple().tm_yday:03d}"
    folder_in = os.path.join(folder_radar, str(start_time.year), suffix)
    name_zipfile = f"{product_name}{product_suffix}{suffix}.zip"
    path_zipfile = os.path.join(folder_in, name_zipfile)

    with zipfile.ZipFile(path_zipfile, "r") as zipp:
        all_files = zipp.namelist()

        # Precompile regex if used
        if pattern and pattern_type == "regex":
            pattern_re = re.compile(pattern)

        selected = []
        timestamps = []
        sweeps_list = []

        for name in all_files:
            base = os.path.basename(name)
            if pattern:
                if pattern_type == "shell" and not fnmatch.fnmatch(base, pattern):
                    continue
                if pattern_type == "regex" and not pattern_re.match(base):
                    continue
                if pattern_type not in ("shell", "regex"):
                    raise ValueError(
                        'Unknown pattern_type, must be "shell" or "regex".'
                    )

            try:
                dt = datetime.datetime.strptime(name[3:12], "%y%j%H%M").replace(
                    tzinfo=datetime.timezone.utc
                )
            except ValueError:
                continue  # skip files with bad timestamp format

            if not (start_time <= dt <= end_time):
                continue

            if sweeps is not None:
                try:
                    sweep = int(name[-3:])
                except ValueError:
                    continue
                if sweep not in sweeps:
                    continue
                sweeps_list.append(sweep)

            selected.append(name)
            timestamps.append(dt)

        if not selected:
            warn(
                "No file was found corresponding to this format, verify pattern and product_name"
            )
            return []

        if not folder_out:
            return [BytesIO(zipp.read(file_name)) for file_name in selected]
        else:
            # Only unzip files that don't already exist
            existing_files = {
                os.path.basename(f)
                for f in os.listdir(folder_out)
                if os.path.isfile(os.path.join(folder_out, f))
            }
            files_to_extract = [
                f for f in selected if os.path.basename(f) not in existing_files
            ]

            for file_name in files_to_extract:
                zipp.extract(file_name, path=folder_out)

            return sorted(
                [os.path.join(folder_out, os.path.basename(f)) for f in selected]
            )


def retrieve_CPCCV(time, stations):
    """Retrieves cross-validation CPC data for a set of stations from
    the xls files prepared by Yanni

    Parameters
    ----------

    time : datetime.datetime instance
        starting time of the time range
    stations : list of str
        list of weather stations at which to retrieve the CPC.CV data

    Returns
    -------
    Three arrays with the variables CPC.CV, CPC and AQC at the given stations
    extracted from the cross-validation files.
    """

    def _file_times(fname):
        tformat = "%Y%m%d%H00"
        if "sheet." not in fname:
            fname = fname.replace("sheet", "sheet.")  # handle buggy files
        tstart, tend = os.path.basename(fname).split(".")[1].split("_")
        tstart = _make_timezone_aware(datetime.datetime.strptime(tstart, tformat))
        tend = _make_timezone_aware(datetime.datetime.strptime(tend, tformat))
        return tstart, tend

    time = _make_timezone_aware(time)
    year = time.year

    # Check if monthly or daily files have to be used
    all_files = sorted(
        glob.glob(
            os.path.join(constants.FOLDER_CPCCV_DAILY, "**", "*.xls"), recursive=True
        )
    )
    first_date, _ = _file_times(all_files[0])

    if time < first_date:
        use_monthly = True
    else:
        use_monthly = False

    if use_monthly:
        folder = constants.FOLDER_CPCCV_MONTHLY
    else:
        folder = os.path.join(constants.FOLDER_CPCCV_DAILY, str(year))

    files = sorted(
        [f for f in glob.glob(os.path.join(folder, "*.xls")) if "s.xls" not in f]
    )

    all_tstart = []
    all_tend = []
    for fname in files:
        tstart, tend = _file_times(fname)
        all_tstart.append(tstart)
        all_tend.append(tend)
    all_tstart = np.array(all_tstart)
    all_tend = np.array(all_tend)

    match = np.where((time <= all_tend) & (time >= all_tstart))[0]

    if not len(match):
        print("Could not find CPC.CV file for time {:s}".format(time))
        return (
            np.zeros((len(stations))) + np.nan,
            np.zeros((len(stations))) + np.nan,
            np.zeros((len(stations))) + np.nan,
        )

    data = io.read_xls(files[match[0]])
    hour = int(datetime.datetime.strftime(time, "%Y%m%d%H00"))
    # Select data at stations and time
    cpc_cv_values = []
    cpc_values = []
    aqc_values = []

    for station in stations:
        sel_row = data.loc[(data["nat.abbr"] == station) & (data["time.stamp"] == hour)]
        if sel_row.empty:
            cpc_cv_values.append(np.nan)
            cpc_values.append(np.nan)
            aqc_values.append(np.nan)
        else:
            row = sel_row.iloc[0]
            cpc_cv_values.append(row["CPC.CV"])
            cpc_values.append(row["CPC"])
            aqc_values.append(row["AQC"])

    return (
        np.array(cpc_cv_values),
        np.array(cpc_values),
        np.array(aqc_values),
    )
