#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function to compute the QPE estimations on the Swiss grid

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019

Modified by D. Wolfensberger and R. Gugerli
December 2022
"""

import warnings
import time
import numpy as np
import datetime
from pathlib import Path
import os
import glob
import fnmatch
import re
import importlib.util
import pandas as pd
from scipy.ndimage import gaussian_filter
from pyart.aux_io.odim_h5_writer import write_odim_grid_h5
from ..common.logger import get_qpe_rt_logger
from ..common import constants
from ..common.lookup import get_lookup
from ..common.utils import split_by_time, envyaml
from ..common.radarprocessing import Radar, HZT_hourly_to_5min
from ..common.io_data import save_gif
from .qpe_utils import disaggregate
from .qpe_utils import pol_to_cart, qpe_to_chgrid, outlier_removal


_PYSTEPS_AVAILABLE = importlib.util.find_spec("pysteps") is None

warnings.filterwarnings("ignore")

###############################################################################
# Centerpoints of all QPE grid cells
Y_QPE_CENTERS = constants.Y_QPE_CENTERS
X_QPE_CENTERS = constants.X_QPE_CENTERS

NBINS_X = len(X_QPE_CENTERS)
NBINS_Y = len(Y_QPE_CENTERS)

dir_path = os.path.dirname(os.path.realpath(__file__))


class QPEProcessor_RT_daemon(object):
    def __init__(self, config_file, models, log_dir="/srn/las/log/", verbose=False):
        """
        Creates a QPEProcessor object which can be used to compute QPE
        realizations with the RandomForest Regressors

        Parameters
        ----------
        config : str
            A yaml file containing all necessary options for the QPE algorithm
            check the default_config.yml file to see which keys are required
        models : dict
            A dictionary containing all RF models to use for prediction,
            keys in this dictionary are used to store outputs in separate
            folders whereas the values must be valid RF regressor instances
            as stored in the rf_models subfolder
        """

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger = get_qpe_rt_logger(log_dir, verbose)

        try:
            config = envyaml(config_file)
        except Exception as err:
            self.logger.warning(
                "Using default config as no valid config file was provided"
            )
            self.logger.warning(f"Error is {err}")
            config_file = dir_path + "/default_config.yml"

        config = envyaml(config_file)

        self.config = config

        self.models = models

        if self.config["RADARS"] == "all":
            self.config["RADARS"] = list(constants.RADARS.Abbrev)

        if self.config["SWEEPS"] == "all":
            self.config["SWEEPS"] = list(range(1, 21))

        # Precompute cart. lookup tables and radar heights
        self.lut_cart = {}
        self.rad_heights = {}

        for rad in self.config["RADARS"]:
            self.lut_cart[rad] = np.array(get_lookup("qpegrid_to_rad", radar=rad))
            self.rad_heights[rad] = {}
            heights = get_lookup("qpegrid_to_height_rad", rad)
            for sweep in self.config["SWEEPS"]:
                self.rad_heights[rad][sweep] = heights[sweep]

        self.model_weights_per_var = {}
        self.cosmo_var = []
        # keys of this dict are the variable used for the RF models, their values
        # is a list of lists with 1. the vertical weighting beta parameters to be used, 2. if visib_weighting is used or not
        for k in self.models.keys():
            for var in self.models[k].variables:
                if var not in self.model_weights_per_var.keys():
                    self.model_weights_per_var[var] = []
                if models[k].beta not in self.model_weights_per_var[var]:
                    self.model_weights_per_var[var].append(
                        (models[k].beta, models[k].visib_weighting)
                    )
                if (var == "T") or (var == "ISO0_HEIGHT"):
                    self.cosmo_var.append(var)

    def _retrieve_prod_RT(
        self, time, product_name, pattern=None, pattern_type="shell", sweeps=None
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

        # Get all files
        folder_radar = constants.FOLDER_RADAR
        folder_in = folder_radar + product_name + "/"

        # Get list of available files
        content_zip = np.array(os.listdir(folder_in))

        if pattern is not None:
            if pattern_type == "shell":
                content_zip = [
                    c
                    for c in content_zip
                    if fnmatch.fnmatch(os.path.basename(c), pattern)
                ]
            elif pattern_type == "regex":
                content_zip = [
                    c
                    for c in content_zip
                    if re.match(os.path.basename(c), pattern) is not None
                ]
            else:
                raise ValueError(
                    'Unknown pattern_type, must be either "shell" or "regex".'
                )

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
            raise ValueError(msg)

        files = sorted(
            np.array([folder_in + c for c in np.array(content_zip)[conditions]])
        )

        return files

    def wait_for_files(self, start_time=None, sleep_time=5, timeout=60):
        """
        Check if STx XML files exist for the given timestamp

        Parameters
        ----------

        """
        if not start_time:
            start_time = datetime.datetime.now(datetime.timezone.utc)

        # Get yydddhhmm format that corresponds to start_time
        start_time_clo = start_time - datetime.timedelta(
            minutes=start_time.minute % 5,
            seconds=start_time.second,
            microseconds=start_time.microsecond,
        )  # latest 5min time-period
        yydddhhmm = start_time_clo.strftime("%y%j%H%M")

        # 2. Loop until timeout or all files present
        radar_list = self.config["RADARS"]
        it = 0
        while True:
            now_time = datetime.datetime.now(datetime.timezone.utc)
            diff_time = (now_time - start_time).total_seconds()

            if diff_time >= timeout:
                self.logger.warning("Timeout reached — not all STx files found.")
                break

            if it == 0:
                self.logger.info(f"Checking STx files for radars: {radar_list}")

            nr_rad = len(radar_list)
            nr_stx = 0

            for rad in radar_list:
                stx_pattern = f"ST{rad}{yydddhhmm}"
                stx_dir = os.path.join(constants.FOLDER_RADAR, f"ST{rad}/")

                if os.path.isdir(stx_dir):
                    fnames = glob.glob(os.path.join(stx_dir, stx_pattern + "*.xml"))

                if len(fnames):
                    nr_stx += 1
                else:
                    self.logger.debug(
                        f"*** Missing: {os.path.join(stx_dir, stx_pattern + '*.xml')}"
                    )

            if nr_stx >= nr_rad:
                self.logger.info("All STx files found.")
                break
            else:
                self.logger.debug(f"Sleeping {sleep_time} seconds before retry...")
                time.sleep(sleep_time)
                it += 1

    def _retrieve_hzt_RT(self, tstep):
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

        try:
            # Sort filelist to most recent prediction
            content_filt = np.array([c for c in content_zip if c.endswith("800")])
            times_filt = np.array(
                [
                    (
                        datetime.datetime.strptime(c[3:12], "%y%j%H%M")
                        + datetime.timedelta(hours=int(c[-2::]))
                    ).replace(tzinfo=datetime.timezone.utc)
                    for c in content_filt
                ]
            )
            conditions = np.array(
                [np.logical_and((t >= start_time), (t <= end_time)) for t in times_filt]
            )

            content_filt = content_filt[conditions]
            times_filt = times_filt[conditions]
        except Exception as err:
            self.logger.error(
                "HZT data does not exist for " + start_time.strftime("%d-%b-%y")
            )
            self.logger.error(f"Error is {err}")
            files = None
            return

        # Check that an hourly estimate is available
        all_hours = pd.date_range(start=start_time, end=end_time, freq="H")

        if len(all_hours) != len(times_filt):
            content_times = np.array(
                [
                    (
                        datetime.datetime.strptime(c[3:12], "%y%j%H%M")
                        + datetime.timedelta(hours=int(c[-2::]))
                    ).replace(tzinfo=datetime.timezone.utc)
                    for c in content_zip
                ]
            )
            # Find time that is missing:
            for hh in all_hours:
                if hh not in times_filt:
                    hh_last = np.where(hh == content_times)
                    times_filt = np.sort(
                        np.append(times_filt, content_times[hh_last][-1])
                    )
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

    def fetch_data_RT(self, tstep):
        """
        Retrieves and add new polar radar and status data to the QPEProcessor
        for a given time range

        Parameters
        ----------
        tstep : datetime
                Get data for this timestamp
        """

        self.radar_files = {}
        self.status_files = {}

        # Retrieve polar files and lookup tables for all radars
        for rad in self.config["RADARS"]:
            self.logger.info("Retrieving data for radar {rad}")
            try:
                radfiles = self._retrieve_prod_RT(
                    tstep, product_name="ML" + rad, sweeps=self.config["SWEEPS"]
                )
                self.radar_files[rad] = split_by_time(radfiles)
                statfiles = self._retrieve_prod_RT(
                    tstep, product_name="ST" + rad, pattern="ST*.xml", sweeps=None
                )
                self.status_files[rad] = split_by_time(statfiles)
            except Exception as err:
                self.logger.error("Failed to retrieve data for radar {rad}")
                self.logger.error(f"Error is {err}")
        # Retrieve iso0 height files
        if "ISO0_HEIGHT" in self.cosmo_var:
            try:
                files_hzt = self._retrieve_hzt_RT(tstep)
                self.files_hzt = split_by_time(files_hzt)
            except Exception as err:
                self.files_hzt = {}
                self.logger.error("Failed to retrieve hzt data")
                self.logger.error(f"Error is {err}")

    def save_output(self, qpe, t, filepath):
        """Saves output as defined in config file

        Args:
            qpe (array): Precipitation estimates on grid
            t (datetime object): timestep of QPE
            filepath (str): Destination where file is saved
        """
        # Output in binary data and .gif format
        if self.config["DATA_FORMAT"] == "DN":
            if self.config["FILE_FORMAT"] == "DN":
                # Find idx from CPC scale
                qpe = np.searchsorted(constants.SCALE_CPC, qpe)
                qpe = qpe.astype("B")  # Convert to byte
                qpe[constants.MASK_NAN] = 255
                qpe.tofile(filepath)
            else:
                if self.config["FILE_FORMAT"] != "gif":
                    self.logger.error(
                        "Invalid file_format with data format DN, using gif output instead"
                    )
                qpe[constants.MASK_NAN] = -99
                filepath += ".gif"
                save_gif(filepath, qpe)

        # Output in binary data and .gif format
        qpe[constants.MASK_NAN] = np.nan
        if self.config["DATA_FORMAT"] == "float":
            if self.config["FILE_FORMAT"] == "float":
                qpe.astype(np.float32).tofile(filepath)
            else:
                if self.config["FILE_FORMAT"] != "ODIM":
                    self.logger.error(
                        "Invalid file format with data format float, using ODIM HDF5 output instead"
                    )
                grid = qpe_to_chgrid(qpe, t, missing_files=self.missing_files)
                filepath += ".h5"
                self.logger.info(f"Writing file {filepath}")
                write_odim_grid_h5(filepath, grid)

    def compute(self, output_folder, timestep=5, basename="RFO%y%j%H%MVH"):
        """
        Computes QPE values in real-time mode and stores them in a
        folder, in a binary format.
        Real-time mode will run forever and keep processing new data on the fly.

        Parameters
        ----------
        output_folder : str
            Folder where to store the computed QPE fields, note that subfolders
            for every model will be created in this folder
        timestep : int (optional)
            In case you don't to generate a new product every 5 minute, you can
            change the time here, f.ex. 10 min, will compute the QPE only every
            two sets of radar scans
        basename: str (optional)
            Pattern for the filenames, default is  'RFQ%y%j%H%M' which uses
            the same standard as other MeteoSwiss products
            (example RFQ191011055)

        """

        for model in self.models.keys():
            if self.config["ADVECTION_CORRECTION"]:
                model += "_AC"
            if not os.path.exists(Path(output_folder, model)):
                Path(output_folder, model).mkdir(parents=True, exist_ok=True)

        qpe_prev = {}
        X_prev = {}
        hzt_cosmo_fields = {}

        it = 0
        while True:  # infinite loop
            now = datetime.datetime.now(datetime.timezone.utc)
            delta_min = now.minute % 5
            if delta_min < 1:
                # If we are less than 2 min after the last 5 min period, we wait for the radar files
                # to arrive
                self.wait_for_files()

            now = datetime.datetime.now(datetime.timezone.utc)
            ct = now - datetime.timedelta(
                minutes=now.minute % 5, seconds=now.second, microseconds=now.microsecond
            )  # latest 5min time-period

            self.logger.info("====")

            # Get lead time file
            if it == 0:
                for k in self.models.keys():
                    tL_x_file = f"{self.config['TMP_FOLDER']}/{k}_{ct.strftime(basename)}_xprev.npy"
                    tL_qpe_file = f"{self.config['TMP_FOLDER']}/{k}_{ct.strftime(basename)}_qpeprev.npy"
                    one_file_missing = False
                    try:
                        X_prev[k] = np.load(tL_x_file)
                        qpe_prev[k] = np.load(tL_qpe_file)
                        os.remove(tL_x_file)
                        os.remove(tL_qpe_file)
                    except Exception:
                        one_file_missing = True
                # If all the files could be loaded, go directly to current timestep
                if not one_file_missing:
                    self.logger.info(f"Already available: LEAD time {ct}")
                    continue
                else:
                    self.logger.info(f"Processing LEAD time {ct}")
            else:
                self.logger.info(f"Processing time {ct}")

            # Retrieve data for time range
            self.fetch_data_RT(ct)
            # Log missing radar files
            self.missing_files = {}

            # Initialize RF features
            rf_features_cart = {}
            weights_cart = {}

            for var in self.model_weights_per_var.keys():
                for weight in self.model_weights_per_var[var]:
                    if weight not in list(rf_features_cart.keys()):
                        rf_features_cart[weight] = {}

                    rf_features_cart[weight][var] = np.zeros((NBINS_X, NBINS_Y))

                    # add weights
                    if weight not in weights_cart.keys():
                        weights_cart[weight] = np.zeros((NBINS_X, NBINS_Y))

            # Get HZT temperature for all radars for this timestamp
            if "ISO0_HEIGHT" in self.cosmo_var:
                if ct not in hzt_cosmo_fields:
                    try:
                        hzt_cosmo_fields = HZT_hourly_to_5min(
                            ct, self.files_hzt, tsteps_min=timestep
                        )
                        self.logger.info(f"Interpolating HZT fields for timestep {ct}")
                    except Exception as err:
                        hzt_cosmo_fields[ct] = hzt_cosmo_fields[
                            min(hzt_cosmo_fields, key=lambda k: abs(k - ct))
                        ]
                        self.logger.error(
                            f"HZT fields for timestep {ct} missing, using data from the previous valid one"
                        )
                        self.logger.error(f"Error is {err}")

            """Part one - compute radar variables and mask"""
            # Begin compilation of radarobject
            radobjects = {}
            for rad in self.config["RADARS"]:
                # if radar does not exist, add to missing file list
                if (rad not in self.radar_files) or (self.radar_files[rad] is None):
                    self.missing_files[rad] = ct
                    continue

                # if file does not exist, add to missing file list
                if (ct not in self.radar_files[rad]) or (
                    self.radar_files[rad][ct] is None
                ):
                    self.missing_files[rad] = ct
                    continue

                # Read raw radar file and create a RADAR object
                radobjects[rad] = Radar(
                    rad, self.radar_files[rad][ct], self.status_files[rad][ct]
                )

                # if problem with radar file, exclude it and add to missing files list
                if not len(radobjects[rad].radarfields):
                    self.missing_files[rad] = ct
                    self.logger.info(f"Removing timestep {ct} of radar {rad}")
                    continue

                # Process the radar data
                radobjects[rad].visib_mask(
                    self.config["VISIB_CORR"]["MIN_VISIB"],
                    self.config["VISIB_CORR"]["MAX_CORR"],
                )
                # If it cannot compute noise from the status file, remove timestep
                try:
                    radobjects[rad].snr_mask(self.config["SNR_THRESHOLD"])
                except Exception as e:
                    self.missing_files[rad] = ct
                    self.logger.info(e)
                    self.logger.info(f"Removing timestep {ct} of radar {rad}")
                    continue

                radobjects[rad].compute_kdp(self.config["KDP_PARAMETERS"])

                # Add temperature indication to radar object
                if "ISO0_HEIGHT" in self.cosmo_var:
                    radobjects[rad].add_hzt_data(hzt_cosmo_fields[ct])

            for sweep in self.config["SWEEPS"]:  # Loop on sweeps
                self.logger.info("---")
                self.logger.info(f"Processing sweep {sweep}")

                for rad in self.config["RADARS"]:  # Loop on radars, A,D,L,P,W
                    # If there is no radar file for the specific radar, continue to next radar
                    if rad not in radobjects.keys():
                        self.logger.info(
                            f"Processing radar {rad} - no data for this timestep!"
                        )
                        continue
                    elif sweep not in radobjects[rad].radsweeps.keys():
                        self.logger.info(
                            f"Processing sweep {sweep} of {rad} - no data for this timestep!"
                        )
                        continue
                    else:
                        self.logger.info(f"Processing radar {rad}")

                    try:
                        """Part two - retrieve radar data at every sweep"""
                        datasweep = {}
                        ZH = np.ma.filled(
                            radobjects[rad].get_field(sweep, "ZH"), np.nan
                        )

                        for var in self.model_weights_per_var.keys():
                            # These variables are computed differently
                            if "RADAR_prop" in var or var == "HEIGHT":
                                continue

                            datasweep[var] = np.ma.filled(
                                radobjects[rad].get_field(sweep, var), np.nan
                            )

                        # Mask on minimum zh
                        invalid = np.logical_or(
                            np.isnan(ZH), ZH < self.config["ZH_THRESHOLD"]
                        )

                        for var in datasweep.keys():
                            # Because add.at does not support nan we replace by zero
                            datasweep[var][invalid] = 0

                        """Part three - convert to Cartesian"""
                        # get cart index of all polar gates for this sweep
                        lut_elev = self.lut_cart[rad][
                            self.lut_cart[rad][:, 0] == sweep - 1
                        ]  # 0-indexed

                        # Convert from Swiss-coordinates to array index
                        idx_ch = np.vstack(
                            (
                                len(X_QPE_CENTERS)
                                - (lut_elev[:, 4] - np.min(X_QPE_CENTERS)),
                                lut_elev[:, 3] - np.min(Y_QPE_CENTERS),
                            )
                        ).astype(int)

                        idx_polar = [lut_elev[:, 1], lut_elev[:, 2]]

                        # Compute VISIB at a given sweep/radar integrated over QPE grid
                        # This is used for visibility weighting in the vert integration
                        visib_radsweep = pol_to_cart(
                            datasweep["VISIB"][idx_polar[0], idx_polar[1]], idx_ch
                        )
                        # Compute validity of ZH at a given sweep/radar integrated over QPE grid
                        # This is used to compute radar fraction, it will be 1 only if at least one ZH is defined at a given
                        # QPE grid for this sweep/radar
                        isvalidzh_radsweep = pol_to_cart(
                            np.isfinite(ZH[idx_polar[0], idx_polar[1]]), idx_ch
                        )

                        for weight in rf_features_cart.keys():
                            beta, visibweighting = weight

                            # Compute altitude weighting
                            W = 10 ** (beta * (self.rad_heights[rad][sweep] / 1000.0))
                            # Compute visib weighting
                            if visibweighting:
                                W *= visib_radsweep / 100

                            for var in rf_features_cart[weight].keys():
                                if var == "HEIGHT":  # precomputed in lookup tables
                                    var_radsweep = self.rad_heights[rad][sweep]
                                elif var == "VISIB":
                                    var_radsweep = visib_radsweep
                                elif "RADAR_prop_" in var:
                                    if (
                                        var[-1] != rad
                                    ):  # Can update RADAR_prop only for the current radar
                                        continue
                                    var_radsweep = isvalidzh_radsweep
                                else:
                                    if var in datasweep.keys():
                                        # Compute variable integrated over QPE grid for rad/sweep
                                        var_radsweep = pol_to_cart(
                                            datasweep[var][idx_polar[0], idx_polar[1]],
                                            idx_ch,
                                        )

                                # Do a weighted update of the rf_features_cart array
                                rf_features_cart[weight][var] = np.nansum(
                                    np.dstack(
                                        (
                                            rf_features_cart[weight][var],
                                            var_radsweep
                                            * W
                                            * (isvalidzh_radsweep == 1),
                                        )
                                    ),
                                    2,
                                )

                            # Do a weighted update of the sum of vertical weights, only where radar measures are available (ZH)
                            weights_cart[weight] = np.nansum(
                                np.dstack(
                                    (
                                        weights_cart[weight],
                                        W * (isvalidzh_radsweep == 1),
                                    )
                                ),
                                2,
                            )
                    except Exception as err:
                        self.logger.error(f"Could not compute sweep {sweep}")
                        self.logger.error(err)
                        pass

            """Part four - RF prediction"""
            # Get QPE estimate
            # X: current time step; X_prev: previous timestep (t-5min)
            for k in self.models.keys():
                model = self.models[k]
                X = []
                for v in model.variables:
                    dat = (
                        rf_features_cart[(model.beta, model.visib_weighting)][v]
                        / weights_cart[(model.beta, model.visib_weighting)]
                    )
                    # Inf occurs when weights are zero
                    dat[np.isinf(dat)] = np.nan
                    X.append(dat.ravel())

                X = np.array(X).T

                if (it == 0) or (k not in X_prev.keys()):
                    X_prev[k] = X

                # Take average between current timestep and t-5min
                Xcomb = np.nanmean((X_prev[k], X), axis=0)
                X_prev[k] = X

                # Save files in a temporary format
                np.save(
                    f"{self.config['TMP_FOLDER']}/{k}_{ct.strftime(basename)}_xprev", X
                )

                # Remove axis with only zeros
                Xcomb[np.isnan(Xcomb)] = 0
                validrows = (Xcomb > 0).any(axis=1)

                # Convert radar data to precipitation estimates
                qpe = np.zeros((NBINS_X, NBINS_Y), dtype=np.float32).ravel()
                try:
                    qpe[validrows] = self.models[k].predict(Xcomb[validrows, :])
                except Exception as err:
                    self.logger.error("Model inference failed!")
                    self.logger.error(f"Error is {err}")
                    pass

                qpe = np.reshape(qpe, (NBINS_X, NBINS_Y))
                if self.config["SAVE_NON_POSTPROCESSED_DATA"]:
                    qpe_no_temp = qpe.copy()

                """Temporal disaggregation; Rescale qpe through rproxy"""
                idx_zh = np.where(np.array(model.variables) == "zh_VISIB")[0][0]

                rproxy = (X[:, idx_zh] / constants.A_QPE) ** (1 / constants.B_QPE)
                rproxy[np.isnan(rproxy)] = 0

                rproxy_mean = (Xcomb[:, idx_zh] / constants.A_QPE) ** (
                    1 / constants.B_QPE
                )
                rproxy_mean[np.isnan(rproxy_mean)] = 0

                disag = rproxy / rproxy_mean
                disag = np.reshape(disag, (NBINS_X, NBINS_Y))
                disag[np.isnan(disag)] = 0
                qpe = qpe * disag

                if self.config["SAVE_NON_POSTPROCESSED_DATA"]:
                    qpe_no_pp = qpe.copy()

                """Part five - Postprocessing"""
                if self.config["OUTLIER_REMOVAL"]:
                    qpe = outlier_removal(qpe)

                if self.config["GAUSSIAN_SIGMA"] > 0:
                    qpe = gaussian_filter(qpe, self.config["GAUSSIAN_SIGMA"])

                # Save files in a temporary format
                np.save(
                    f"{self.config['TMP_FOLDER']}/{k}_{ct.strftime(basename)}_xprev", X
                )

                if (it == 0) or (k not in qpe_prev.keys()):
                    qpe_prev[k] = qpe
                    # Because this is only the lead time, we go out here:
                    self.logger.info(
                        f"Processing time {ct} was only used as lead time and the final QPE map will not be saved"
                    )
                    leadtime = True
                    continue
                else:
                    leadtime = False

                """ for advection correction use separate path """
                comp = np.array([qpe_prev[k].copy(), qpe.copy()])
                qpe_prev[k] = qpe

                if self.config["ADVECTION_CORRECTION"] and it > 0:
                    if not _PYSTEPS_AVAILABLE:
                        self.logger.error(
                            "Pysteps is not available, no qpe disaggregation will be performed!"
                        )
                    qpe_ac = disaggregate(comp)

                    tstr = datetime.datetime.strftime(ct, basename)
                    filepath = output_folder + "/" + k + "_AC/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    self.save_output(qpe_ac, ct, filepath + tstr)

                """Part six - Save main output"""
                tstr = datetime.datetime.strftime(ct, basename)
                filepath = output_folder + "/" + k
                filepath += "/" + tstr
                self.save_output(qpe, ct, filepath)

                if self.config["SAVE_NON_POSTPROCESSED_DATA"]:
                    # Coding of folders (0 not applied, 1 applied)
                    # T: temporal disaggregation
                    # O: outlier removal
                    # G: Gaussian smoothing

                    tstr = datetime.datetime.strftime(ct, basename)
                    filepath = output_folder + "/" + k + "_T0O0G0/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    self.save_output(qpe_no_temp, ct, filepath + tstr)

                    tstr = datetime.datetime.strftime(ct, basename)
                    filepath = output_folder + "/" + k + "_T1O0G0/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    self.save_output(qpe_no_pp, ct, filepath + tstr)

                    # With outlier removal (O1), but without Gaussian smoothing (G0)
                    tstr = datetime.datetime.strftime(ct, basename)
                    filepath = output_folder + "/" + k + "_T1O1G0/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    qpe_temp = outlier_removal(qpe_no_pp)
                    self.save_output(qpe_temp, ct, filepath + tstr)

                    # Without outlier removal (O0), but without Gaussian smoothing (G1)
                    tstr = datetime.datetime.strftime(ct, basename)
                    filepath = output_folder + "/" + k + "_T1O0G1/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    qpe_temp = gaussian_filter(qpe_no_pp, self.config["GAUSSIAN_SIGMA"])
                    self.save_output(qpe_temp, ct, filepath + tstr)

            if not leadtime:
                # Sleep to one minute after the next 05 period
                sec_to_sleep = int(
                    (
                        (ct + datetime.timedelta(minutes=5))
                        - datetime.datetime.now(datetime.timezone.utc)
                    ).total_seconds()
                )
                if sec_to_sleep > 0:
                    self.logger.info(f"Sleeping {sec_to_sleep} sec")
                    time.sleep(sec_to_sleep)
                    self.wait_for_files()
            it += 1
