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
import pickle
import numpy as np
import datetime
from pathlib import Path
import os
import importlib.util
from ..common.logger import get_logger
from ..common import constants
from ..common.retrieve_data import (
    retrieve_prod,
    retrieve_hzt_prod,
    retrieve_prod_RT,
    retrieve_hzt_RT,
)
from ..common.lookup import get_lookup
from ..common.utils import split_by_time, envyaml
from ..common.radarprocessing import Radar, HZT_hourly_to_5min
from ..common.io_data import save_gif
from .qpe_utils import disaggregate, features_to_chgrid, pol_to_cart_valid
from .qpe_utils import pol_to_cart, qpe_to_chgrid, outlier_removal
from scipy.ndimage import gaussian_filter
from pyart.aux_io.odim_h5_writer import write_odim_grid_h5

_PYSTEPS_AVAILABLE = importlib.util.find_spec("pysteps") is None
warnings.filterwarnings("ignore")

###############################################################################
# Centerpoints of all QPE grid cells
Y_QPE_CENTERS = constants.Y_QPE_CENTERS
X_QPE_CENTERS = constants.X_QPE_CENTERS

NBINS_X = len(X_QPE_CENTERS)
NBINS_Y = len(Y_QPE_CENTERS)

dir_path = os.path.dirname(os.path.realpath(__file__))


class QPEProcessor(object):
    def __init__(self, config_file, models, rt=False, verbose=True):
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
        rt : bool
            Whether or not to run the model in real-time mode
        """
        self.logger = get_logger("QPE postproc", verbose)

        try:
            config = envyaml(config_file)
        except Exception as err:
            self.logger.warning(
                "Using default config as no valid config file was provided"
            )
            config_file = os.path.join(dir_path, "/default_config.yml")
            self.logger.warning(f"Error is {err}")

        config = envyaml(config_file)

        self.config = config
        self.models = models
        self.rt = rt

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

        if self.rt:
            # Used for testing the RT mode on CSCS
            import socket

            if "balfrin" in socket.gethostname() or "nid" in socket.gethostname():
                constants.FOLDER_RADAR = (
                    "/scratch/rgugerli/rainforest_debug/output/output_RT/temp/"
                )
                constants.FOLDER_ISO0 = (
                    "/scratch/rgugerli/rainforest_debug/output/output_RT/temp/HZT/"
                )

    def fetch_data(self, t0, t1=None):
        """
        Retrieves and add new polar radar and status data to the QPEProcessor
        for a given time range

        Parameters
        ----------
        t0 : datetime
            Start time of the timerange in datetime format
        t1 : datetime
            End time of the timerange in datetime format, will not be used if rt mode is on,
            as in this case only data for t0 will be obtained
        """

        self.radar_files = {}
        self.status_files = {}

        if self.rt:
            tstep = t0
        # Retrieve polar files and lookup tables for all radars
        for rad in self.config["RADARS"]:
            self.logger.info(f"Retrieving data for radar {rad}")
            try:
                if self.rt:
                    radfiles = retrieve_prod_RT(
                        tstep, product_name="ML" + rad, sweeps=self.config["SWEEPS"]
                    )
                    statfiles = retrieve_prod_RT(
                        tstep, product_name="ST" + rad, pattern="ST*.xml", sweeps=None
                    )
                else:
                    radfiles = retrieve_prod(
                        t0,
                        t1,
                        product_name="ML" + rad,
                        folder_out=self.config["TMP_FOLDER"],
                        sweeps=self.config["SWEEPS"],
                    )
                    statfiles = retrieve_prod(
                        t0,
                        t1,
                        product_name="ST" + rad,
                        folder_out=self.config["TMP_FOLDER"],
                        pattern="ST*.xml",
                    )

                self.radar_files[rad] = split_by_time(radfiles)
                self.status_files[rad] = split_by_time(statfiles)
            except Exception as err:
                error_msg = f"Failed to retrieve data for radar {rad}, error is {err}"
                self.logger.error(error_msg)

        # Retrieve iso0 height files
        if "ISO0_HEIGHT" in self.cosmo_var:
            try:
                if self.rt:
                    files_hzt = retrieve_hzt_RT(tstep)
                else:
                    files_hzt = retrieve_hzt_prod(self.config["TMP_FOLDER"], t0, t1)
                self.files_hzt = split_by_time(files_hzt)
            except Exception as err:
                self.files_hzt = {}
                self.logger.error("Failed to retrieve HZT data")
                self.logger.error(f"Error is {err}")

    def fetch_data_test(self, t0, t1):
        """
        Fetch the data for a qpe run on the cloud, to be used for unit tests only
        """
        from ..common.object_storage import ObjectStorage

        objsto = ObjectStorage()

        INPUT_FOLDER = Path(os.environ["RAINFOREST_DATAPATH"], "references", "qpe_run")
        self.radar_files = {}
        self.status_files = {}
        self.T_files = {}
        self.files_hzt = {}

        # Retrieve polar files and lookup tables for all radars
        for rad in self.config["RADARS"]:
            self.logger.info(f"Retrieving data for radar {rad}")
            try:
                radfiles = []
                for sweep in self.config["SWEEPS"]:
                    datestr0 = datetime.datetime.strftime(t0, "%y%j%H%M")
                    datestr1 = datetime.datetime.strftime(t1, "%y%j%H%M")

                    fname0 = str(
                        Path(
                            INPUT_FOLDER,
                            "ML" + rad + datestr0 + "0U.0" + str(sweep).zfill(2),
                        )
                    )
                    fname1 = str(
                        Path(
                            INPUT_FOLDER,
                            "ML" + rad + datestr1 + "0U.0" + str(sweep).zfill(2),
                        )
                    )

                    radfiles.append(objsto.check_file(fname0))
                    radfiles.append(objsto.check_file(fname1))

                self.radar_files[rad] = split_by_time(radfiles)

                fname0 = str(Path(INPUT_FOLDER, "ST" + rad + datestr0 + "0U.xml"))
                fname1 = str(Path(INPUT_FOLDER, "ST" + rad + datestr1 + "0U.xml"))

                statfiles = [objsto.check_file(fname0), objsto.check_file(fname1)]

                # For COSMO we get use only data at end of timestep for test case
                fname = objsto.check_file(
                    str(Path(INPUT_FOLDER, "TL" + rad + datestr1 + "0.p"))
                )
                T_files = [fname]

                self.status_files[rad] = split_by_time(statfiles)
                self.T_files[rad] = split_by_time(T_files)
            except Exception as err:
                self.logger.error(f"Failed to retrieve data for radar {rad}")
                self.logger.error(f"Error is {err}")

        # ISOTHERMAL HEIGHT is independent of radar gate, i.e. this step is done later
        try:
            # For COSMO we get use only data at end of timestep for test case
            fname = objsto.check_file(str(Path(INPUT_FOLDER, "hzt" + datestr1 + "0.p")))
            self.files_hzt = split_by_time([fname])
        except Exception as err:
            self.logger.error("Failed to retrieve hzt data")
            self.logger.error(f"Error is {err}")

    def save_output(self, qpe, tstamp, filepath):
        """Saves output as defined in config file

        Args:
            qpe (array): Precipitation estimates on grid
            tstamp (datetime object): timestep of QPE
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
                grid = qpe_to_chgrid(qpe, tstamp, missing_files=self.missing_files)
                filepath += ".h5"
                self.logger.info(f"Writing file {filepath}")
                write_odim_grid_h5(
                    filepath,
                    grid,
                    time_ref="end",
                    undefined_value=0,
                    odim_convention="ODIM_H5/V2_3",
                )

    def save_features(self, features, features_labels, t, filepath):
        grid = features_to_chgrid(
            features, features_labels, t, missing_files=self.missing_files
        )
        filepath += ".h5"
        self.logger.info(f"Writing file {filepath}")
        write_odim_grid_h5(
            filepath,
            grid,
            time_ref="end",
            undefined_value=0,
            odim_convention="ODIM_H5/V2_3",
        )

    def compute(
        self,
        output_folder,
        t0,
        t1,
        timestep=5,
        basename="RFO%y%j%H%MVH",
        test_mode=False,
    ):
        """
        Computes QPE values for a given time range and stores them in a
        folder, in a binary format

        Parameters
        ----------
        output_folder : str
            Folder where to store the computed QPE fields, note that subfolders
            for every model will be created in this folder
        t0 : datetime
            Start time of the timerange in datetime format
        t1 : datetime
            End time of the timerange in datetime format
        timestep : int (optional)
            In case you don't to generate a new product every 5 minute, you can
            change the time here, f.ex. 10 min, will compute the QPE only every
            two sets of radar scans
        basename: str (optional)
            Pattern for the filenames, default is  'RFQ%y%j%H%M' which uses
            the same standard as other MeteoSwiss products
            (example RFQ191011055), in real-time version, it is 'RFQ%y%j%H%MVH'
        test_mode : bool
            Is used only in github actions CI with special data for unit tests, should always be set to
            false

        """

        # Force utc tzinfo
        t0 = t0.replace(tzinfo=datetime.timezone.utc)
        t1 = t1.replace(tzinfo=datetime.timezone.utc)

        # Retrieve one timestamp before beginning for lead time
        tL = t0 - datetime.timedelta(minutes=timestep)

        if not self.rt:
            # Retrieve data for whole time range
            if test_mode:
                self.fetch_data_test(tL, t1)
            else:
                self.fetch_data(tL, t1)

        # Get all timesteps in time range
        n_incr = int((t1 - tL).total_seconds() / (60 * timestep))
        timeserie = tL + np.array(
            [datetime.timedelta(minutes=timestep * i) for i in range(n_incr + 1)]
        )

        qpe_prev = {}
        X_prev = {}
        hzt_cosmo_fields = {}
        for i, ct in enumerate(timeserie):  # Loop on timesteps
            self.logger.info("====")
            self.logger.info(f"Processing time {ct}")
            tstr = datetime.datetime.strftime(ct, basename)

            # Get lead time file in RT mode
            if i == 0 and self.rt:
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
                    self.logger.info("Already available: LEAD time " + str(ct))
                    continue
                else:
                    self.logger.info("Processing LEAD time " + str(ct))
            else:
                self.logger.info("Processing time " + str(ct))

            if self.rt:
                # Retrieve data for timestep
                self.fetch_data(ct)

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

            # Get COSMO temperature for all radars for this timestamp
            if not test_mode:
                if "ISO0_HEIGHT" in self.cosmo_var:
                    if ct not in hzt_cosmo_fields:  # check if not none
                        try:
                            hzt_cosmo_fields = HZT_hourly_to_5min(
                                ct, self.files_hzt, tsteps_min=timestep
                            )
                            self.logger.info(
                                f'Interpolating HZT fields for timestep {ct.strftime("%Y%m%d%H%M")}'
                            )
                            self.logger.debug("Saving temporary HZT data to cache")

                            np.save(
                                f"{self.config['TMP_FOLDER']}/hzt_cached",
                                np.ma.filled(hzt_cosmo_fields[ct], np.nan),
                            )

                        except Exception as err:
                            # Retrieve latest valid value from cache
                            self.logger.warning(
                                f"HZT temporal interpolation failed with error {err}, retrieving latest valid value from cache"
                            )
                            hzt_val = np.load(f"{self.config['TMP_FOLDER']}/hzt_cached")
                            hzt_val = np.ma.masked_invalid(hzt_val)
                            hzt_cosmo_fields = {ct: hzt_val}

            """Part one - compute radar variables and mask"""
            # Begin compilation of radarobject
            self.logger.info("Preparing all polar radar data")
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
                    rad,
                    self.radar_files[rad][ct],
                    self.status_files[rad][ct],
                    metranet_reader="C",
                )

                # if problem with radar file, exclude it and add to missing files list
                if len(radobjects[rad].radarfields) == 0:
                    self.missing_files[rad] = ct
                    self.logger.warning(f"Removing timestep {ct} of radar {rad}")
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
                    self.logger.warning(e)
                    self.logger.warning("Removing timestep {t} of radar {rad}")
                    continue
                radobjects[rad].compute_kdp(self.config["KDP_PARAMETERS"])

                if test_mode:
                    if "ISO0_HEIGHT" in self.cosmo_var:
                        hzt_cosmo_fields = pickle.load(open(self.files_hzt[t1], "rb"))

                # Add temperature indication to radar object
                if "ISO0_HEIGHT" in self.cosmo_var:
                    radobjects[rad].add_hzt_data(hzt_cosmo_fields[ct])

                # Delete files if config files requires
                try:
                    if self.config["CLEANUP"] == "delete_all":
                        for f in self.radar_files[rad][ct]:
                            if os.path.exists(f):
                                os.remove(f)
                        if os.path.exists(self.status_files[rad][ct]):
                            os.remove(self.status_files[rad][ct])
                except Exception:
                    self.logger.error(
                        "No cleanup was defined, unzipped files remain in temp-folder"
                    )

            self.logger.info("Processing all sweeps of all radars")
            for sweep in self.config["SWEEPS"]:  # Loop on sweeps
                if not self.rt:
                    self.logger.info("---")
                    self.logger.info("Processing sweep " + str(sweep))

                for rad in self.config["RADARS"]:  # Loop on radars, A,D,L,P,W
                    # If there is no radar file for the specific radar, continue to next radar
                    if rad not in radobjects.keys():
                        self.logger.info(
                            f"Processing radar {rad} - no data for this timestep!"
                        )
                        continue
                    elif sweep not in radobjects[rad].radsweeps.keys():
                        self.logger.info(
                            f"Processing sweep {sweep} of radar {rad} - no data for this timestep!"
                        )
                        continue
                    else:
                        if not self.rt:
                            self.logger.info(f"Processing sweep {sweep} of radar {rad}")

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
                                radobjects[rad]
                                .get_field(sweep, var)
                                .astype(np.float64),
                                np.nan,
                            )

                        # Mask on minimum zh
                        invalid = np.logical_or(
                            np.isnan(ZH), ZH < self.config["ZH_THRESHOLD"]
                        )

                        for var in datasweep.keys():
                            # Because add.at does not support nan we replace by zero
                            if var not in ["VISIB", "ISO0_HEIGHT", "HEIGHT"]:
                                datasweep[var][invalid] = np.nan

                        """Part three - convert to Cartesian"""
                        # get cart index of all polar gates for this sweep
                        lut_elev = self.lut_cart[rad][
                            self.lut_cart[rad][:, 0] == sweep - 1
                        ]  # 0-indexed

                        # Convert from Swiss-coordinates to array index
                        # wod: 9.02.2022: after careful examination and tests with
                        # random fields it seems that the second index must be incremented by 1
                        # for it to work
                        idx_ch = np.array(
                            (
                                len(X_QPE_CENTERS)
                                - (lut_elev[:, 4] - np.min(X_QPE_CENTERS)),
                                lut_elev[:, 3] - np.min(Y_QPE_CENTERS) + 1,
                            )
                        ).astype(int)

                        idx_ch = idx_ch.astype(int)
                        idx_polar = [lut_elev[:, 1], lut_elev[:, 2]]

                        # Compute VISIB at a given sweep/radar integrated over QPE grid
                        # This is used for visibility weighting in the vert integration
                        visib_radsweep = pol_to_cart(
                            (datasweep["VISIB"][idx_polar[0], idx_polar[1]]), idx_ch
                        )

                        # Compute validity of ZH at a given sweep/radar integrated over QPE grid
                        # This is used to compute radar fraction, it will be 1 only if at least one ZH is defined at a given
                        # QPE grid for this sweep/radar
                        isvalidzh_radsweep = pol_to_cart_valid(
                            np.isfinite(ZH[idx_polar[0], idx_polar[1]]).astype(int),
                            idx_ch,
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
                                            var_radsweep * W * isvalidzh_radsweep,
                                        )
                                    ),
                                    2,
                                )
                            # Do a weighted update of the sum of vertical weights, only where radar measures are available (ZH)
                            weights_cart[weight] = np.nansum(
                                np.dstack(
                                    (weights_cart[weight], W * isvalidzh_radsweep)
                                ),
                                2,
                            )

                    except Exception as err:
                        self.logger.error(f"Could not compute sweep {sweep}")
                        self.logger.error(f"Error is {err}")
                        pass

            """Part four - RF prediction"""
            self.logger.info("Applying RF model to retrieve predictions")
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
                if i == 0 or (self.rt and (k not in X_prev.keys())):
                    X_prev[k] = X

                # Take average between current timestep and t-5min
                Xcomb = np.nanmean((X_prev[k], X), axis=0)
                X_prev[k] = X

                if self.rt:
                    # Save files in a temporary format
                    np.save(
                        f"{self.config['TMP_FOLDER']}/{k}_{ct.strftime(basename)}_xprev",
                        X,
                    )
                if self.config["SAVE_FEATURES"]:
                    filepath = output_folder + "/" + k + "_FEATURES/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    self.logger.info(
                        f"Input features to RainForest written to {os.path.join(filepath, tstr)}"
                    )
                    self.save_features(Xcomb, model.variables, ct, filepath + tstr)

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
                    qpe_no_pp = qpe.copy()

                """Part five - Postprocessing"""
                if self.config["OUTLIER_REMOVAL"]:
                    qpe = outlier_removal(qpe)

                if self.config["GAUSSIAN_SIGMA"] > 0:
                    qpe = gaussian_filter(qpe, self.config["GAUSSIAN_SIGMA"])

                if self.rt:
                    # Save files in a temporary format
                    np.save(
                        f"{self.config['TMP_FOLDER']}/{k}_{ct.strftime(basename)}_qpeprev",
                        qpe,
                    )

                if i == 0 or (k not in X_prev.keys()):
                    qpe_prev[k] = qpe
                    if self.config["ADVECTION_CORRECTION"]:
                        # Because this is only the lead time, we go out here:
                        self.logger.info(
                            f"Processing time {ct} was only used as lead time and the QPE maps will not be saved"
                        )
                        continue

                """ for advection correction use separate path """
                comp = np.array([qpe_prev[k].copy(), qpe.copy()])
                qpe_prev[k] = qpe

                if self.config["ADVECTION_CORRECTION"] and i > 0:
                    if not _PYSTEPS_AVAILABLE:
                        self.logger.error(
                            "Pysteps is not available, no qpe disaggregation will be performed!"
                        )
                    qpe_ac = disaggregate(comp)

                    filepath = output_folder + "/" + k + "_AC/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    self.save_output(qpe_ac, ct, filepath + tstr)

                """Part six - Save main output"""
                filepath = output_folder + "/" + k
                if not os.path.exists(filepath):
                    os.mkdir(filepath)
                filepath += "/" + tstr
                self.save_output(qpe, ct, filepath)

                if self.config["SAVE_NON_POSTPROCESSED_DATA"]:
                    # Coding of folders (0 not applied, 1 applied)
                    # T: temporal disaggregation
                    # O: outlier removal
                    # G: Gaussian smoothing

                    filepath = output_folder + "/" + k + "_T0O0G0/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    self.save_output(qpe_no_temp, ct, filepath + tstr)

                    filepath = output_folder + "/" + k + "_T1O0G0/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    self.save_output(qpe_no_pp, ct, filepath + tstr)

                    # With outlier removal (O1), but without Gaussian smoothing (G0)
                    filepath = output_folder + "/" + k + "_T1O1G0/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    qpe_temp = outlier_removal(qpe_no_pp)
                    self.save_output(qpe_temp, ct, filepath + tstr)

                    # Without outlier removal (O0), but without Gaussian smoothing (G1)
                    filepath = output_folder + "/" + k + "_T1O0G1/"
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    qpe_temp = gaussian_filter(qpe_no_pp, self.config["GAUSSIAN_SIGMA"])
                    self.save_output(qpe_temp, ct, filepath + tstr)
