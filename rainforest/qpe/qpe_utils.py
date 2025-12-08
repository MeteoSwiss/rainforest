import os
import numpy as np
import datetime
from pyart.testing import make_empty_grid
from pyart.aux_io.odim_h5 import proj4_to_dict
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates

try:
    import pysteps

    _PYSTEPS_AVAILABLE = True
except ImportError:
    _PYSTEPS_AVAILABLE = False

from ..common.utils import get_version
from ..common import constants
from ..common.add_at import add_at_64, add_at_int

###############################################################################
# Centerpoints of all QPE grid cells
Y_QPE_CENTERS = constants.Y_QPE_CENTERS
X_QPE_CENTERS = constants.X_QPE_CENTERS

NBINS_X = len(X_QPE_CENTERS)
NBINS_Y = len(Y_QPE_CENTERS)

dir_path = os.path.dirname(os.path.realpath(__file__))


def pol_to_cart_valid(isvalid, idx_cart):
    """
    Converts polar data to the Cartesian QPE grid

    Parameters
    ----------
    isvalid : ndarray
        1D array of polar radar data mask, 1 if data is valid, 0 otherwise
    idx_cart : ndarray
        List of Cartesian pixel coordinates of pol_data, its shape must be N x 2, where N
        is the length of pol_data

    Returns
    -------
    A numpy array of the size of the Cartesian grid

    """
    weights = np.zeros((NBINS_X, NBINS_Y)).astype(int)
    add_at_int(weights, idx_cart, isvalid)
    return weights > 0


def pol_to_cart(pol_data, idx_cart):
    """
    Converts polar data to the Cartesian QPE grid

    Parameters
    ----------
    pol_data : ndarray
        1D array of polar radar data to convert to Cartesian
    idx_cart : ndarray
        List of Cartesian pixel coordinates of pol_data, its shape must be N x 2, where N
        is the length of pol_data

    Returns
    -------
    A numpy array of the size of the Cartesian grid

    """
    cart_data = np.zeros((NBINS_X, NBINS_Y))
    weights = np.zeros((NBINS_X, NBINS_Y)).astype(int)
    add_at_int(weights, idx_cart, np.isfinite(pol_data).astype(int))
    pol_data[np.isnan(pol_data)] = 0
    add_at_64(cart_data, idx_cart, pol_data)
    return cart_data / weights


def features_to_chgrid(features, features_labels, time, missing_files):
    """
    Creates a pyart grid object from a features array

    Parameters
    ----------
    features : ndarray
        2D numpy array containing the input features for the rainforest algorithm
    features_labels : list
        names of all features
    time : datetime
        Start time of the scan
    missing_files : dict
        Containing all radars with corresponding timestamps that are missing


    Returns
    -------
    A pyart Grid object
    """

    grid = make_empty_grid(
        [1, NBINS_X, NBINS_Y],
        [
            [0, 0],
            [1000 * np.min(X_QPE_CENTERS), 1000 * np.max(X_QPE_CENTERS)],
            [1000 * np.min(Y_QPE_CENTERS), 1000 * np.max(Y_QPE_CENTERS)],
        ],
    )

    time_start = time - datetime.timedelta(seconds=5 * 60)
    grid.time["units"] = "seconds since {:s}".format(
        datetime.datetime.strftime(time_start, "%Y-%m-%dT%H:%M:%SZ")
    )
    grid.time["data"] = np.arange(0, 5 * 60 + 1)
    grid.origin_latitude["data"] = 46.9524
    grid.origin_longitude["data"] = 7.43958333
    grid.projection = proj4_to_dict(
        "+proj=somerc +lat_0=46.95240555555556 "
        + "+lon_0=7.439583333333333 +k_0=1 +x_0=600000 +y_0=200000"
        + " +ellps=bessel +towgs84=674.4,15.1,405.3,0,0,0,0 +units=m +no_defs"
    )

    for i in range(features.shape[1]):
        data = {}
        data["data"] = np.reshape(features[:, i], (NBINS_X, NBINS_Y))
        data["nodata"] = np.nan
        data["units"] = "UNKNOWN"
        data["long_name"] = features_labels[i]
        data["coordinates"] = "elevation azimuth range"
        data["product"] = b"RF_FEATURE"
        data["prodname"] = b"RF_FEATURE"
        data["nodata"] = np.nan
        data["_FillValue"] = np.nan

        grid.fields["RF_" + features_labels[i]] = data

    grid.metadata["source"] = b"ORG:215, CTY:644, CMT:MeteoSwiss (Switzerland)"
    grid.metadata["version"] = b"H5rad 2.3"
    grid.metadata["sw_version"] = get_version()
    # Add missing radar information
    quality = "ADLPW"
    if len(missing_files) != 0:
        rad_list = list(missing_files.keys())
        qual_new = quality
        for rad in rad_list:
            qual_new = qual_new.replace(rad, "-")
        quality = qual_new
    grid.metadata["radar"] = quality.encode()
    grid.metadata["nodes"] = "WMO:06661,WMO:06699,WMO:06768,WMO:06726,WMO:06776"

    return grid


def qpe_to_chgrid(qpe, time, missing_files, precision=2):
    """
    Creates a pyart grid object from a QPE array

    Parameters
    ----------
    qpe : ndarray
        2D numpy array containing the QPE data in the Swiss QPE grid
    time : datetime
        Start time of the scan
    missing_files : dict
        Containing all radars with corresponding timestamps that are missing
    precision : int
        Precision to use when storing the QPE data in the grid, default is 2
        (0.01)


    Returns
    -------
    A pyart Grid object
    """

    grid = make_empty_grid(
        [1, NBINS_X, NBINS_Y],
        [
            [0, 0],
            [1000 * np.min(X_QPE_CENTERS), 1000 * np.max(X_QPE_CENTERS)],
            [1000 * np.min(Y_QPE_CENTERS), 1000 * np.max(Y_QPE_CENTERS)],
        ],
    )

    time_start = time - datetime.timedelta(seconds=5 * 60)
    grid.time["units"] = "seconds since {:s}".format(
        datetime.datetime.strftime(time_start, "%Y-%m-%dT%H:%M:%SZ")
    )
    grid.time["data"] = np.arange(0, 5 * 60 + 1)
    grid.origin_latitude["data"] = 46.9524
    grid.origin_longitude["data"] = 7.43958333
    grid.projection = proj4_to_dict(
        "+proj=somerc +lat_0=46.95240555555556 "
        + "+lon_0=7.439583333333333 +k_0=1 +x_0=600000 +y_0=200000"
        + " +ellps=bessel +towgs84=674.4,15.1,405.3,0,0,0,0 +units=m +no_defs"
    )
    data = {}
    data["data"] = np.around(qpe, precision)
    data["units"] = "mm/hr"
    data["long_name"] = "Rainforest estimated rain rate"
    data["coordinates"] = "elevation azimuth range"
    data["product"] = b"RR"
    data["nodata"] = np.nan
    data["_FillValue"] = np.nan

    grid.fields["radar_estimated_rain_rate"] = data
    grid.fields["radar_estimated_rain_rate"]["prodname"] = "CHRFO"
    grid.metadata["source"] = b"ORG:215, CTY:644, CMT:MeteoSwiss (Switzerland)"
    grid.metadata["version"] = b"H5rad 2.3"
    grid.metadata["sw_version"] = get_version()
    # Add missing radar information
    quality = "ADLPW"
    if len(missing_files) != 0:
        rad_list = list(missing_files.keys())
        qual_new = quality
        for rad in rad_list:
            qual_new = qual_new.replace(rad, "-")
        quality = qual_new
    grid.metadata["radar"] = quality.encode()

    if "-" not in quality:
        grid.metadata["nodes"] = "WMO:06661,WMO:06699,WMO:06768,WMO:06726,WMO:06776"
    else:
        # 06661: Albis; 06699: Dôle; 06768: Lema; 06726: Plaine Morte; 06776: Weissfluh
        all_wmo = ["WMO:06661", "WMO:06699", "WMO:06768", "WMO:06726", "WMO:06776"]
        rad_wmo = []
        for ir, rad in enumerate(["A", "D", "L", "P", "W"]):
            if rad in quality:
                rad_wmo.append(all_wmo[ir])
        grid.metadata["nodes"] = ",".join(rad_wmo)

    return grid


def outlier_removal(image, N=3, threshold=3):
    """
    Performs localized outlier correction by standardizing the data in a moving
    window and remove values that are below - threshold or above + threshold

    Parameters
    ----------
    image : ndarray
        2D numpy array
    N : int
        size of the moving window, for both rows and columns ( the window is
        square)
    threshold : threshold for a standardized value to be considered an outlier

    Returns
    -------
    An outlier removed version of the image with the same shape
    """

    im = np.array(image, dtype=float)
    im2 = im**2

    im_copy = im.copy()
    ones = np.ones(im.shape)

    kernel = np.ones((2 * N + 1, 2 * N + 1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")

    mean = s / ns
    std = np.sqrt((s2 - s**2 / ns) / ns)

    z = (image - mean) / std
    im_copy[z >= threshold] = mean[z >= threshold]
    return im_copy


def disaggregate(
    R,
    T=5,
    t=1,
):
    """
    Disaggregates a set of two consecutive QPE images to 1 min resolution and
    then averages them to get a new advection corrected QPE estimates

    Parameters
    ----------
    R : list
        List of two numpy 2D arrays, containing the previous and the current
        QPE estimate
    T : int
        The time interval that separates the two QPE images, default is 5 min
    t : int
        The reference time interval used for the disaggregation, 1 min by
        default, should not be touched I think

    Returns
    -------
    An advection corrected QPE estimate

    """
    x, y = np.meshgrid(
        np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float)
    )
    oflow_method = pysteps.motion.get_method("LK")
    V1 = oflow_method(np.log(R))
    Rd = np.zeros((R[0].shape))

    for i in range(1 + int(T / t)):
        pos1 = (y - i / T * V1[1], x - i / T * V1[0])
        R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * V1[1], x + (T - i) / T * V1[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        Rd += (T - i) * R1 + i * R2
    return 1 / T**2 * Rd
