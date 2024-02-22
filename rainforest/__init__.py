import importlib.metadata as _importlib_metadata

from . import qpe
from . import common
from . import database
from . import ml
from . import performance

# Get the version
try:
    __version__ = _importlib_metadata.version("rainforest_mch")
except _importlib_metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

