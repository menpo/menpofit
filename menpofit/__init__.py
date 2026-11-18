from . import builder
from . import differentiable
from . import fitter
from . import modelinstance

from . import aam
from . import atm
from . import clm
from . import dlib
from . import lk
from . import math
from . import result
from . import sdm
from . import transform
from . import visualize


try:
    from ._version import __version__
except ImportError:  # running from a source tree that has not been built or installed
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        __version__ = _version("menpofit")
    except PackageNotFoundError:
        __version__ = "unknown"
