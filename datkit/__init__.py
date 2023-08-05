#
# Main module
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
"""
Datkit.

Some scripts to work with time series data.

Assumes regularly sampled time series, where time is always increasing. If
these conditions are not met the behaviour is undefined.
"""

#
# Version information
#
from ._datkit_version import (  # noqa
    __version__,
    __version_tuple__,
)


#
# Paths
#

# Datkit root
import os, inspect  # noqa
try:
    frame = inspect.currentframe()
    DIR_DATKIT = os.path.abspath(os.path.dirname(inspect.getfile(frame)))
finally:
    # Always manually delete frame
    # https://docs.python.org/3/library/inspect.html
    del frame
del os, inspect


#
# Imports
#
from ._check_times import (  # noqa
    is_increasing,
    is_regularly_increasing,
    sampling_interval,
)

from ._points import (  # noqa
    abs_max_on,
    index,
    index_near,
    index_on,
    max_on,
    mean_on,
    min_on,
    value_at,
    value_near,
)

