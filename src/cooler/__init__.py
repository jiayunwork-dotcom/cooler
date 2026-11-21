"""
Cooler
~~~~~~

A cool place to store your Hi-C.

:author: Nezar Abdennur
:license: BSD-3-Clause

"""

from . import fileops, parallel
from ._balance import balance_cooler
from ._logging import get_verbosity_level, log_to_stderr, set_verbosity_level
from ._reduce import coarsen_cooler, merge_coolers, zoomify_cooler
from ._version import __format_version__, __version__
from .api import Cooler, annotate
from .create import create_cooler, create_scool, rename_chroms
from .expected import (
    BackgroundModel,
    ExpectedResult,
    estimate_expected,
    list_expected,
    load_expected,
    remove_expected,
    save_expected,
)
from .util import binnify, fetch_chromsizes, read_chromsizes

__all__ = [
    "BackgroundModel",
    "Cooler",
    "ExpectedResult",
    "__format_version__",
    "__version__",
    "annotate",
    "balance_cooler",
    "binnify",
    "coarsen_cooler",
    "create_cooler",
    "create_scool",
    "estimate_expected",
    "fetch_chromsizes",
    "fileops",
    "get_verbosity_level",
    "list_expected",
    "load_expected",
    "log_to_stderr",
    "merge_coolers",
    "parallel",
    "read_chromsizes",
    "remove_expected",
    "rename_chroms",
    "save_expected",
    "set_verbosity_level",
    "zoomify_cooler",
]
