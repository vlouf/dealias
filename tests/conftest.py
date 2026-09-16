"""
Pytest configuration for the UNRAVEL test suite.

Keeps the live log readable: third-party libraries are extremely chatty at DEBUG
(numba logs every JIT decoration, matplotlib its font cache, h5py every open), so
they are pinned to WARNING. That way `--log-cli-level=DEBUG` shows the per-sweep
dealiasing breakdown and little else.
"""

import logging

NOISY_LOGGERS = (
    "matplotlib",
    "numba",
    "h5py",
    "PIL",
    "asyncio",
    "urllib3",
    "requests",
    "dask",
    "distributed",
    "fsspec",
    "pyart",
    "xarray",
)


def pytest_configure(config):
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
