import os
import time
import logging
import datetime
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
import requests

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    os.environ["PYART_QUIET"] = "1"  # Disable Py-ART disclaimer
    import pyart

import unravel

LOGGER = logging.getLogger("unravel.tests")

# Flag values produced by the dealiasing (see dealiasing_process_2D docstring).
FLAG_NODATA, FLAG_UNPROCESSED, FLAG_UNCHANGED, FLAG_DEALIASED = -3, 0, 1, 2


def logm(message: str) -> None:
    """
    Log a message to the pytest live log.

    Shown by default at INFO. Use `--log-cli-level=DEBUG` for the per-sweep
    breakdown, or `--log-cli-level=WARNING` to silence the run.
    """
    LOGGER.info(message, stacklevel=2)


def logd(message: str) -> None:
    """Log per-sweep detail, only shown with --log-cli-level=DEBUG."""
    LOGGER.debug(message, stacklevel=2)


@contextmanager
def timed(what: str):
    """Time a phase, so a slow stage is distinguishable from a hang."""
    logm(f"{what}...")
    start = time.perf_counter()
    yield
    logm(f"{what}: done in {time.perf_counter() - start:.1f}s")


def _pct(count: int, total: int) -> str:
    """Percentage, guarding the empty-sweep case."""
    return "  n/a" if total == 0 else f"{100.0 * count / total:5.1f}%"


def _absmax(values: np.ndarray) -> float:
    """Largest absolute velocity, ignoring NaNs and the all-NaN case."""
    values = np.ma.filled(values, np.nan).astype(float)
    return float("nan") if not np.any(np.isfinite(values)) else float(np.nanmax(np.abs(values)))


def log_odim_volume(datasets, vel_name: str, out_vel_name: str, flag_name: str) -> None:
    """
    Report what the dealiasing actually did to an ODIM volume.

    One summary line at INFO, one line per sweep at DEBUG. `|v|max` rising above
    the Nyquist velocity is the visible sign that gates really were unfolded.
    """
    gates = valid = unprocessed = dealiased = 0
    elevations, nyquists = [], []

    for idx, ds in enumerate(datasets):
        flag = ds[flag_name].values
        n_valid = int((flag != FLAG_NODATA).sum())
        n_unprocessed = int((flag == FLAG_UNPROCESSED).sum())
        n_dealiased = int((flag == FLAG_DEALIASED).sum())
        elevation = float(ds["elevation"].values[0])
        nyquist = float(ds.attrs["NI"])

        gates += flag.size
        valid += n_valid
        unprocessed += n_unprocessed
        dealiased += n_dealiased
        elevations.append(elevation)
        nyquists.append(nyquist)

        logd(
            f"  sweep {idx:02d} elev={elevation:5.1f} NI={nyquist:5.2f} "
            f"{flag.shape[0]:>4}x{flag.shape[1]:<4} data={_pct(n_valid, flag.size)} "
            f"unproc={_pct(n_unprocessed, n_valid)} unfolded={_pct(n_dealiased, n_valid)} "
            f"|v|max {_absmax(ds[vel_name].values):5.1f} -> {_absmax(ds[out_vel_name].values):5.1f}"
        )

    logm(
        f"{len(datasets)} sweeps, elev {min(elevations):.1f}-{max(elevations):.1f} deg, "
        f"NI {min(nyquists):.2f}-{max(nyquists):.2f} m/s, {gates / 1e6:.2f}M gates | "
        f"{_pct(valid, gates)} with data, {_pct(dealiased, valid)} unfolded, "
        f"{_pct(unprocessed, valid)} left unprocessed"
    )


def log_pyart_volume(radar, velname: str, dealiased: np.ndarray) -> None:
    """Report what the dealiasing did to a Py-ART volume (no flag field returned)."""
    before = np.ma.filled(radar.fields[velname]["data"], np.nan).astype(float)
    after = np.ma.filled(dealiased, np.nan).astype(float)
    comparable = np.isfinite(before) & np.isfinite(after)
    changed = int((comparable & ~np.isclose(before, after)).sum())
    logm(
        f"{radar.nsweeps} sweeps, elev {radar.elevation['data'].min():.1f}-"
        f"{radar.elevation['data'].max():.1f} deg, {before.size / 1e6:.2f}M gates | "
        f"{_pct(int(comparable.sum()), before.size)} with data, "
        f"{_pct(changed, int(comparable.sum()))} unfolded | "
        f"|v|max {_absmax(before):.1f} -> {_absmax(after):.1f} m/s"
    )


def download_cpol_data(date: datetime.datetime) -> str:
    """
    Download CPOL data for given date.
    Parameters:
    ===========
    date: str or datetime or pd.Timestamp
        Date time for which we want the CPOL data
    ppi: bool
        True for downloading the PPIs and False for downloading the gridded data.
    """
    year = date.year
    datestr = date.strftime("%Y%m%d")
    datetimestr = date.strftime("%Y%m%d.%H%M")
    url = f"https://dapds00.nci.org.au/thredds/fileServer/hj10/cpol/cpol_level_1b/v2020/ppi/{year}/{datestr}/twp10cpolppi.b1.{datetimestr}00.nc"
    fname = os.path.basename(url)
    try:
        os.mkdir("dwl")
    except FileExistsError:
        pass
    outfilename = os.path.join("dwl", fname)
    if os.path.isfile(outfilename):
        logm("Radar data file already exists, doing nothing")
        return outfilename
    r = requests.get(url, timeout=30)
    try:
        r.raise_for_status()
    except Exception as exc:
        raise ValueError(
            "No file found for this date. CPOL ran from 1998-12-6 to 2017-5-2, wet season only. Try another date."
        ) from exc
    with open(outfilename, "wb") as fid:
        fid.write(r.content)
    return outfilename


def get_odim_test_file() -> str:
    """
    Get path to ODIM test file.

    Returns:
    ========
    str: Path to ODIM test file
    """
    # Default test file location - use os.path.join for cross-platform compatibility
    default_file = os.path.join("tests", "data", "49_20240825_070000.pvol.h5")

    # Allow override via environment variable for CI/CD
    test_file = os.environ.get("ODIM_TEST_FILE", default_file)

    # Normalize path for current OS
    test_file = os.path.normpath(test_file)

    if not os.path.isfile(test_file):
        pytest.skip(f"ODIM test file not found at {test_file}. Expected at {os.path.normpath(default_file)}")

    logm(f"Using ODIM test file: {test_file}")
    return test_file


@pytest.mark.filterwarnings("ignore:.*CfRadial module is deprecated.*:UserWarning")
def test_pyart():
    """Test Py-ART dealiasing on CPOL data."""
    date = datetime.datetime(2014, 2, 18, 20, 0)
    with timed(f"Fetching CPOL data for {date:%Y-%m-%d %H:%M}"):
        filename = download_cpol_data(date)

    with timed(f"Reading {os.path.basename(filename)} with Py-ART"):
        radar = pyart.io.read(filename)
    assert isinstance(radar, pyart.core.Radar), "Radar object not created successfully"
    logm(f"Read {radar.nsweeps} sweeps, {len(radar.fields)} fields")
    logd(f"  fields: {', '.join(sorted(radar.fields))}")

    with timed("Dealiasing (pyart, nyquist=13.3 m/s)"):
        vel = unravel.unravel_3D_pyart(radar, "velocity", "corrected_reflectivity", nyquist_velocity=13.3)

    assert vel is not None, "Dealiased velocity field is None"
    log_pyart_volume(radar, "velocity", vel)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pyodim_from_file():
    """Test pyodim dealiasing by reading directly from file."""
    try:
        import pyodim
    except ImportError:
        pytest.skip("pyodim not installed")

    test_file = get_odim_test_file()

    # Read and dealias in one call, straight from the file path.
    with timed("Dealiasing from file (strategy=long_range, alpha=0.6)"):
        datasets = unravel.unravel_3D_pyodim(
            test_file, vel_name="VRADH", output_vel_name="unraveled_velocity", strategy="long_range", alpha=0.6
        )

    assert datasets is not None, "Returned datasets is None"
    assert isinstance(datasets, list), "Returned object is not a list"
    assert len(datasets) > 0, "No datasets returned"

    # Check first dataset has required fields
    first_ds = datasets[0]
    assert "unraveled_velocity" in first_ds, "Dealiased velocity field not found"
    assert "unraveled_velocity_flag" in first_ds, "Flag field not found"

    log_odim_volume(datasets, "VRADH", "unraveled_velocity", "unraveled_velocity_flag")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pyodim_from_datasets():
    """Test pyodim dealiasing with pre-loaded datasets (preprocessing workflow)."""
    try:
        import pyodim
    except ImportError:
        pytest.skip("pyodim or xarray not installed")

    test_file = get_odim_test_file()

    # Step 1: Load datasets with pyodim
    # pyodim >= 0.7: reading is eager by default (lazy=True returns dask delayed)
    with timed("Reading sweeps with pyodim"):
        datasets = pyodim.read_odim(test_file)
    logm(f"Loaded {len(datasets)} sweeps, variables: {', '.join(sorted(datasets[0].data_vars))}")

    # Step 2: Simulate preprocessing (e.g., dual-PRF correction would go here)
    # For testing, we'll just pass the datasets as-is
    logm("Applying preprocessing (simulated: datasets passed through unchanged)")
    preprocessed_datasets = datasets  # In real use: apply corrections here
    original_datasets = list(datasets)  # identity refs, to detect in-place mutation
    original_velocities = [ds["VRADH"].values.copy() for ds in datasets]

    # Step 3: Apply dealiasing to pre-loaded datasets
    with timed("Dealiasing pre-loaded datasets (strategy=long_range, alpha=0.6)"):
        dealiased_datasets = unravel.unravel_3D_pyodim(
            preprocessed_datasets,
            vel_name="VRADH",
            output_vel_name="velocity_dealias",
            strategy="long_range",
            alpha=0.6,
        )

    assert dealiased_datasets is not None, "Returned datasets is None"
    assert isinstance(dealiased_datasets, list), "Returned object is not a list"
    assert len(dealiased_datasets) == len(datasets), "Number of output datasets doesn't match input"

    # Check that dealiased fields were added
    for idx, ds in enumerate(dealiased_datasets):
        assert "velocity_dealias" in ds, f"Dealiased velocity not found in sweep {idx}"
        assert "velocity_dealias_flag" in ds, f"Flag field not found in sweep {idx}"

    log_odim_volume(dealiased_datasets, "VRADH", "velocity_dealias", "velocity_dealias_flag")

    # Verify the caller's list was NOT mutated: a new list of new datasets
    # is returned, and the input sweeps gained no field.
    logm("Checking the pre-loaded datasets were left untouched")
    assert dealiased_datasets is not datasets, "Returned list is the caller's list"
    assert datasets == original_datasets, "Caller's list had its elements replaced"
    for idx, dataset in enumerate(datasets):
        assert "velocity_dealias" not in dataset, f"Input sweep {idx} was modified in place"
        assert "velocity_dealias_flag" not in dataset, f"Input sweep {idx} was modified in place"

    # Verify the input velocity field was NOT modified in place
    for idx, (dataset, original_vel) in enumerate(zip(dealiased_datasets, original_velocities)):
        np.testing.assert_array_equal(
            dataset["VRADH"].values,
            original_vel,
            err_msg=f"Original velocity field was modified in place in sweep {idx}",
        )
    logm(f"Input list, sweeps and VRADH arrays unchanged across all {len(datasets)} sweeps")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pyodim_with_condition():
    """Test pyodim dealiasing with data filtering condition."""
    try:
        import pyodim
    except ImportError:
        pytest.skip("pyodim not installed")

    test_file = get_odim_test_file()

    # Apply dealiasing with a reflectivity threshold condition
    with timed("Dealiasing with condition DBZH < 10 dBZ (strategy=default, alpha=0.6)"):
        datasets = unravel.unravel_3D_pyodim(
            test_file,
            vel_name="VRADH",
            output_vel_name="unraveled_velocity",
            load_all_fields=True,  # Need to load reflectivity for condition
            condition=("DBZH", "lower", 10.0),  # Filter out weak echoes
            strategy="default",
            alpha=0.6,
        )

    assert datasets is not None, "Returned datasets is None"
    assert len(datasets) > 0, "No datasets returned"

    # Check that condition created a cleaned field
    first_ds = datasets[0]
    assert "unraveled_velocity" in first_ds, "Dealiased velocity field not found"

    # The condition masks weak echoes out of VRADH_clean, so coverage here should
    # be lower than the unfiltered run in test_pyodim_from_file.
    log_odim_volume(datasets, "VRADH_clean", "unraveled_velocity", "unraveled_velocity_flag")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
