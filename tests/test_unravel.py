import os
import datetime
import pytest
import requests
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    os.environ["PYART_QUIET"] = "1"  # Disable Py-ART disclaimer
    import pyart
import unravel


def logm(message: str):
    """
    Log message to console.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"{date} - {message}"
    print(message)


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

    r = requests.get(url)
    try:
        r.raise_for_status()
    except Exception:
        raise ValueError("No file found for this date. CPOL ran from 1998-12-6 to 2017-5-2, wet season only. Try another date.")

    with open(outfilename, "wb") as fid:
        fid.write(r.content)

    return outfilename


@pytest.mark.filterwarnings("ignore:.*CfRadial module is deprecated.*:UserWarning")
def test_pyart():
    date = datetime.datetime(2014, 2, 18, 20, 0)
    logm("Downloading data")
    filename = download_cpol_data(date)
    logm("Test data downloaded")

    try:
        logm("Reading data with Py-ART")
        radar = pyart.io.read(filename)
        logm("Data read successfully")
        assert isinstance(radar, pyart.core.Radar), "Radar object not created successfully"
        logm("Starting dealiasing process")
        # Run the dealiasing process
        vel = unravel.unravel_3D_pyart(radar, "velocity", "corrected_reflectivity", nyquist_velocity=13.3)
        logm("Dealiasing process completed")
        assert vel is not None, "Dealiased velocity field is None"
        assert True
    except Exception as e:
        pytest.fail(f"Dealias function failed with error: {e}")
