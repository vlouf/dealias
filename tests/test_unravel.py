# STD lib
import os
import datetime
import requests

# Unravel Lib
import pyart
import unravel

# Unit-test lib
import pytest


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
    url = f"http://dapds00.nci.org.au/thredds/fileServer/hj10/cpol/cpol_level_1b/v2020/ppi/{year}/{datestr}/twp10cpolppi.b1.{datetimestr}00.nc"
    fname = os.path.basename(url)
    try:
        os.mkdir("dwl")
    except FileExistsError:
        pass
    outfilename = os.path.join("dwl", fname)
    if os.path.isfile(outfilename):
        print("Output file already exists, doing nothing")
        return outfilename

    r = requests.get(url)
    try:
        r.raise_for_status()
    except Exception:
        print("No file found for this date. CPOL ran from 1998-12-6 to 2017-5-2, wet season only. Try another date.")
        return None

    with open(outfilename, "wb") as fid:
        fid.write(r.content)

    return outfilename


def test_dealias():
    date = datetime.datetime(2014, 2, 18, 20, 0)
    filename = download_cpol_data(date)

    try:
        radar = pyart.io.read(filename)
        vel = unravel.unravel_3D_pyart(radar, "velocity", "corrected_reflectivity", nyquist_velocity=13.3)
        assert True
    except Exception as e:
        pytest.fail(f"Dealias function failed with error: {e}")
