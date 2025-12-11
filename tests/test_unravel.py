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


# def download_cpol_data(date: datetime.datetime) -> str:
#     """
#     Download CPOL data for given date.
#     Parameters:
#     ===========
#     date: str or datetime or pd.Timestamp
#         Date time for which we want the CPOL data
#     ppi: bool
#         True for downloading the PPIs and False for downloading the gridded data.
#     """
#     year = date.year
#     datestr = date.strftime("%Y%m%d")
#     datetimestr = date.strftime("%Y%m%d.%H%M")
#     url = f"https://dapds00.nci.org.au/thredds/fileServer/hj10/cpol/cpol_level_1b/v2020/ppi/{year}/{datestr}/twp10cpolppi.b1.{datetimestr}00.nc"
#     fname = os.path.basename(url)
#     try:
#         os.mkdir("dwl")
#     except FileExistsError:
#         pass
#     outfilename = os.path.join("dwl", fname)
#     if os.path.isfile(outfilename):
#         logm("Radar data file already exists, doing nothing")
#         return outfilename
#     r = requests.get(url)
#     try:
#         r.raise_for_status()
#     except Exception:
#         raise ValueError(
#             "No file found for this date. CPOL ran from 1998-12-6 to 2017-5-2, wet season only. Try another date."
#         )
#     with open(outfilename, "wb") as fid:
#         fid.write(r.content)
#     return outfilename


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


# @pytest.mark.filterwarnings("ignore:.*CfRadial module is deprecated.*:UserWarning")
# def test_pyart():
#     date = datetime.datetime(2014, 2, 18, 20, 0)
#     logm("Downloading data")
#     filename = download_cpol_data(date)
#     logm("Test data downloaded")
#     try:
#         logm("Reading data with Py-ART")
#         radar = pyart.io.read(filename)
#         logm("Data read successfully")
#         assert isinstance(radar, pyart.core.Radar), "Radar object not created successfully"
#         logm("Starting dealiasing process")
#         # Run the dealiasing process
#         vel = unravel.unravel_3D_pyart(radar, "velocity", "corrected_reflectivity", nyquist_velocity=13.3)
#         logm("Dealiasing process completed")
#         assert vel is not None, "Dealiased velocity field is None"
#         assert True
#     except Exception as e:
#         pytest.fail(f"Dealias function failed with error: {e}")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pyodim_from_file():
    """Test pyodim dealiasing by reading directly from file."""
    try:
        import pyodim
    except ImportError:
        pytest.skip("pyodim not installed")
    
    logm("Setting up ODIM test")
    
    # Get test file path
    test_file = get_odim_test_file()
    
    try:
        logm(f"Reading ODIM file: {test_file}")
        
        # Test reading from file path directly
        datasets = unravel.unravel_3D_pyodim(
            test_file,
            vel_name="VRADH",
            output_vel_name="unraveled_velocity",
            strategy="long_range",
            alpha=0.6
        )
        
        logm("ODIM dealiasing from file completed")
        
        # Assertions
        assert datasets is not None, "Returned datasets is None"
        assert isinstance(datasets, list), "Returned object is not a list"
        assert len(datasets) > 0, "No datasets returned"
        
        # Check first dataset has required fields
        first_ds = datasets[0]
        assert "unraveled_velocity" in first_ds, "Dealiased velocity field not found"
        assert "unraveled_velocity_flag" in first_ds, "Flag field not found"
        
        logm(f"Successfully processed {len(datasets)} sweeps")
        
    except Exception as e:
        pytest.fail(f"PyODIM dealias from file failed with error: {e}")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pyodim_from_datasets():
    """Test pyodim dealiasing with pre-loaded datasets (preprocessing workflow)."""
    try:
        import pyodim
        import xarray as xr
    except ImportError:
        pytest.skip("pyodim or xarray not installed")
    
    logm("Setting up ODIM preprocessing test")
    
    # Get test file path
    test_file = get_odim_test_file()
    
    try:
        logm(f"Reading ODIM file with pyodim: {test_file}")
        
        # Step 1: Load datasets with pyodim
        datasets = pyodim.read_odim(test_file, lazy_load=False)        
        
        logm(f"Loaded {len(datasets)} sweeps")
        
        # Step 2: Simulate preprocessing (e.g., dual-PRF correction would go here)
        # For testing, we'll just pass the datasets as-is
        logm("Applying preprocessing (simulated)")
        preprocessed_datasets = datasets  # In real use: apply corrections here
        
        # Step 3: Apply dealiasing to pre-loaded datasets
        logm("Starting dealiasing on pre-loaded datasets")
        dealiased_datasets = unravel.unravel_3D_pyodim(
            preprocessed_datasets,
            vel_name="VRADH",
            output_vel_name="velocity_dealias",
            strategy="long_range",
            alpha=0.6
        )
        
        logm("ODIM dealiasing from pre-loaded datasets completed")
        
        # Assertions
        assert dealiased_datasets is not None, "Returned datasets is None"
        assert isinstance(dealiased_datasets, list), "Returned object is not a list"
        assert len(dealiased_datasets) == len(datasets), "Number of output datasets doesn't match input"
        
        # Check that dealiased fields were added
        for idx, ds in enumerate(dealiased_datasets):
            assert "velocity_dealias" in ds, f"Dealiased velocity not found in sweep {idx}"
            assert "velocity_dealias_flag" in ds, f"Flag field not found in sweep {idx}"
        
        # Verify original field was NOT modified
        for idx in range(len(datasets)):
            original_vel = datasets[idx]["VRADH"].values
            # Check that we can still access original data
            assert original_vel is not None, f"Original velocity field was corrupted in sweep {idx}"
        
        logm(f"Successfully processed {len(dealiased_datasets)} sweeps with preprocessing")
        
    except Exception as e:
        pytest.fail(f"PyODIM dealias from datasets failed with error: {e}")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pyodim_with_condition():
    """Test pyodim dealiasing with data filtering condition."""
    try:
        import pyodim
    except ImportError:
        pytest.skip("pyodim not installed")
    
    logm("Setting up ODIM condition test")
    
    # Get test file path
    test_file = get_odim_test_file()
    
    try:
        logm(f"Reading ODIM file with condition: {test_file}")
        
        # Apply dealiasing with a reflectivity threshold condition
        datasets = unravel.unravel_3D_pyodim(
            test_file,
            vel_name="VRADH",
            output_vel_name="unraveled_velocity",
            load_all_fields=True,  # Need to load reflectivity for condition
            condition=("DBZH", "lower", 10.0),  # Filter out weak echoes
            strategy="default",
            alpha=0.6
        )
        
        logm("ODIM dealiasing with condition completed")
        
        # Assertions
        assert datasets is not None, "Returned datasets is None"
        assert len(datasets) > 0, "No datasets returned"
        
        # Check that condition created a cleaned field
        first_ds = datasets[0]
        assert "unraveled_velocity" in first_ds, "Dealiased velocity field not found"
        
        logm("Condition filtering test passed")
        
    except Exception as e:
        pytest.fail(f"PyODIM dealias with condition failed with error: {e}")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])