import pyart
import pytest
import unravel


def test_dealias():
    try:
        radar = pyart.aux_io.read_odim_h5("E:/Jupyter/OceanPOL/20191122/9776HUB-PPIVol-20191122-114200-0000.hdf", file_field_names=True)
        vel = unravel.unravel_3D_pyart(radar, "VRAD", "DBZH", nyquist_velocity=13.3)
        assert True
    except Exception as e:
        pytest.fail(f"dealias function failed with error: {e}")
