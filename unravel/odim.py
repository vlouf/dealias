"""
ODIM convention file reader. Natively reads ODIM H5 files.

@title: odim.py
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 02/02/2020

.. autosummary::
    :toctree: generated/

    
"""
import datetime

import h5py
import pyproj
import pandas as pd
import numpy as np
import xarray as xr


def _to_str(t):
    '''
    Transform binary into string.
    '''
    return t.decode('utf-8')


def field_metadata(quantity_name):
    '''
    Populate metadata for common fields using Py-ART get_metadata() function.
    (Optionnal).
    Parameter:
    ==========
    quantity_name: str
        ODIM H5 quantity attribute name.
    Returns:
    ========
    attrs: dict()
        Metadata dictionnary.
    '''
    try:
        from pyart.config import get_metadata
    except Exception:
        return {}
    ODIM_H5_FIELD_NAMES = {'TH': 'total_power',  # uncorrected reflectivity, horizontal
                           'TV': 'total_power',  # uncorrected reflectivity, vertical
                           'DBZH': 'reflectivity',  # corrected reflectivity, horizontal
                           'DBZH_CLEAN': 'reflectivity',  # corrected reflectivity, horizontal
                           'DBZV': 'reflectivity',  # corrected reflectivity, vertical
                           'ZDR': 'differential_reflectivity',  # differential reflectivity
                           'RHOHV': 'cross_correlation_ratio',
                           'LDR': 'linear_polarization_ratio',
                           'PHIDP': 'differential_phase',
                           'KDP': 'specific_differential_phase',
                           'SQI': 'normalized_coherent_power',
                           'SNR': 'signal_to_noise_ratio',
                           'SNRH': 'signal_to_noise_ratio',
                           'VRAD': 'velocity', # radial velocity, marked for deprecation in ODIM HDF5 2.2
                           'VRADH': 'velocity', # radial velocity, horizontal polarisation
                           'VRADDH': 'corrected_velocity', # radial velocity, horizontal polarisation
                           'VRADV': 'velocity', # radial velocity, vertical polarisation
                           'WRAD': 'spectrum_width',
                           'QIND': 'quality_index',}

    try:
        fname = ODIM_H5_FIELD_NAMES[quantity_name]
        attrs = get_metadata(fname)
        attrs.pop('coordinates')
    except KeyError:
        return {}

    return attrs


def cartesian_to_geographic(x, y, lon0, lat0):
    '''
    Transform cartesian coordinates to lat/lon using the Azimuth Equidistant
    projection.
    Parameters:
    ===========
    x: ndarray
        x-axis cartesian coordinates.
    y: ndarray
        y-axis cartesian coordinates. Same dimension as x
    lon0: float
        Radar site longitude.
    lat0: float
        Radar site latitude.
    Returns:
    lon: ndarray
        Longitude of each gate.
    lat: ndarray
        Latitude of each gate.
    '''
    georef = pyproj.Proj(f"+proj=aeqd +lon_0={lon0} +lat_0={lat0} +ellps=WGS84")
    lon, lat = georef(x, y, inverse=True)
    lon = lon.astype(np.float32)
    lat = lat.astype(np.float32)
    return lon, lat


def radar_coordinates_to_xyz(r, azimuth, elevation):
    '''
    Transform radar coordinates to cartesian coordinates.
    Parameters:
    ===========
    r: ndarray<nbins>
        Sweep range.
    azimuth: ndarray<nrays>
        Sweep azimuth.
    elevation: float
        Sweep elevation.
    Returns:
    ========
    x, y, z: ndarray<nrays, nbins>
        XYZ cartesian coordinates.
    '''
    # To proper spherical coordinates.
    theta = np.deg2rad(90 - elevation)
    phi = 450 - azimuth
    phi[phi >= 360] -= 360
    phi = np.deg2rad(phi)

    R, PHI = np.meshgrid(r, phi)
    R = R.astype(np.float32)
    PHI = PHI.astype(np.float32)

    x = R * np.sin(theta) * np.cos(PHI)
    y = R * np.sin(theta) * np.sin(PHI)
    z = R * np.cos(theta)

    return x, y, z


def generate_timestamp(stime, etime, nrays, a1gate):
    '''
    Generate timestamp for each ray.
    Parameters:
    ===========
    stime: str
        Sweep starting time.
    etime:
        Sweep ending time.
    nrays: int
        Number of rays in sweep.
    a1gate: int
        Azimuth of the ray measured first by the radar.
    Returns:
    ========
    trange: Timestamp<nrays>
        Timestamp for each ray.
    '''
    sdtime = datetime.datetime.strptime(stime, '%Y%m%d_%H%M%S')
    edtime = datetime.datetime.strptime(etime, '%Y%m%d_%H%M%S')
    trange = pd.date_range(sdtime, edtime, nrays)

    return np.roll(trange, a1gate)


def get_root_metadata(hfile):
    '''
    Get the metadata at the root of the ODIM H5 file.
    Parameters:
    ===========
    hfile: h5py.File
        H5 file identifier.
    Returns:
    ========
    rootmetadata: dict
        Metadata at the root of the ODIM H5 file.
    '''
    rootmetadata = {}
    # Root
    rootmetadata['Conventions'] = _to_str(hfile.attrs['Conventions'])

    # Where
    rootmetadata['latitude'] = hfile['/where'].attrs['lat']
    rootmetadata['longitude'] = hfile['/where'].attrs['lon']
    rootmetadata['height'] = hfile['/where'].attrs['height']

    # What
    sdate = hfile['/what'].attrs['date'].decode('utf-8')
    stime = hfile['/what'].attrs['time'].decode('utf-8')
    rootmetadata['date'] = datetime.datetime.strptime(sdate + stime, '%Y%m%d%H%M%S').isoformat()
    rootmetadata['object'] = _to_str(hfile['/what'].attrs['object'])
    rootmetadata['source'] = _to_str(hfile['/what'].attrs['source'])
    rootmetadata['version'] = _to_str(hfile['/what'].attrs['version'])

    # How
    try:
        rootmetadata['beamwH'] = hfile['/how'].attrs['beamwH']
        rootmetadata['beamwV'] = hfile['/how'].attrs['beamwV']
    except Exception:
        pass
    rootmetadata['copyright'] = _to_str(hfile['/how'].attrs['copyright'])
    rootmetadata['rpm'] = hfile['/how'].attrs['rpm']
    rootmetadata['wavelength'] = hfile['/how'].attrs['wavelength']

    return rootmetadata
