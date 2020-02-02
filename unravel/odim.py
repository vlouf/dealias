"""
ODIM convention file reader. Natively reads ODIM H5 files.

@title: odim.py
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 02/02/2020

.. autosummary::
    :toctree: generated/

    _to_str
    field_metadata
    cartesian_to_geographic
    radar_coordinates_to_xyz
    generate_timestamp
    get_root_metadata
    coord_from_metadata
    get_dataset_metadata
    check_nyquist
    read_odim_slice
    read_odim
"""
import datetime

import dask
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


def coord_from_metadata(metadata):
    '''
    Create the radar coordinates from the ODIM H5 metadata specification.
    Parameter:
    ==========
    metadata: dict()
        Metadata dictionnary containing the specific ODIM H5 keys: astart,
        nrays, nbins, rstart, rscale, elangle.
    Returns:
    ========
    r: ndarray<nbins>
        Sweep range
    azimuth: ndarray<nrays>
        Sweep azimuth
    elev: float
        Sweep elevation
    '''
    da = 360 / metadata['nrays']
    azimuth = np.linspace(metadata['astart'] + da / 2,
                          360 - da,
                          metadata['nrays'], dtype=np.float32)

    # rstart is in KM !!! STUPID.
    rstart_center = 1e3 * metadata['rstart'] + metadata['rscale'] / 2
    r = np.arange(rstart_center,
                  rstart_center + metadata['nbins'] * metadata['rscale'],
                  metadata['rscale'], dtype=np.float32)

    elev = np.array([metadata['elangle']], dtype=np.float32)
    return r, azimuth, elev


def get_dataset_metadata(hfile, dataset='dataset1'):
    '''
    Get the dataset metadata of the ODIM H5 file.
    Parameters:
    ===========
    hfile: h5py.File
        H5 file identifier.
    dataset: str
        Key of the dataset for which to extract the metadata
    Returns:
    ========
    metadata: dict
        General metadata of the dataset.
    coordinates_metadata: dict
        Coordinates-specific metadata.
    '''
    metadata = dict()
    coordinates_metadata = dict()
    # General metadata
    metadata['NI'] = hfile[f'/{dataset}/how'].attrs['NI']
    metadata['highprf'] = hfile[f'/{dataset}/how'].attrs['highprf']
    metadata['product'] = _to_str(hfile[f'/{dataset}/what'].attrs['product'])

    sdate = _to_str(hfile[f'/{dataset}/what'].attrs['startdate'])
    stime = _to_str(hfile[f'/{dataset}/what'].attrs['starttime'])
    edate = _to_str(hfile[f'/{dataset}/what'].attrs['enddate'])
    etime = _to_str(hfile[f'/{dataset}/what'].attrs['endtime'])
    metadata['start_time'] = f'{sdate}_{stime}'
    metadata['end_time'] = f'{edate}_{etime}'

    # Coordinates:
    try:
        coordinates_metadata['astart'] = hfile[f'/{dataset}/how'].attrs['astart']
    except KeyError:
        # Optional coordinates (!).
        coordinates_metadata['astart'] = 0
    coordinates_metadata['a1gate'] = hfile[f'/{dataset}/where'].attrs['a1gate']
    coordinates_metadata['nrays'] = hfile[f'/{dataset}/where'].attrs['nrays']

    coordinates_metadata['rstart'] = hfile[f'/{dataset}/where'].attrs['rstart']
    coordinates_metadata['rscale'] = hfile[f'/{dataset}/where'].attrs['rscale']
    coordinates_metadata['nbins'] = hfile[f'/{dataset}/where'].attrs['nbins']

    coordinates_metadata['elangle'] = hfile[f'/{dataset}/where'].attrs['elangle']

    return metadata, coordinates_metadata


def check_nyquist(dset):
    '''
    Check if the dataset Nyquist velocity corresponds to the PRF information.
    '''
    wavelength = dset.attrs['wavelength']
    prf = dset.attrs['highprf']
    nyquist = dset.attrs['NI']
    ny_int = 1e-2 * prf * wavelength / 4

    assert np.abs(nyquist - ny_int) < 0.5, 'Nyquist not consistent with PRF'


def read_odim_slice(odim_file, nslice=0, include_fields=[], exclude_fields=[]):
    '''
    Read into an xarray dataset one sweep of the ODIM file.
    Parameters:
    ===========
    odim_file: str
        ODIM H5 filename.
    nslice: int
        Slice number we want to extract (start indexing at 0).
    include_fields: list
        Specific fields to be exclusively read.
    exclude_fields: list
        Specific fields to be excluded from reading.
    Returns:
    ========
    dataset: xarray.Dataset
        xarray dataset of one sweep of the ODIM file.
    '''
    if nslice == 0:
        raise ValueError('Slice numbering start at 1.')
    if type(include_fields) is not list:
        raise TypeError('Argument `include_fields` should be a list')

    with h5py.File(odim_file) as hfile:
        # Number of sweep in dataset
        nsweep = len([k for k in hfile['/'].keys() if k.startswith('dataset')])
        assert nslice <= nsweep, f"Wrong slice number asked. Only {nsweep} available."

        # Order datasets by increasing elevations.
        sweeps = dict()
        for key in hfile['/'].keys():
            if key.startswith('dataset'):
                sweeps[key] = hfile[f'/{key}/where'].attrs['elangle']

        sorted_keys = sorted(sweeps, key=lambda k: sweeps[k])
        rootkey = sorted_keys[nslice]

        # Retrieve dataset metadata and coordinates metadata.
        metadata, coordinates_metadata = get_dataset_metadata(hfile, rootkey)

        dataset = xr.Dataset()
        dataset.attrs = get_root_metadata(hfile)
        dataset.attrs.update(metadata)
        check_nyquist(dataset)
        for datakey in hfile[f'/{rootkey}'].keys():
            if not datakey.startswith('data'):
                continue

            gain = hfile[f'/{rootkey}/{datakey}/what'].attrs['gain']
            nodata = hfile[f'/{rootkey}/{datakey}/what'].attrs['nodata']
            offset = hfile[f'/{rootkey}/{datakey}/what'].attrs['offset']
            name = _to_str(hfile[f'/{rootkey}/{datakey}/what'].attrs['quantity'])
            # Check if field should be read.
            if len(include_fields) > 0:
                if name not in include_fields:
                    continue
            if name in exclude_fields:
                continue

            data_value = hfile[f'/{rootkey}/{datakey}/data'][:].astype(np.float32)
            data_value = gain * np.ma.masked_equal(data_value, nodata) + offset
            dataset = dataset.merge({name: (('azimuth', 'range'), data_value)})
            dataset[name].attrs = field_metadata(name)

    time = generate_timestamp(metadata['start_time'],
                              metadata['end_time'],
                              coordinates_metadata['nrays'],
                              coordinates_metadata['a1gate'])
    r, azi, elev = coord_from_metadata(coordinates_metadata)
    x, y, z = radar_coordinates_to_xyz(r, azi, elev)
    longitude, latitude = cartesian_to_geographic(x, y,
                                                  dataset.attrs['longitude'],
                                                  dataset.attrs['latitude'])

    dataset = dataset.merge({'range': (('range'), r),
                             'azimuth': (('azimuth'), azi),
                             'elevation': (('elevation'), elev),
                             'time': (('time'), time),
                             'x': (('azimuth', 'range'), x),
                             'y': (('azimuth', 'range'), y),
                             'z': (('azimuth', 'range'), z + dataset.attrs['height']),
                             'longitude': (('azimuth', 'range'), longitude),
                             'latitude': (('azimuth', 'range'), latitude)})

    return dataset


def read_odim(odim_file, lazy_load=True, **kwargs):
    '''
    Read an ODIM H5 file.
    Parameters:
    ===========
    odim_file: str
        ODIM H5 filename.
    lazy_load: bool
        Lazily load the data if true, read and load in memory the entire dataset
        if false.
    include_fields: list
        Specific fields to be exclusively read.
    exclude_fields: list
        Specific fields to be excluded from reading.
    Returns:
    ========
    radar: list
        List of xarray datasets, each item in a the list is one sweep of the
        radar data (ordered from lowest elevation scan to highest).
    '''
    with h5py.File(odim_file) as hfile:
        nsweep = len([k for k in hfile['/'].keys() if k.startswith('dataset')])

    radar = []
    for sweep in range(1, nsweep + 1):
        c = dask.delayed(read_odim_slice)(odim_file, sweep, **kwargs)
        radar.append(c)

    if not lazy_load:
        radar = [r.compute() for r in radar]

    return radar
