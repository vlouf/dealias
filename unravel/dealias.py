"""
Driver script for the dealiasing module.
TODO: Implement native ODIM H5 file reader.
TODO: Implement new scan strategy with different PRF (Nyquist) for each elev.

@title: dealias
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 05/04/2018
@date: 28/01/2020

    dealiasing_process_2D
    dealias_long_range
    unravel_3D_pyart
"""
# Python Standard Library
import traceback

# Other python libraries.
import numpy as np

# Local
from . import continuity
from . import filtering
from . import initialisation
from . import find_reference
from .core import Dealias


def dealiasing_process_2D(r, azimuth, elevation, velocity, nyquist_velocity, alpha=0.6):
    """
    Dealiasing processing for 2D slice of the Doppler radar velocity field.

    Parameters:
    ===========
    r: ndarray
        Radar scan range.
    azimuth: ndarray
        Radar scan azimuth.
    elevation: float
        Elevation angle of the velocity field.
    velocity: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    nyquist_velocity: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array (-3: No data, 0: Unprocessed, 1: Processed - no change -,
                    2: Processed - dealiased.)
    """
    if not np.isscalar(elevation):
        raise TypeError('Elevation should be scalar, not an array.')
    if velocity.shape != (len(azimuth), len(r)):
        raise ValueError('The dimensions of the velocity field should be <azimuth, range>.')
    dtype = velocity.dtype

    dealias_2D = Dealias(r, azimuth, elevation, velocity, nyquist_velocity, alpha)

    # Initialization
    dealias_2D.initialize()

    # Dealiasing modules
    dealias_2D.correct_range()
    for window in [6, 12]:
        dealias_2D.correct_range(window)
        dealias_2D.correct_clock(window)
        if dealias_2D.check_completed():
            break

    if not dealias_2D.check_completed():
        for window in [(5, 2), (20, 10), (40, 20), (80, 40)]:
            dealias_2D.correct_box(window)
            if dealias_2D.check_completed():
                break

    if not dealias_2D.check_completed():
        dealias_2D.correct_leastsquare()

    if not dealias_2D.check_completed():
        dealias_2D.correct_linregress()

    if not dealias_2D.check_completed():
        dealias_2D.correct_closest()

    # Checking modules.
    dealias_2D.check_leastsquare()
    dealias_2D.check_box()

    unfold_vel = np.ma.masked_where(dealias_2D.flag == -3, dealias_2D.dealias_vel)

    return unfold_vel.astype(dtype), dealias_2D.flag


def dealias_long_range(r, azimuth, elevation, velocity, nyquist_velocity, alpha=0.6):
    """
    Dealiasing processing for 2D slice of the Doppler radar velocity field.

    Parameters:
    ===========
    r: ndarray
        Radar scan range.
    azimuth: ndarray
        Radar scan azimuth.
    elevation: float
        Elevation angle of the velocity field.
    velocity: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    nyquist_velocity: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array (-3: No data, 0: Unprocessed, 1: Processed - no change -,
                    2: Processed - dealiased.)
    """
    if not np.isscalar(elevation):
        raise TypeError('Elevation should be scalar, not an array.')
    if velocity.shape != (len(azimuth), len(r)):
        raise ValueError('The dimensions of the velocity field should be <azimuth, range>.')
    dtype = velocity.dtype

    dealias_2D = Dealias(r, azimuth, elevation, velocity, nyquist_velocity, alpha)

    dealias_2D.initialize()
    dealias_2D.correct_range()
    for window in [6, 12, 24, 48, 96]:
        dealias_2D.correct_range(window)
        dealias_2D.correct_clock(window)
        if dealias_2D.check_completed():
            break

    if not dealias_2D.check_completed():
        for window in [(20, 20), (40, 40), (80, 80)]:
            dealias_2D.correct_box(window)
            if dealias_2D.check_completed():
                break

    if not dealias_2D.check_completed():
        dealias_2D.correct_linregress()

    if not dealias_2D.check_completed():
        dealias_2D.correct_closest()

    dealias_2D.check_box()

    unfold_vel = np.ma.masked_where(dealias_2D.flag == -3, dealias_2D.dealias_vel)

    return unfold_vel.astype(dtype), dealias_2D.flag


def unravel_3D_pyart(radar,
                     velname="VEL",
                     dbzname="DBZ",
                     gatefilter=None,
                     nyquist_velocity=None,
                     strategy='default',
                     **kwargs):
    """
    Process driver.
    Full dealiasing process 2D + 3D.

    Parameters:
    ===========
    radar: PyART Radar Object
        Py-ART radar object.
    gatefilter: Object GateFilter
        GateFilter for filtering noise. If not provided it will automaticaly
        compute it with help of the dual-polar variables.
    velname: str
        Name of the velocity field.
    dbzname: str
        Name of the reflectivity field.
    strategy: ['default', 'long_range']
        Using the default dealiasing strategy or the long range strategy.

    Returns:
    ========
    unraveled_velocity: ndarray
        Dealised velocity field.
    """
    # Check arguments
    if strategy not in ['default', 'long_range']:
        raise ValueError("Dealiasing strategy not understood please choose 'default' or 'long_range'")
    if gatefilter is None:
        gatefilter = filtering.do_gatefilter(radar, velname, dbzname)
    if nyquist_velocity is None:
        nyquist_velocity = radar.instrument_parameters['nyquist_velocity']['data'][0]
        if nyquist_velocity is None:
            raise ValueError('Nyquist velocity not found.')

    # Read the velocity field.
    try:
        velocity = radar.fields[velname]['data'].filled(np.NaN)
    except Exception:
        velocity = radar.fields[velname]['data']
    velocity[gatefilter.gate_excluded] = np.NaN

    # Read coordinates and start with the first sweep.
    sweep = radar.get_slice(0)
    r = radar.range['data']
    azimuth_reference = radar.azimuth['data'][sweep]
    elevation_reference = radar.elevation['data'][sweep].mean()
    velocity_reference = velocity[sweep]

    # Dealiasing first sweep.
    if strategy == 'default':
        final_vel, flag_vel = dealiasing_process_2D(r,
                                                    azimuth_reference,
                                                    velocity_reference,
                                                    elevation_reference,
                                                    nyquist_velocity,
                                                    **kwargs)
    else:
        final_vel, flag_vel = dealias_long_range(r,
                                                 azimuth_reference,
                                                 elevation_reference,
                                                 velocity_reference,
                                                 nyquist_velocity,
                                                 **kwargs)

    velocity_reference, flag_reference = final_vel.copy(), flag_vel.copy()
    unraveled_velocity = np.zeros(radar.fields[velname]['data'].shape)
    unraveled_velocity[sweep] = final_vel.copy()

    for slice_number in range(1, radar.nsweeps):
        sweep = radar.get_slice(slice_number)
        azimuth_slice = radar.azimuth['data'][sweep]
        elevation_slice = radar.elevation['data'][sweep].mean()

        vel = np.ma.masked_where(gatefilter.gate_excluded, velocity)[sweep]
        velocity_slice = vel.filled(np.NaN)

        flag_slice = np.zeros_like(velocity_slice) + 1
        flag_slice[np.isnan(velocity_slice)] = -3

        if strategy == 'default':
            final_vel, flag_vel = dealiasing_process_2D(r,
                                                        azimuth_slice,
                                                        velocity_slice,
                                                        elevation_slice,
                                                        nyquist_velocity,
                                                        **kwargs)
        else:
            final_vel, flag_vel = dealias_long_range(r,
                                                     azimuth_slice,
                                                     elevation_slice,
                                                     velocity_slice,
                                                     nyquist_velocity,
                                                     **kwargs)

        final_vel = final_vel.filled(np.NaN)
        final_vel, flag_slice, _, _ = continuity.unfolding_3D(r,
                                                              azimuth_reference,
                                                              elevation_reference,
                                                              velocity_reference,
                                                              flag_reference,
                                                              r,
                                                              azimuth_slice,
                                                              elevation_slice,
                                                              final_vel,
                                                              flag_vel,
                                                              velocity[sweep],
                                                              nyquist_velocity)

        final_vel, flag_slice = continuity.box_check(azimuth_slice,
                                                     final_vel,
                                                     flag_slice,
                                                     nyquist_velocity,
                                                     window_range=250,
                                                     **kwargs)

        azimuth_reference = azimuth_slice.copy()
        velocity_reference = final_vel.copy()
        flag_reference = flag_vel.copy()
        elevation_reference = elevation_slice
        unraveled_velocity[sweep] = final_vel.copy()

    unraveled_velocity = np.ma.masked_invalid(unraveled_velocity)
    return unraveled_velocity


def unravel_3D_pyodim(odim_file,
                      vel_name='VRADH',
                      output_vel_name='unraveled_velocity',
                      gatefilter=None,
                      strategy='long_range',
                      **kwargs):
    '''
    Support for ODIM H5 files and Nyquist changing with the elevation. The new
    scan strategy is using single-PRF, to avoid dual-PRF artifacts, but with
    a PRF (and thus Nyquist velocity) that changes at each sweep.

    Parameters:
    ===========
    odim_file: str
        ODIM H5 file name.
    vel_name: str
        Velocity field name.

    Returns:
    ========
    radar_datasets: List
        List of xarray datasets. PyODIM output data model.
    '''
    # NOTE: This function is made to handle a variable PRF, and thus a variable
    # Nyquist. We use the sweeps with the highest Nyquist in the lowest
    # elevation scans as reference and we dealise in 3D, down and up from that
    # sweep.
    if strategy not in ['default', 'long_range']:
        raise ValueError("Dealiasing strategy not understood please choose 'default' or 'long_range'")
    if gatefilter is not None:
        raise ValueError('gatefilter not supported with pyodim structure. Please use Py-ART instead.')

    import pyodim
    radar_datasets = pyodim.read_odim(odim_file, include_fields=[vel_name])
    radar_datasets = [r.compute() for r in radar_datasets]
    elev_angles = [r['elevation'].values[0] for r in radar_datasets]
    nyquists = [r.attrs['NI'] for r in radar_datasets]

    # Looking for low-elevation sweep with the highest Nyquist velocity to use
    # as reference.
    nslice_ref = np.argmax(nyquists[:len(elev_angles) // 2])

    r_reference = radar_datasets[nslice_ref].range.values
    azimuth_reference = radar_datasets[nslice_ref].azimuth.values
    velocity_reference = radar_datasets[nslice_ref][vel_name].values
    elevation_reference = radar_datasets[nslice_ref]['elevation'].values[0]
    nyquist_velocity = radar_datasets[nslice_ref].attrs['NI']

    # Dealiasing first sweep.
    if strategy == 'default':
        final_vel, flag_vel = dealiasing_process_2D(r_reference,
                                                    azimuth_reference,
                                                    velocity_reference,
                                                    elevation_reference,
                                                    nyquist_velocity,
                                                    **kwargs)
    else:
        final_vel, flag_vel = dealias_long_range(r_reference,
                                                 azimuth_reference,
                                                 elevation_reference,
                                                 velocity_reference,
                                                 nyquist_velocity,
                                                 **kwargs)
    velocity_reference, flag_reference = final_vel.copy(), flag_vel.copy()
    radar_datasets[nslice_ref] = radar_datasets[nslice_ref].merge({output_vel_name: (('azimuth', 'range'),
                                                                                     velocity_reference)})

    # Processing sweeps by decreasing elevations from the nslice_ref sweeps
    if nslice_ref != 0:
        for sweep in np.arange(nslice_ref)[::-1]:
            r_slice = radar_datasets[sweep].range.values
            azimuth_slice = radar_datasets[sweep].azimuth.values
            velocity_slice = radar_datasets[sweep][vel_name].values
            elevation_slice = radar_datasets[sweep]['elevation'].values[0]
            nyquist_velocity = radar_datasets[sweep].attrs['NI']

            if strategy == 'default':
                final_vel, flag_vel = dealiasing_process_2D(r_slice,
                                                            azimuth_slice,
                                                            elevation_slice,
                                                            velocity_slice,
                                                            nyquist_velocity,
                                                            **kwargs)
            else:
                final_vel, flag_vel = dealias_long_range(r_slice,
                                                        azimuth_slice,
                                                        elevation_slice,
                                                        velocity_slice,
                                                        nyquist_velocity,
                                                        **kwargs)

            final_vel, flag_slice, _, _ = continuity.unfolding_3D(r_reference,
                                                                azimuth_reference,
                                                                elevation_reference,
                                                                velocity_reference,
                                                                flag_reference,
                                                                r_slice,
                                                                azimuth_slice,
                                                                elevation_slice,
                                                                final_vel,
                                                                flag_vel,
                                                                velocity_slice,
                                                                nyquist_velocity)

            azimuth_reference = azimuth_slice.copy()
            velocity_reference = final_vel.copy()
            flag_reference = flag_vel.copy()
            elevation_reference = elevation_slice
            r_reference = r_slice

            radar_datasets[sweep] = radar_datasets[sweep].merge({output_vel_name: (('azimuth', 'range'), final_vel)})

        r_reference = radar_datasets[nslice_ref].range.values
        azimuth_reference = radar_datasets[nslice_ref].azimuth.values
        velocity_reference = radar_datasets[nslice_ref][output_vel_name].values
        elevation_reference = radar_datasets[nslice_ref]['elevation'].values[0]

    # Processing sweeps by increasing elevations from the nslice_ref sweeps
    for sweep in range(nslice_ref + 1, len(radar_datasets)):
        r_slice = radar_datasets[sweep].range.values
        azimuth_slice = radar_datasets[sweep].azimuth.values
        velocity_slice = radar_datasets[sweep][vel_name].values
        elevation_slice = radar_datasets[sweep]['elevation'].values[0]
        nyquist_velocity = radar_datasets[sweep].attrs['NI']

        if strategy == 'default':
                final_vel, flag_vel = dealiasing_process_2D(r_slice,
                                                            azimuth_slice,
                                                            elevation_slice,
                                                            velocity_slice,
                                                            nyquist_velocity,
                                                            **kwargs)
        else:
            final_vel, flag_vel = dealias_long_range(r_slice,
                                                    azimuth_slice,
                                                    elevation_slice,
                                                    velocity_slice,
                                                    nyquist_velocity,
                                                    **kwargs)

        final_vel, flag_slice, _, _ = continuity.unfolding_3D(r_reference,
                                                            azimuth_reference,
                                                            elevation_reference,
                                                            velocity_reference,
                                                            flag_reference,
                                                            r_slice,
                                                            azimuth_slice,
                                                            elevation_slice,
                                                            final_vel,
                                                            flag_vel,
                                                            velocity_slice,
                                                            nyquist_velocity)

        azimuth_reference = azimuth_slice.copy()
        velocity_reference = final_vel.copy()
        flag_reference = flag_vel.copy()
        elevation_reference = elevation_slice
        r_reference = r_slice

        radar_datasets[sweep] = radar_datasets[sweep].merge({output_vel_name: (('azimuth', 'range'), final_vel)})

    return radar_datasets