"""
Driver script for the debugging module.

@title: dealias
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 05/04/2018
@date: 24/01/2020

    debug_dealiasing
"""

# Python Standard Library
import time
import traceback

# Other python libraries.
import numpy as np

# Local
from . import continuity
from . import filtering
from . import initialisation
from . import find_reference
from .dealias import dealiasing_process_2D, dealias_long_range


def debug_dealiasing(radar,
                     velname="VEL",
                     dbzname="DBZ",
                     gatefilter=None,
                     nyquist_velocity=None,
                     strategy='default',
                     debug=True,
                     alpha=0.6):
    """
    Process driver.
    Full dealiasing process 2D + 3D.

    Parameters:
    ===========
    input_file: str
        Input radar file to dealias. File must be compatible with Py-ART.
    gatefilter: Object GateFilter
        GateFilter for filtering noise. If not provided it will automaticaly
        compute it with help of the dual-polar variables.
    vel_name: str
        Name of the velocity field.
    dbz_name: str
        Name of the reflectivity field.
    strategy: ['default', 'long_range']
        Using the default dealiasing strategy or the long range strategy.

    Returns:
    ========
    unraveled_velocity: ndarray
        Dealised velocity field.
    """
    if strategy not in ['default', 'long_range']:
        raise ValueError("Dealiasing strategy not understood please choose 'default' or 'long_range'")
    # Filter
    if gatefilter is None:
        gatefilter = filtering.do_gatefilter(radar, velname, dbzname)

    if nyquist_velocity is None:
        nyquist_velocity = radar.instrument_parameters['nyquist_velocity']['data'][0]

    # Start with first reference.
    myslice = radar.get_slice(0)

    r = radar.range['data'].copy()
    velocity = radar.fields[velname]['data'].copy()
    azimuth_reference = radar.azimuth['data'][myslice]
    elevation_reference = radar.elevation['data'][myslice].mean()
    velocity_reference = np.ma.masked_where(gatefilter.gate_excluded, velocity)[myslice]

    # Dealiasing first sweep.
    if strategy == 'default':
        final_vel, flag_vel = dealiasing_process_2D(r,
                                                    azimuth_reference,
                                                    velocity_reference,
                                                    elevation_reference,
                                                    nyquist_velocity,
                                                    alpha=alpha)
    else:
        final_vel, flag_vel = dealias_long_range(r,
                                                 azimuth_reference,
                                                 elevation_reference,
                                                 velocity_reference,
                                                 nyquist_velocity,
                                                 alpha=alpha)

    velocity_reference = final_vel.copy()
    flag_reference = flag_vel.copy()

    # 3D/2D processing array results.
    shape = radar.fields[velname]['data'].shape
    unraveled_velocity = np.zeros(shape)
    unraveled_velocity_2D = np.zeros(shape)
    debug_3D_velocity =  np.zeros(shape)
    processing_flag = np.zeros(shape)

    unraveled_velocity[myslice] = final_vel.copy()
    unraveled_velocity_2D[myslice] = final_vel.copy()

    for slice_number in range(1, radar.nsweeps):
        if debug:
            print(slice_number)
        myslice = radar.get_slice(slice_number)
        azimuth_slice = radar.azimuth['data'][myslice]
        elevation_slice = radar.elevation['data'][myslice].mean()

        if len(azimuth_slice) < 60:
            print(f"Problem with slice #{slice_number}, only {len(azimuth_slice)} radials.")
            continue

        vel = np.ma.masked_where(gatefilter.gate_excluded, velocity)[myslice]
        velocity_slice = vel.filled(np.NaN)

        flag_slice = np.zeros_like(velocity_slice) + 1
        flag_slice[np.isnan(velocity_slice)] = -3

        if strategy == 'default':
            final_vel, flag_vel = dealiasing_process_2D(r,
                                                     azimuth_slice,
                                                     velocity_slice,
                                                     elevation_slice,
                                                     nyquist_velocity,
                                                     debug=debug,
                                                     alpha=alpha)
        else:
            final_vel, flag_vel = dealias_long_range(r,
                                                     azimuth_slice,
                                                     velocity_slice,
                                                     elevation_slice,
                                                     nyquist_velocity,
                                                     debug=debug,
                                                     alpha=alpha)

        final_vel[flag_vel == -3] = np.NaN
        unraveled_velocity_2D[myslice] = final_vel.copy()

        final_vel, flag_slice, vel_as_ref, proc_flag = continuity.unfolding_3D(r,
                                                              azimuth_reference,
                                                              elevation_reference,
                                                              velocity_reference,
                                                              flag_reference,
                                                              r,
                                                              azimuth_slice,
                                                              elevation_slice,
                                                              final_vel,
                                                              flag_vel,
                                                              velocity[myslice],
                                                              nyquist_velocity)

        # Final box check to the 3D unfolding.
        final_vel, flag_slice = continuity.box_check(azimuth_slice,
                                                     final_vel,
                                                     flag_slice,
                                                     nyquist_velocity,
                                                     window_range=250,
                                                     alpha=alpha)

        azimuth_reference = azimuth_slice.copy()
        velocity_reference = final_vel.copy()
        flag_reference = flag_vel.copy()
        elevation_reference = elevation_slice

        unraveled_velocity[myslice] = final_vel.copy()
        debug_3D_velocity[myslice] = vel_as_ref
        processing_flag[myslice] = proc_flag

    unraveled_velocity = np.ma.masked_where(gatefilter.gate_excluded, unraveled_velocity)

    return unraveled_velocity, unraveled_velocity_2D, debug_3D_velocity, processing_flag
