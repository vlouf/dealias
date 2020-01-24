"""
Driver script for the dealiasing module. You want to call process_3D.

@title: dealias
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 05/04/2018
@date: 04/12/2018

    count_proc
    dealiasing_process_2D
    dealias_long_range
    unravel_3D_pyart
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


def count_proc(myflag, debug=False):
    """
    Count how many gates are left to dealias.

    Parameters:
    ===========
    myflag: ndarray (int)
        Processing flag array.
    debug: bool
        Print switch.

    Returns:
    ========
    perc: float
        Percentage of gates processed.
    """
    count = np.sum(myflag == 0)
    total = myflag.size
    perc = (total - count) / total * 100
    if debug:
        print(f"Still {count} gates left to dealias. {perc:0.1f}% done.")
    return perc


def dealiasing_process_2D(r, azimuth, velocity, elev_angle, nyquist_velocity,
                          debug=False, alpha=0.6):
    """
    Dealiasing processing for 2D slice of the Doppler radar velocity field.

    Parameters:
    ===========
    r: ndarray
        Radar scan range.
    azimuth: ndarray
        Radar scan azimuth.
    velocity: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    elev_angle: float
        Elevation angle of the velocity field.
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
    # Parameters from Michel Chong
    vshift = 2 * nyquist_velocity  # By how much the velocity shift when folding
    delta_vmax = 0.5 * nyquist_velocity  # The authorised change in velocity from one gate to the other.

    # Pre-processing, filtering noise.
    flag_vel = np.zeros(velocity.shape, dtype=int)
    flag_vel[np.isnan(velocity)] = -3
    velocity, flag_vel = filtering.filter_data(velocity, flag_vel, nyquist_velocity, vshift, delta_vmax)
    velocity[flag_vel == -3] = np.NaN

    st_time = time.time()  # tic

    tot_gate = velocity.shape[0] * velocity.shape[1]
    nmask_gate = np.sum(np.isnan(velocity))
    if debug:
        print(f"There are {tot_gate - nmask_gate} gates to dealias at elevation {elev_angle}.")

    start_beam, end_beam = find_reference.find_reference_radials(azimuth, velocity)
    azi_start_pos = np.argmin(np.abs(azimuth - start_beam))
    azi_end_pos = np.argmin(np.abs(azimuth - end_beam))
    # quadrant = find_reference.get_quadrant(azimuth, azi_start_pos, azi_end_pos)

    dealias_vel, flag_vel = initialisation.initialize_unfolding(r, azimuth, azi_start_pos, azi_end_pos, velocity,
                                                                flag_vel, vnyq=nyquist_velocity)

    vel = velocity.copy()
    vel[azi_start_pos, :] = dealias_vel[azi_start_pos, :]
    delta_vmax = 0.75 * nyquist_velocity

    dealias_vel, flag_vel = initialisation.first_pass(azi_start_pos, vel, dealias_vel, flag_vel, nyquist_velocity,
                                                      vshift, delta_vmax)

    # Range
    dealias_vel, flag_vel = continuity.correct_range_onward(velocity, dealias_vel, flag_vel,
                                                            nyquist_velocity, alpha=alpha)
    dealias_vel, flag_vel = continuity.correct_range_backward(velocity, dealias_vel, flag_vel,
                                                              nyquist_velocity, alpha=alpha)

    for _ in range(2):
        if count_proc(flag_vel, False) < 100:
            azimuth_iteration = np.arange(azi_start_pos, azi_start_pos + len(azimuth))
            azimuth_iteration[azimuth_iteration >= len(azimuth)] -= len(azimuth)
            dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity, dealias_vel, flag_vel,
                                                                 azimuth_iteration, nyquist_velocity, 6, alpha=alpha)
            dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity, dealias_vel, flag_vel,
                                                                        azimuth_iteration,
                                                                        nyquist_velocity, 6, alpha=alpha)

        if count_proc(flag_vel, False) < 100:
            dealias_vel, flag_vel = continuity.correct_range_onward(velocity, dealias_vel, flag_vel, nyquist_velocity,
                                                                    alpha=alpha, window_len=6)
            dealias_vel, flag_vel = continuity.correct_range_backward(velocity, dealias_vel, flag_vel, nyquist_velocity,
                                                                      alpha=alpha, window_len=6)

    # Box error check with respect to surrounding velocities
    for window in [(5, 2), (20, 10), (40, 20), (80, 40)]:
        if count_proc(flag_vel, debug) < 100:
            dealias_vel, flag_vel = continuity.correct_box(azimuth, velocity, dealias_vel, flag_vel,
                                                        nyquist_velocity, window[0], window[1], alpha=alpha)

    # Dealiasing with a circular area of points around the unprocessed ones (no point left unprocessed after this step).
    if elev_angle <= 6:
        # Least squares error check in the radial direction
        dealias_vel, flag_vel = continuity.radial_least_square_check(r, azimuth, velocity, dealias_vel,
                                                                     flag_vel, nyquist_velocity, alpha=alpha)
        # No flag.
        dealias_vel = continuity.least_square_radial_last_module(r, azimuth, dealias_vel, nyquist_velocity, alpha=alpha)

    # Using clear air data to build a reference for the whole radial.
    if count_proc(flag_vel, debug) < 100:
        dealias_vel, flag_vel = continuity.correct_linear_interp(velocity, dealias_vel, flag_vel,
                                                                 nyquist_velocity, alpha=alpha)

    # Looking for the closest reference..
    if count_proc(flag_vel, False) < 100:
        dealias_vel, flag_vel = continuity.correct_closest_reference(azimuth, velocity, dealias_vel,
                                                                     flag_vel, nyquist_velocity, alpha=alpha)

    dealias_vel, flag_vel = continuity.box_check(azimuth, dealias_vel, flag_vel, nyquist_velocity, alpha=alpha)

    if debug:
        print(f"2D fields processed in {time.time() - st_time:0.2f} seconds for {elev_angle:0.2f} elevation.")

    try:
        dealias_vel = dealias_vel.filled(np.NaN)
    except Exception:
        pass

    return dealias_vel, flag_vel, azi_start_pos, azi_end_pos


def dealias_long_range(r,
                       azimuth,
                       velocity,
                       elev_angle,
                       nyquist_velocity,
                       debug=False,
                       alpha=0.6):
    """
    Dealiasing processing for 2D slice of the Doppler radar velocity field.

    Parameters:
    ===========
    r: ndarray
        Radar scan range.
    azimuth: ndarray
        Radar scan azimuth.
    velocity: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    elev_angle: float
        Elevation angle of the velocity field.
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
    # Parameters from Michel Chong
    vshift = 2 * nyquist_velocity  # By how much the velocity shift when folding
    delta_vmax = 0.5 * nyquist_velocity  # The authorised change in velocity from one gate to the other.

    # Pre-processing, filtering noise.
    flag_vel = np.zeros(velocity.shape, dtype=int)
    flag_vel[np.isnan(velocity)] = -3
    velocity, flag_vel = filtering.filter_data(velocity, flag_vel, nyquist_velocity, vshift, delta_vmax)
    velocity[flag_vel == -3] = np.NaN

    if debug:
        tot_gate = velocity.shape[0] * velocity.shape[1]
        nmask_gate = np.sum(np.isnan(velocity))
        print(f"There are {tot_gate - nmask_gate} gates to dealias at elevation {elev_angle}.")

    start_beam, end_beam = find_reference.find_reference_radials(azimuth, velocity)
    azi_start_pos = np.argmin(np.abs(azimuth - start_beam))
    azi_end_pos = np.argmin(np.abs(azimuth - end_beam))
    dealias_vel, flag_vel = initialisation.initialize_unfolding(r,
                                                                azimuth,
                                                                azi_start_pos,
                                                                azi_end_pos,
                                                                velocity,
                                                                flag_vel,
                                                                vnyq=nyquist_velocity)

    vel = velocity.copy()
    vel[azi_start_pos, :] = dealias_vel[azi_start_pos, :]
    delta_vmax = 0.75 * nyquist_velocity
    dealias_vel, flag_vel = initialisation.first_pass(azi_start_pos,
                                                      vel,
                                                      dealias_vel,
                                                      flag_vel,
                                                      nyquist_velocity,
                                                      vshift, delta_vmax)

    # Range
    dealias_vel, flag_vel = continuity.correct_range_onward(velocity,
                                                            dealias_vel,
                                                            flag_vel,
                                                            nyquist_velocity,
                                                            alpha=alpha)
    dealias_vel, flag_vel = continuity.correct_range_backward(velocity,
                                                              dealias_vel,
                                                              flag_vel,
                                                              nyquist_velocity,
                                                              alpha=alpha)

    for window in [6, 12, 24, 48, 96]:
        if count_proc(flag_vel, debug) < 100:
            azimuth_iteration = np.arange(azi_start_pos, azi_start_pos + len(azimuth)) % len(azimuth)
            dealias_vel, flag_vel = continuity.correct_clockwise(r,
                                                                 azimuth,
                                                                 velocity,
                                                                 dealias_vel,
                                                                 flag_vel,
                                                                 azimuth_iteration,
                                                                 nyquist_velocity,
                                                                 window_len=window,
                                                                 alpha=alpha)
            dealias_vel, flag_vel = continuity.correct_counterclockwise(r,
                                                                        azimuth,
                                                                        velocity,
                                                                        dealias_vel,
                                                                        flag_vel,
                                                                        azimuth_iteration,
                                                                        nyquist_velocity,
                                                                        window_len=window,
                                                                        alpha=alpha)
            dealias_vel, flag_vel = continuity.correct_range_onward(velocity,
                                                                    dealias_vel,
                                                                    flag_vel,
                                                                    nyquist_velocity,
                                                                    alpha=alpha,
                                                                    window_len=window)
            dealias_vel, flag_vel = continuity.correct_range_backward(velocity,
                                                                      dealias_vel,
                                                                      flag_vel,
                                                                      nyquist_velocity,
                                                                      alpha=alpha,
                                                                      window_len=window)

    # Box error check with respect to surrounding velocities
    for window in [(20, 20), (40, 40), (80, 80)]:
        if count_proc(flag_vel, debug) < 100:
            dealias_vel, flag_vel = continuity.correct_box(azimuth,
                                                           velocity,
                                                           dealias_vel,
                                                           flag_vel,
                                                           nyquist_velocity,
                                                           window[0],
                                                           window[1],
                                                           alpha=alpha)
    # Using clear air data to build a reference for the whole radial.
    if count_proc(flag_vel, debug) < 100:
        dealias_vel, flag_vel = continuity.correct_linear_interp(velocity,
                                                                 dealias_vel,
                                                                 flag_vel,
                                                                 nyquist_velocity,
                                                                 alpha=alpha)
    # Looking for the closest reference..
    if count_proc(flag_vel, debug) < 100:
        dealias_vel, flag_vel = continuity.correct_closest_reference(azimuth,
                                                                     velocity,
                                                                     dealias_vel,
                                                                     flag_vel,
                                                                     nyquist_velocity,
                                                                     alpha=alpha)

    # Checking modules
    dealias_vel, flag_vel = continuity.box_check(azimuth, dealias_vel, flag_vel, nyquist_velocity, alpha=alpha)

    return dealias_vel, flag_vel, azi_start_pos, azi_end_pos


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
        final_vel, flag_vel, azi_s, azi_e = dealiasing_process_2D(r,
                                                                  azimuth_reference,
                                                                  velocity_reference,
                                                                  elevation_reference,
                                                                  nyquist_velocity,
                                                                  **kwargs)
    else:
        final_vel, flag_vel, azi_s, azi_e = dealias_long_range(r,
                                                               azimuth_reference,
                                                               velocity_reference,
                                                               elevation_reference,
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
            final_vel, flag_vel, azi_s, azi_e = dealiasing_process_2D(r,
                                                                      azimuth_slice,
                                                                      velocity_slice,
                                                                      elevation_slice,
                                                                      nyquist_velocity,
                                                                      **kwargs)
        else:
            final_vel, flag_vel, azi_s, azi_e = dealias_long_range(r,
                                                                   azimuth_slice,
                                                                   velocity_slice,
                                                                   elevation_slice,
                                                                   nyquist_velocity,
                                                                   **kwargs)

        final_vel[flag_vel == -3] = np.NaN
        final_vel, flag_slice, _, _ = continuity.unfolding_3D(r,
                                                              elevation_reference,
                                                              azimuth_reference,
                                                              elevation_slice,
                                                              azimuth_slice,
                                                              velocity_reference,
                                                              flag_reference,
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
