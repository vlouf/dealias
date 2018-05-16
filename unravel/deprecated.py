"""
That module contains a bunch of functions that were once useful and are not used
anymore. Keeping them just in case they'd be needed in the future. All these
functions comes from the continuity module and cannot work without it.

@title: deprecated
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 08/03/2018

    find_ref_vel
    correct_clockwise_loose
    correct_counterclockwise_loose
    radial_continuity_roi
"""
# Other Libraries
import numpy as np

from numba import jit, int64, float64
from scipy.stats import linregress

from .continuity import *


@jit(nopython=True)
def find_ref_vel(azi, nazi, ngate, final_vel, flag_vel):
    """
    Find a value of reference for the velocity.

    Parameters:
    ===========
    azi: array
        Azimuth
    nazi: int
        Position of azimuth being processed.
    ngate: int
        Position of gate being processed.
    final_vel: array
        Array of unfolded velocities.

    Returns:
    ========
    mean_vel_ref: float
        Velocity of reference for comparison.
    """
    # Checking for good vel
    velref_ngate = final_vel[get_iter_pos(azi, nazi - 15, 14), ngate]
    flagref_ngate = flag_vel[get_iter_pos(azi, nazi - 15, 14), ngate]
    if np.sum((flagref_ngate == 1)) < 1:
        return None
    else:
        mean_vel_ref = np.median(velref_ngate[(flagref_ngate >= 1)])

    return mean_vel_ref


@jit
def correct_clockwise_loose(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq):
    maxgate = len(r)
    for nazi in myquadrant[3:]:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nazi, ngate] >= 1:
                continue

            # We want the previous 3 radials.
            npos = nazi - 4
            # Unfolded velocity
            velref = final_vel[get_iter_pos(azi, npos, 3), ngate]
            flagvelref = flag_vel[get_iter_pos(azi, npos, 3), ngate]

            # Folded velocity
            vel1 = vel[nazi, ngate]

            if np.sum((flagvelref == 1) | (flagvelref == 2)) < 2:
                mean_vel_ref = find_ref_vel(azi, nazi, ngate, final_vel, flag_vel)
            else:
                mean_vel_ref = np.mean(velref[(flagvelref == 1) | (flagvelref == 2)])

            if mean_vel_ref is None:
                # No reference found.
                continue

            decision = take_decision(mean_vel_ref, vel1, vnyq)

            if decision == -3:
                flag_vel[nazi, ngate] = -3
                continue
            elif decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, vel1)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit
def correct_counterclockwise_loose(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq):
    maxgate = len(r)
    for nazi in myquadrant:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nazi, ngate] >= 1:
                continue

            # We want the next 3 radials.
            npos = nazi + 1
            # Unfolded velocity.
            velref = final_vel[get_iter_pos(azi, npos, 3), ngate]
            flagvelref = flag_vel[get_iter_pos(azi, npos, 3), ngate]

            # Folded velocity
            vel1 = vel[nazi, ngate]

            if np.sum((flagvelref == 1) | (flagvelref == 2)) < 2:
                mean_vel_ref = find_ref_vel(azi, nazi, ngate, final_vel, flag_vel)
            else:
                mean_vel_ref = np.mean(velref[(flagvelref == 1) | (flagvelref == 2)])

            if mean_vel_ref is None:
                # No reference found.
                continue

            decision = take_decision(mean_vel_ref, vel1, vnyq)

            if decision == -3:
                flag_vel[nazi, ngate] = -3
                continue
            elif decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, vel1)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def radial_continuity_roi(azi, vel, final_vel, flag_vel, vnyq):
    maxazi, maxrange = final_vel.shape

    window_azimuth = 10
    window_range = 30

    unproc_azi, unproc_rng = np.where(flag_vel == 0)
    for nazi, ngate in zip(unproc_azi, unproc_rng):
        vel1 = vel[nazi, ngate]

        knt = 0
        while knt < 5:
            knt += 1
            npos_azi = get_iter_pos(azi, nazi - window_azimuth * knt // 2, window_azimuth * knt)
            npos_range = get_iter_range(ngate, window_range * knt, maxrange)

            flag_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN
            vel_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN

            # I know a slice would be better, but this is for jit to work.
            cnt = -1
            for na in npos_azi:
                for nr in npos_range:
                    cnt += 1
                    if (na, nr) == (nazi, ngate):
                        continue
                    vel_ref_vec[cnt] = final_vel[na, nr]
                    flag_ref_vec[cnt] = flag_vel[na, nr]

            mean_vel_ref = np.nanmedian(flag_ref_vec[vel_ref_vec > 0])
            decision = take_decision(mean_vel_ref, vel1, vnyq)

            if decision > 0:
                break

        if decision == 1:
            final_vel[nazi, ngate] = vel1
            flag_vel[nazi, ngate] = 1
        elif decision == 2:
            vtrue = unfold(mean_vel_ref, vel1)
            if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=0.8):
                final_vel[nazi, ngate] = vtrue
                flag_vel[nazi, ngate] = 2
            else:
                final_vel[nazi, ngate] = mean_vel_ref
                flag_vel[nazi, ngate] = 3

    return final_vel, flag_vel


def dealiasing_process_2D(r, azimuth, velocity, elev_angle, nyquist_velocity, debug=False, inherit_flag=None):
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
        Flag array (-3: No data, 0: Unprocessed, 1: Processed - no change -, 2: Processed - dealiased.)
    """
    st_time = time.time()  # tic
    try:
        velocity_nomask = velocity.filled(np.NaN)
    except Exception:
        velocity_nomask = velocity.copy()

    # Dealiasing based upon previously corrected velocities starting from two reference
    # radials, approximately 180° apart, where the wind is nearly orthogonal to the radar beam..
    try:
        azi_start_pos, azi_end_pos = find_reference.find_reference_radials(azimuth, velocity, debug)
    except ValueError:
        print(f"No reference found for elevation {elev_angle}.")
        azi_start_pos = 0
        azi_end_pos = len(azimuth) // 2

    # Looking for midpoints between the two reference radials. (4 quadrants to iter to).
    quadrant = find_reference.get_quadrant(azimuth, azi_start_pos, azi_end_pos)
    # Initialize unfolding, verifying reference radials.
    try:
        dealias_vel, flag_vel = initialisation.initialize_unfolding(r, azimuth, azi_start_pos,
                                                                    azi_end_pos, velocity_nomask, nyquist_velocity)
    except Exception:
        flag_vel = inherit_flag
        if inherit_flag is None:
            raise ValueError("No possible starting point found. Cannot dealias.")

        dealias_vel = velocity_nomask.copy()
        pos = flag_vel != 1
        flag_vel[pos] = 0
        dealias_vel[pos] = 0
        dealias_vel[np.isnan(dealias_vel)] = 0

    # This is very strict continuity, we want to make the most of it.
    nperc_previous = 0
    nperc = count_proc(flag_vel)
    while nperc - nperc_previous > 0.1:
        nperc_previous = nperc
        dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel, flag_vel,
                                                             quadrant[0], nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                                    flag_vel, quadrant[1], nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel, flag_vel,
                                                             quadrant[2], nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                                    flag_vel, quadrant[3], nyquist_velocity)

        dealias_vel, flag_vel = continuity.correct_range_onward(velocity, dealias_vel, flag_vel, nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_range_backward(velocity, dealias_vel, flag_vel, nyquist_velocity)

        nperc = count_proc(flag_vel)
        if nperc == 100:
            break

        # Radial dealiasing inside the quadrants, starting from midpoints. With less strict reference findings.
        dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                                    flag_vel, quadrant[0][::-1], nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel, flag_vel,
                                                             quadrant[1][::-1], nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                                    flag_vel, quadrant[2][::-1], nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel, flag_vel,
                                                             quadrant[3][::-1], nyquist_velocity)

        dealias_vel, flag_vel = continuity.correct_range_onward(velocity, dealias_vel, flag_vel, nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_range_backward(velocity, dealias_vel, flag_vel, nyquist_velocity)

        nperc = count_proc(flag_vel)
        if nperc == 100:
            break

    # One full sweep.
    # dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel,
    #                                                      flag_vel, np.arange(dealias_vel.shape[0]), nyquist_velocity)
    # Loose radial area dealiasing.
    nperc = count_proc(flag_vel)
    if nperc < 100:
        dealias_vel, flag_vel = continuity.correct_range_onward_loose(azimuth, velocity_nomask,
                                                                      dealias_vel, flag_vel,
                                                                      nyquist_velocity)

    # Looking for the closest reference..
    nperc = count_proc(flag_vel)
    if nperc < 100:
        dealias_vel, flag_vel = continuity.correct_closest_reference(azimuth, velocity_nomask,
                                                                     dealias_vel, flag_vel,
                                                                     nyquist_velocity)

    # Box error check with respect to surrounding velocities
    dealias_vel, flag_vel = continuity.correct_box(azimuth, velocity_nomask, dealias_vel,
                                                   flag_vel, nyquist_velocity)

    # Dealiasing with a circular area of points around the unprocessed ones (no point left unprocessed after this step).
    if elev_angle <= 6:
        # Least squares error check in the radial direction
        dealias_vel, flag_vel = continuity.radial_least_square_check(r, azimuth, velocity_nomask,
                                                                     dealias_vel, flag_vel,
                                                                     nyquist_velocity)

    # No flag.
    dealias_vel = continuity.least_square_radial_last_module(r, azimuth, dealias_vel,
                                                             nyquist_velocity)

    dealias_vel, flag_vel = continuity.box_check(azimuth, dealias_vel, flag_vel, nyquist_velocity)

    if debug:
        print(f"2D fields processed in {time.time() - st_time:0.2f} seconds for {elev_angle:0.2f} elevation.")

    try:
        dealias_vel = dealias_vel.filled(np.NaN)
    except Exception:
        pass

    return dealias_vel, flag_vel


def process_3D(radar, velname="VEL", dbzname="DBZ", zdrname="ZDR", rhohvname="RHOHV",
               gatefilter=None, nyquist_velocity=None, two_passes=False, debug=False):
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
    zdr_name: str
        Name of the differential reflectivity field.
    rho_name: str
        Name of the cross correlation ratio field.

    Returns:
    ========
    ultimate_dealiased_velocity: ndarray
        Dealised velocity field.
    """
    # Filter
    if gatefilter is None:
        gatefilter = filtering.do_gatefilter(radar, "VEL", "DBZ", zdr_name="ZDR", rho_name="RHOHV")

    if nyquist_velocity is None:
        nyquist_velocity = radar.instrument_parameters['nyquist_velocity']['data'][0]

    # Start with first reference.
    slice_number = 0
    myslice = radar.get_slice(slice_number)

    r = radar.range['data'].copy()
    velocity = radar.fields["VEL"]['data'].copy()
    azimuth_reference = radar.azimuth['data'][myslice]
    elevation_reference = radar.elevation['data'][myslice].mean()

    velocity_reference = np.ma.masked_where(gatefilter.gate_excluded, velocity)[myslice]

    # Dealiasing first sweep.
    final_vel, flag_vel, azi_s, azi_e = dealiasing_process_2D(r, azimuth_reference, velocity_reference,
                                                              elevation_reference, nyquist_velocity, debug=False)

    velocity_reference = final_vel.copy()
    flag_reference = flag_vel.copy()

    ultimate_dealiased_velocity = np.zeros(radar.fields["VEL"]['data'].shape)
    ultimate_dealiased_velocity[myslice] = final_vel.copy()

    for slice_number in range(1, radar.nsweeps):
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

        # 3D dealiasing
        velocity_slice, flag_slice = continuity.unfolding_3D(r, elevation_reference,
                                                             azimuth_reference, elevation_slice,
                                                             azimuth_slice, velocity_reference,
                                                             flag_reference, velocity_slice,
                                                             flag_slice, nyquist_velocity,
                                                             loose=True)

        final_vel, flag_vel, azi_s, azi_e = dealiasing_process_2D(r, azimuth_slice, velocity_slice,
                                                                  elevation_slice, nyquist_velocity,
                                                                  debug=False, inherit_flag=flag_slice,
                                                                  inherit_azi_start=azi_s, inherit_azi_end=azi_e)

        if two_passes:
            velocity_slice, flag_slice = continuity.unfolding_3D(r, elevation_reference,
                                                                 azimuth_reference,
                                                                 elevation_slice,
                                                                 azimuth_slice,
                                                                 velocity_reference,
                                                                 flag_reference,
                                                                 final_vel, flag_vel,
                                                                 nyquist_velocity, loose=False)
        azimuth_reference = azimuth_slice.copy()
        velocity_reference = final_vel.copy()
        flag_reference = flag_vel.copy()
        elevation_reference = elevation_slice

        ultimate_dealiased_velocity[myslice] = final_vel.copy()
    #     plot_radar(final_vel, flag_vel, slice_number)

    ultimate_dealiased_velocity = np.ma.masked_where(gatefilter.gate_excluded,
                                                     ultimate_dealiased_velocity)

    return ultimate_dealiased_velocity



# @jit(nopython=True)
# def correct_diagonal_right(vel, final_vel, flag_vel, vnyq):
#     maxazi, maxrange = final_vel.shape
#     for ncol in range(maxazi):
#         for i in range(maxrange - 1):
#             if ncol == 0 and i == 0:
#                 continue            
#             rpos = i
#             apos = i + ncol

#             while apos >= maxazi:
#                 apos -= maxazi
                
#             if flag_vel[apos, rpos] < 1:
#                 # Velocity used for reference has NOT been processed.
#                 continue

#             rpos_cmp = rpos + 1
#             apos_cmp = apos + 1
            
#             while apos_cmp >= maxazi:
#                 apos_cmp -= maxazi
            
#             if flag_vel[apos_cmp, rpos_cmp] != 0:
#                 # Velocity to unfold has already been processed.
#                 continue          
            
#             vel_cmp = vel[apos_cmp, rpos_cmp]
#             vel_ref = final_vel[apos, rpos]
            
#             decision = take_decision(vel_ref, vel_cmp, vnyq)
#             if decision == 1:
#                 final_vel[apos_cmp, rpos_cmp] = vel_cmp
#                 flag_vel[apos_cmp, rpos_cmp] = 1
#                 continue
#             elif decision == 2:
#                 vtrue = unfold(vel_ref, vel_cmp, vnyq)
#                 if is_good_velocity(vel_ref, vtrue, vnyq, alpha=0.4):
#                     final_vel[apos_cmp, rpos_cmp] = vtrue
#                     flag_vel[apos_cmp, rpos_cmp] = 2
                    
#     return final_vel, flag_vel


# @jit(nopython=True)
# def correct_diagonal_left(vel, final_vel, flag_vel, vnyq):
#     maxazi, maxrange = final_vel.shape
#     for ncol in range(maxazi - 1, -1, -1):
#         for i in range(maxrange - 1):
#             if ncol == 0 and i == 0:
#                 continue            
#             rpos = i
#             apos = i - ncol

#             while apos < 0:
#                 apos += maxazi

#             rpos_cmp = rpos + 1
#             apos_cmp = apos - 1
            
#             if apos_cmp < 0:
#                 apos_cmp += maxazi
            
#             if flag_vel[apos_cmp, rpos_cmp] != 0:
#                 # Velocity to unfold has already been processed.
#                 continue          
                
#             if flag_vel[apos, rpos] < 1:
#                 # Velocity used for reference has NOT been processed.
#                 continue
                
#             vel_cmp = vel[apos_cmp, rpos_cmp]
#             vel_ref = final_vel[apos, rpos]
            
#             decision = take_decision(vel_ref, vel_cmp, vnyq)
#             if decision == 1:
#                 final_vel[apos_cmp, rpos_cmp] = vel_cmp
#                 flag_vel[apos_cmp, rpos_cmp] = 1
#                 continue
#             elif decision == 2:
#                 vtrue = unfold(vel_ref, vel_cmp, vnyq)
#                 if is_good_velocity(vel_ref, vtrue, vnyq, alpha=0.4):
#                     final_vel[apos_cmp, rpos_cmp] = vtrue
#                     flag_vel[apos_cmp, rpos_cmp] = 2
                    
#     return final_vel, flag_vel
