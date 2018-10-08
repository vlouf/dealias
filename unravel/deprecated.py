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


def get_static_rays(vel):
    """
    Compute the number of static gate (close to 0 m/s) and returns the best
    azimuths.

    To be a reference radial, four criteria must be met. First, they are minima
    in the curve of the normalized average of the absolute values of the
    measured velocities at all the valid gates along each radial.

    Parameter:
    ==========
    nvel: array [azi, range]
        velocity field.

    Returns:
    ========
    minpos: array
        Sorted array of the best azimuths.
    """
    try:
        nvel = vel.filled(np.NaN)
    except AttributeError:
        nvel = vel

    # Criterion 1: Top third of valid gates
    sum_good = np.sum(~np.isnan(nvel), axis=1)
    valid_pos = (sum_good / np.max(sum_good) > 2 / 3)

    n = np.sum(np.abs(vel), axis=1) / np.max(np.abs(vel), axis=1)
    d = np.sum(~np.isnan(nvel), axis=1)
    yall = n / d

    minpos = np.argsort(yall)[valid_pos]

    return minpos


def get_opposite_azimuth(myazi, tolerance=20):
    """
    Get the opposite azimuth plus/minus a tolerance.

    To be a reference radial, four criteria must be met.
    Second, these two initial reference radials should be separated by
    approximately 180.

    Parameters:
    ===========
    myazi: int
        Azimuth angle.
    tolerance: int
        Range of tolerance for the azimuth.

    Returns:
    ========
    minazi: int
        Opposite angle minimun range
    maxazi: int
        Opposite angle maximum range
    """
    azi_range = 180 - tolerance
    minazi = myazi + azi_range
    maxazi = myazi - azi_range
    if minazi > 360:
        minazi -= 360
    if maxazi < 0:
        maxazi += 360

    return [minazi, maxazi]


def get_valid_rays(vel):
    """
    Compute the quantity of valid gates for each rays, and returns the best
    azimuths.

    To be a reference radial, four criteria must be met.
    The fourth criterion is that the number of data points for the radial with
    the minimum sum must contain at least two-thirds of the average number
    of valid gates in all the radials in all azimuths


    Parameter:
    ==========
    nvel: array [azi, range]
        velocity field.

    Returns:
    ========
    extpos: array
        Sorted array of the best azimuths.
    """
    try:
        nvel = vel.filled(np.NaN)
    except AttributeError:
        nvel = vel
    # Criterion 1: Top third of valid gates
    sum_good = np.sum(~np.isnan(nvel), axis=1)
    valid_pos = (sum_good / np.max(sum_good) > 2 / 3)

    y2all = (np.max(np.abs(vel), axis=1) - np.min(np.abs(vel), axis=1))
#     y2 = y2all[valid_pos]

    extpos = np.argsort(y2all)[valid_pos]

    return extpos


def find_reference_radials(azi, vel, debug=False):
    """
    A reference radial is one that exhibits little or no aliasing. The most
    likely position for this to occur is where the wind direction is almost
    orthogonal to the direction the antenna is pointing. Also, the average value
    of the absolute value of that radial's Doppler velocity will be at a minimum.

    To be a reference radial, four criteria must be met. First, they are minima
    in the curve of the normalized average of the absolute values of the
    measured velocities at all the valid gates along each radial.
    Second, these two initial reference radials should be separated by
    approximately 180.

    Parameter:
    ==========
    azi
    vel
    rhohv

    Returns:
    ========
    minpos: array
        Sorted array of the best azimuths.
    """
    pos_valid = get_valid_rays(vel)
    pos_static = get_static_rays(vel)

    # Finding intersects of criteria 1 to 3.
    weight_valid = np.arange(0, len(pos_valid), 1)
    weight_static = np.arange(0, len(pos_static), 1)

    total_weight = np.zeros(len(pos_valid)) + np.NaN
    for cnt, (one_valid, one_valid_weight) in enumerate(zip(pos_valid, weight_valid)):
        try:
            one_static_weight = weight_static[one_valid == pos_static][0]
        except IndexError:
            one_static_weight = 9999

        total_weight[cnt] = one_static_weight + one_valid_weight

    pos1 = pos_valid[np.argmin(total_weight)]

#     # Finding the 2nd radial of reference
#     pos2 = pos1 + len(azi) // 2
#     if pos2 >= len(azi):
#         pos2 -= len(azi)

    try:
        ref2_range_min, ref2_range_max = get_opposite_azimuth(azi[pos1])
        if ref2_range_min < ref2_range_max:
            goodpos = np.where((azi >= ref2_range_min) & (azi <= ref2_range_max))[0]
        else:
            goodpos = np.where((azi >= ref2_range_min) | (azi <= ref2_range_max))[0]

        rslt = [(a, total_weight[a == pos_valid][0]) for a in goodpos if a in pos_valid]
        opposite_pos, opposite_weight = zip(*rslt)
        pos2 = opposite_pos[np.argmin(opposite_weight)]
    except Exception:
        pos2 = pos1 + len(azi) // 2
        if pos2 > len(azi):
            pos2 -= len(azi)
    if debug:
        print(f"References are azimuths {azi[pos1]} and {azi[pos2]}, i.e. azimuthal positions {pos1} and {pos2}.")

    return pos1, pos2


def get_quadrant(azi, posang1, posang2, full=False):
    """
    Compute the 4 part of the quadrant based on the 2 reference radials
    Quadrant 1 : reference radial 1 -> clockwise
    Quadrant 2 : reference radial 1 -> counter-clockwise
    Quadrant 3 : reference radial 2 -> clockwise
    Quadrant 4 : reference radial 2 -> counter-clockwise
    """
    ang1, ang2 = posang1, posang2
    maxazipos = len(azi)

    def get_sl(a, b, clock=1):
        if clock == 1:
            if a < b:
                return list(range(a, b + 1))
            else:
                return [*range(a, maxazipos), *range(0, b + 1)]
        else:
            if a > b:
                return list(range(a, b - 1, -1))
            else:
                return [*range(a, -1, -1), *range(maxazipos - 1, b - 1, -1)]

    if ang1 > ang2:
        dist1 = ang1 - ang2
        dist2 = maxazipos - dist1
        mid1 = ang1 + dist1 // 2
        if mid1 >= maxazipos:
            mid1 -= maxazipos

        mid2 = ang1 - dist2 // 2
        if mid2 < 0:
            mid2 += maxazipos

        quad = [None] * 4
        if not full:
            quad[0] = get_sl(ang1, mid1, 1)
            quad[1] = get_sl(ang1, mid2, -1)
            quad[2] = get_sl(ang2, mid2, 1)
            quad[3] = get_sl(ang2, mid1, -1)
        else:
            quad[0] = get_sl(ang1, ang2, 1)
            quad[1] = get_sl(ang1, ang2, -1)
            quad[2] = get_sl(ang2, ang1, 1)
            quad[3] = get_sl(ang2, ang1, -1)

    else:
        dist1 = ang2 - ang1
        dist2 = maxazipos - dist1
        mid1 = ang1 + dist1 // 2
        if mid1 >= maxazipos:
            mid1 -= maxazipos

        mid2 = ang1 - dist2 // 2
        if mid2 < 0:
            mid2 += maxazipos

        quad = [None] * 4
        if not full:
            quad[0] = get_sl(ang1, mid1, 1)
            quad[1] = get_sl(ang1, mid2, -1)
            quad[2] = get_sl(ang2, mid2, 1)
            quad[3] = get_sl(ang2, mid1, -1)
        else:
            quad[0] = get_sl(ang1, ang2, 1)
            quad[1] = get_sl(ang1, ang2, -1)
            quad[2] = get_sl(ang2, ang1, 1)
            quad[3] = get_sl(ang2, ang1, -1)

    return quad


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
