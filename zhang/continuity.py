# Other Libraries
import numpy as np

from numba import jit, int64, float64
from scipy.stats import linregress


@jit(nopython=True)
def unfold(v1, v2, vnyq=13.3, half_nyq=False):
    if half_nyq:
        n = np.arange(1, 7, 1)
    else:
        n = np.arange(2, 7, 2)
    if v1 > 0:
        voff = v1 + (n * vnyq - np.abs(v1 - v2))
    else:
        voff = v1 - (n * vnyq - np.abs(v1 - v2))

    pos = np.argmin(np.abs(voff - v1))
    vtrue = voff[pos]

    return vtrue


@jit(nopython=True)
def is_good_velocity(vel1, vel2, vnyq, alpha=0.8):
    return np.abs(vel2 - vel1) < alpha * vnyq


@jit(nopython=True)
def get_iter_pos(azi, st, nb=180):
    """
    jit-friendly function. Iteration over azimuth
    """
    if st < 0:
        st += len(azi)
    if st >= len(azi):
        st -= len(azi)

    ed = st + nb
    if ed >= len(azi):
        ed -= len(azi)
    if ed < 0:
        ed += len(azi)

    posazi = np.arange(0, len(azi))
    mypos = np.empty_like(posazi)

    if nb > 0:
        if st < ed:
            end = ed - st
            mypos[:end] = posazi[st:ed]
        else:
            mid = (len(azi) - st)
            end = (len(azi) - st + ed)
            mypos[:mid] = posazi[st:]
            mypos[mid:end] = posazi[:ed]
    else:  # Goin backward.
        if st < ed:
            mid = st + 1
            end = st + len(azi) - ed
            mypos[:mid] = posazi[st::-1]
            mypos[mid:end] = posazi[-1:ed:-1]
        else:
            end = np.abs(st - ed)
            mypos[:end] = posazi[st:ed:-1]

    out = np.zeros((end, ), dtype=mypos.dtype)
    for n in range(end):
        out[n] = mypos[n]

    return out


@jit(nopython=True)
def get_iter_range(pos_center, nb_gate, maxrange):
    """
    jit-friendly function. Iteration over range
    """
    half_range = nb_gate // 2
    if pos_center < half_range:
        st_pos = 0
    else:
        st_pos = pos_center - half_range

    if pos_center + half_range >= maxrange:
        end_pos = maxrange
    else:
        end_pos = pos_center + half_range

    return np.arange(st_pos, end_pos)


# @jit(nopython=True)
# def find_ref_vel(azi, nazi, ngate, final_vel, flag_vel):
#     """
#     Find a value of reference for the velocity.
#
#     Parameters:
#     ===========
#     azi: array
#         Azimuth
#     nazi: int
#         Position of azimuth being processed.
#     ngate: int
#         Position of gate being processed.
#     final_vel: array
#         Array of unfolded velocities.
#
#     Returns:
#     ========
#     mean_vel_ref: float
#         Velocity of reference for comparison.
#     """
#     # Checking for good vel
#     velref_ngate = final_vel[get_iter_pos(azi, nazi - 15, 14), ngate]
#     flagref_ngate = flag_vel[get_iter_pos(azi, nazi - 15, 14), ngate]
#     if np.sum((flagref_ngate == 1)) < 1:
#         return None
#     else:
#         mean_vel_ref = np.median(velref_ngate[(flagref_ngate >= 1)])
#
#     return mean_vel_ref


@jit(int64(float64, float64, float64), nopython=True)
def take_decision(velocity_reference, velocity_to_check, vnyq):
    """
    Make a decision after comparing two velocities.

    Parameters:
    ===========
    velocity_to_check: float
        what we want to check
    velocity_reference: float
        reference

    Returns:
    ========
    -3: missing data (velocity we want to check does not exist)
    0: missing data (velocity used as reference does not exist)
    1: velocity is perfectly fine.
    2: velocity is folded.
    """
    if np.isnan(velocity_to_check):
        return -3
    elif np.isnan(velocity_reference):
        return 0
    elif is_good_velocity(velocity_reference, velocity_to_check, vnyq):
        return 1
    else:
        return 2


@jit(nopython=True)
def correct_clockwise(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq):
    maxgate = len(r)
    for nazi in myquadrant[3:]:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nazi, ngate] == 1:
                continue

            # We want the previous 3 radials.
            npos = nazi - 3
            # Unfolded velocity
            velref = final_vel[get_iter_pos(azi, npos, 3), ngate]
            flagvelref = flag_vel[get_iter_pos(azi, npos, 3), ngate]

            # Folded velocity
            vel1 = vel[nazi, ngate]

            if np.sum((flagvelref == 1) | (flagvelref == 2)) < 2:
                continue

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
def correct_counterclockwise(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq):
    maxgate = len(r)
    for nazi in myquadrant:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nazi, ngate] == 1:
                continue

            # We want the next 3 radials.
            npos = nazi + 1
            # Unfolded velocity.
            velref = final_vel[get_iter_pos(azi, npos, 3), ngate]
            flagvelref = flag_vel[get_iter_pos(azi, npos, 3), ngate]

            # Folded velocity
            vel1 = vel[nazi, ngate]

            if np.sum((flagvelref == 1) | (flagvelref == 2)) < 2:
                continue

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


# @jit
# def correct_clockwise_loose(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq):
#     maxgate = len(r)
#     for nazi in myquadrant[3:]:
#         for ngate in range(0, maxgate):
#             # Check if already unfolded
#             if flag_vel[nazi, ngate] >= 1:
#                 continue
#
#             # We want the previous 3 radials.
#             npos = nazi - 4
#             # Unfolded velocity
#             velref = final_vel[get_iter_pos(azi, npos, 3), ngate]
#             flagvelref = flag_vel[get_iter_pos(azi, npos, 3), ngate]
#
#             # Folded velocity
#             vel1 = vel[nazi, ngate]
#
#             if np.sum((flagvelref == 1) | (flagvelref == 2)) < 2:
#                 mean_vel_ref = find_ref_vel(azi, nazi, ngate, final_vel, flag_vel)
#             else:
#                 mean_vel_ref = np.mean(velref[(flagvelref == 1) | (flagvelref == 2)])
#
#             if mean_vel_ref is None:
#                 # No reference found.
#                 continue
#
#             decision = take_decision(mean_vel_ref, vel1, vnyq)
#
#             if decision == -3:
#                 flag_vel[nazi, ngate] = -3
#                 continue
#             elif decision == 1:
#                 final_vel[nazi, ngate] = vel1
#                 flag_vel[nazi, ngate] = 1
#                 continue
#             elif decision == 2:
#                 vtrue = unfold(mean_vel_ref, vel1)
#                 if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=0.4):
#                     final_vel[nazi, ngate] = vtrue
#                     flag_vel[nazi, ngate] = 2
#
#     return final_vel, flag_vel
#
#
# @jit
# def correct_counterclockwise_loose(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq):
#     maxgate = len(r)
#     for nazi in myquadrant:
#         for ngate in range(0, maxgate):
#             # Check if already unfolded
#             if flag_vel[nazi, ngate] >= 1:
#                 continue
#
#             # We want the next 3 radials.
#             npos = nazi + 1
#             # Unfolded velocity.
#             velref = final_vel[get_iter_pos(azi, npos, 3), ngate]
#             flagvelref = flag_vel[get_iter_pos(azi, npos, 3), ngate]
#
#             # Folded velocity
#             vel1 = vel[nazi, ngate]
#
#             if np.sum((flagvelref == 1) | (flagvelref == 2)) < 2:
#                 mean_vel_ref = find_ref_vel(azi, nazi, ngate, final_vel, flag_vel)
#             else:
#                 mean_vel_ref = np.mean(velref[(flagvelref == 1) | (flagvelref == 2)])
#
#             if mean_vel_ref is None:
#                 # No reference found.
#                 continue
#
#             decision = take_decision(mean_vel_ref, vel1, vnyq)
#
#             if decision == -3:
#                 flag_vel[nazi, ngate] = -3
#                 continue
#             elif decision == 1:
#                 final_vel[nazi, ngate] = vel1
#                 flag_vel[nazi, ngate] = 1
#                 continue
#             elif decision == 2:
#                 vtrue = unfold(mean_vel_ref, vel1)
#                 if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=0.4):
#                     final_vel[nazi, ngate] = vtrue
#                     flag_vel[nazi, ngate] = 2
#
#     return final_vel, flag_vel


@jit(nopython=True)
def correct_range_onward(vel, final_vel, flag_vel, vnyq):
    maxazi, maxrange = final_vel.shape
    for nazi in range(maxazi):
        for ngate in range(1, maxrange):
            if flag_vel[nazi, ngate] != 0:
                continue

            vel1 = vel[nazi, ngate]
            npos = ngate - 1
            velref = final_vel[nazi, npos]
            flagvelref = flag_vel[nazi, npos]

            if flagvelref <= 0:
                continue

            decision = take_decision(velref, vel1, vnyq)

            if decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1)
                if is_good_velocity(velref, vtrue, vnyq, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit
def correct_range_onward_loose(azi, vel, final_vel, flag_vel, vnyq):
    window_len = 10
    maxazi, maxrange = final_vel.shape
    for nazi in range(maxazi):
        for ngate in range(1, maxrange):
            if flag_vel[nazi, ngate] != 0:
                continue

            vel1 = vel[nazi, ngate]
            npos = ngate - 1
            is_good = 0
            cnt = 0
            while npos > window_len & cnt < 100:
                cnt += 1
                if flag_vel[nazi, npos] > 0:
                    is_good = 1
                    break
                npos -= 1

            if is_good == 0:
                continue

            st_azi = get_iter_pos(azi, nazi - 1, 3)
            velref_vec = final_vel[st_azi, npos - window_len:npos + 1]
            flagvelref = flag_vel[st_azi, npos - window_len:npos + 1]
            velref = np.nanmedian(velref_vec[flagvelref > 0])

            decision = take_decision(velref, vel1, vnyq)

            if decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1)
                if is_good_velocity(velref, vtrue, vnyq, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def correct_range_backward(vel, final_vel, flag_vel, vnyq):
    maxazi, maxrange = final_vel.shape
    for nazi in range(maxazi):
        start_vec = np.where(flag_vel[nazi, :] == 1)[0]
        if len(start_vec) == 0:
            continue

        start_gate = start_vec[-1]
        for ngate in np.arange(start_gate - 1, -1, -1):
            if flag_vel[nazi, ngate] != 0:
                continue

            vel1 = vel[nazi, ngate]
            npos = ngate + 1
            velref = final_vel[nazi, npos]
            flagvelref = flag_vel[nazi, npos]

            if flagvelref <= 0:
                continue

            decision = take_decision(velref, vel1, vnyq)

            if decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1)
                if is_good_velocity(velref, vtrue, vnyq, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def correct_closest_reference(r, azimuth, vel, final_vel, flag_vel, vnyq):
    _window_azi = 5
    _window_gate = 20
    maxazi, maxrange = final_vel.shape

    [R, A] = np.meshgrid(r, azimuth)
    X = R * np.cos(A * np.pi / 180)
    Y = R * np.sin(A * np.pi / 180)

    for nazi in range(maxazi):
        for ngate in range(maxgate):
            if flag_vel[nazi, ngate] != 0:
                continue

            vel1 = vel[nazi, ngate]
            if np.isnan(vel1):
                flag_vel[nazi, ngate] = -3
                continue

            myx = r[ngate] * np.cos(azimuth[nazi] * np.pi / 180)
            myy = r[ngate] * np.sin(azimuth[nazi] * np.pi / 180)

            R_circle = (X - myx) ** 2 + (Y - myy) ** 2

            good_posa, good_posr = np.where(flag_vel > 0)
            olda, oldr = good_posa[0], good_posr[0]
            for na in good_posa:
                for nr in good_posr:
                    if R_circle[na, nr] < R_circle[olda, oldr]:
                        olda, oldr = na, nr

            velref = final_vel[olda, oldr]
            decision = take_decision(velref, vel1, vnyq)

            if decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
            elif decision == 2:
                vtrue = unfold(velref, vel1)
                if is_good_velocity(velref, vtrue, vnyq, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


# @jit(nopython=True)
# def radial_continuity_roi(azi, vel, final_vel, flag_vel, vnyq):
#     maxazi, maxrange = final_vel.shape
#
#     window_azimuth = 10
#     window_range = 30
#
#     unproc_azi, unproc_rng = np.where(flag_vel == 0)
#     for nazi, ngate in zip(unproc_azi, unproc_rng):
#         vel1 = vel[nazi, ngate]
#
#         knt = 0
#         while knt < 5:
#             knt += 1
#             npos_azi = get_iter_pos(azi, nazi - window_azimuth * knt // 2, window_azimuth * knt)
#             npos_range = get_iter_range(ngate, window_range * knt, maxrange)
#
#             flag_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN
#             vel_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN
#
#             # I know a slice would be better, but this is for jit to work.
#             cnt = -1
#             for na in npos_azi:
#                 for nr in npos_range:
#                     cnt += 1
#                     if (na, nr) == (nazi, ngate):
#                         continue
#                     vel_ref_vec[cnt] = final_vel[na, nr]
#                     flag_ref_vec[cnt] = flag_vel[na, nr]
#
#             mean_vel_ref = np.nanmedian(flag_ref_vec[vel_ref_vec > 0])
#             decision = take_decision(mean_vel_ref, vel1, vnyq)
#
#             if decision > 0:
#                 break
#
#         if decision == 1:
#             final_vel[nazi, ngate] = vel1
#             flag_vel[nazi, ngate] = 1
#         elif decision == 2:
#             vtrue = unfold(mean_vel_ref, vel1)
#             if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=0.8):
#                 final_vel[nazi, ngate] = vtrue
#                 flag_vel[nazi, ngate] = 2
#             else:
#                 final_vel[nazi, ngate] = mean_vel_ref
#                 flag_vel[nazi, ngate] = 3
#
#     return final_vel, flag_vel


@jit(nopython=True)
def correct_box(azi, vel, final_vel, flag_vel, vnyq):
    """
    jit-friendly... so there are loops!
    Module 4
    """
    window_range = 20
    window_azimuth = 10
    maxazi, maxrange = final_vel.shape
    for nazi in range(maxazi):
        for ngate in np.arange(maxrange - 1, -1, -1):
            if flag_vel[nazi, ngate] != 0:
                continue

            myvel = vel[nazi, ngate]

            npos_azi = get_iter_pos(azi, nazi - window_azimuth // 2, window_azimuth)
            npos_range = get_iter_range(ngate, window_range, maxrange)

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

            if np.sum(flag_ref_vec >= 1) == 0:
                continue

            mean_vel_ref = np.nanmean(vel_ref_vec[flag_ref_vec >= 1])

            decision = take_decision(mean_vel_ref, myvel, vnyq)

            if decision <= 0:
                continue

            if decision == 1:
                final_vel[nazi, ngate] = myvel
                flag_vel[nazi, ngate] = 1
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, myvel)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def box_check(azi, final_vel, flag_vel, vnyq):
    """
    jit-friendly... so there are loops!
    backward range continuity
    Module 4
    """
    window_range = 40
    window_azimuth = 10
    maxazi, maxrange = final_vel.shape
    for nazi in range(maxazi):
        for ngate in np.arange(maxrange - 1, -1, -1):
            if flag_vel[nazi, ngate] <= 0:
                continue

            myvel = final_vel[nazi, ngate]

            npos_azi = get_iter_pos(azi, nazi - window_azimuth // 2, window_azimuth)
            npos_range = get_iter_range(ngate, window_range, maxrange)

            flag_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN
            vel_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN

            cnt = -1
            for na in npos_azi:
                for nr in npos_range:
                    cnt += 1
                    if (na, nr) == (nazi, ngate):
                        continue
                    vel_ref_vec[cnt] = final_vel[na, nr]
                    flag_ref_vec[cnt] = flag_vel[na, nr]

            if np.sum(flag_ref_vec >= 1) == 0:
                continue

            true_vel = vel_ref_vec[flag_ref_vec >= 1]
            mvel = np.nanmean(true_vel)
            svel = np.nanstd(true_vel)
            myvelref = np.nanmedian(true_vel[(true_vel >= mvel - svel) & (true_vel <= mvel + svel)])

            if not is_good_velocity(myvelref, myvel, vnyq):
                final_vel[nazi, ngate] = myvelref
                flag_vel[nazi, ngate] = 3

    return final_vel, flag_vel


@jit
def radial_least_square_check(r, azi, vel, final_vel, flag_vel, vnyq):
    """
    Module 5 from He et al.
    """
    tmp_vel = np.zeros_like(flag_vel)
    maxazi, maxrange = final_vel.shape
#     window_range = 20
#     window_azimuth = 10
    for nazi in range(maxazi):
        myvel = final_vel[nazi, :]
        myvel[flag_vel[nazi, :] <= 0] = np.NaN

        if len(myvel[~np.isnan(myvel)]) < 2:
            continue

        slope, intercept, _, _, _ = linregress(r[~np.isnan(myvel)], myvel[~np.isnan(myvel)])

        fmin = intercept + slope * r - 0.4 * vnyq
        fmax = intercept + slope * r + 0.4 * vnyq
        vaffine = intercept + slope * r

        for ngate in range(maxrange):
            if flag_vel[nazi, ngate] <= 0:
                continue

            myvel = final_vel[nazi, ngate]
            myr = r[ngate]

            if myvel >= fmin[ngate] and myvel <= fmax[ngate]:
                continue

            mean_vel_ref = vaffine[ngate]
            decision = take_decision(mean_vel_ref, myvel, vnyq)

            if decision <= 0:
                continue

            if decision == 1:
                final_vel[nazi, ngate] = myvel
                flag_vel[nazi, ngate] = 1
            elif decision == 2:
                myvel = vel[nazi, ngate]
                vtrue = unfold(mean_vel_ref, myvel)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit
def least_square_radial_last_module(r, azi, final_vel, vnyq):
    #     r, azimuth, dealias_vel, nyquist_velocity
    """
    Module 7 from He et al.
    """
    maxazi, maxrange = final_vel.shape

    for nazi in range(maxazi):
        vel_radial = final_vel[nazi, :]

        if len(vel_radial[~np.isnan(vel_radial)]) < 10:
            continue

        slope, intercept, _, _, _ = linregress(r[~np.isnan(vel_radial)], vel_radial[~np.isnan(vel_radial)])

        fmin = intercept + slope * r - 0.4 * vnyq
        fmax = intercept + slope * r + 0.4 * vnyq
        vaffine = intercept + slope * r

        for ngate in range(maxrange):
            myvel = final_vel[nazi, ngate]
            if np.isnan(myvel):
                continue

            if myvel >= fmin[ngate] and myvel <= fmax[ngate]:
                continue

            mean_vel_ref = vaffine[ngate]
            decision = take_decision(mean_vel_ref, myvel, vnyq)

            if decision <= 0:
                continue

            if decision == 1:
                final_vel[nazi, ngate] = myvel
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, myvel)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue

    return final_vel


@jit(nopython=True)
def unfolding_3D(r, elevation_reference, azimuth_reference, elevation_slice, azimuth_slice,
                 velocity_reference, flag_reference, velocity_slice, flag_slice, vnyq, theta_3db=1):

    ground_range_reference = r * np.cos(elevation_reference * np.pi / 180)
    ground_range_slice = r * np.cos(elevation_slice * np.pi / 180)

    altitude_reference_max = r * np.sin((elevation_reference + theta_3db) * np.pi / 180)
    altitude_slice_min = r * np.sin((elevation_slice - theta_3db) * np.pi / 180)

    maxazi, maxrange = velocity_slice.shape
    for nazi in range(maxazi):
        for ngate in range(maxrange):
            if flag_slice[nazi, ngate] <= 0:
                continue

            if altitude_reference_max[ngate] < altitude_slice_min[ngate]:
                break

            current_vel = velocity_slice[nazi, ngate]

            rpos_reference = np.argmin(np.abs(ground_range_reference - ground_range_slice[ngate]))
            apos_reference = np.argmin(np.abs(azimuth_reference - azimuth_slice[nazi]))

            apos_iter = get_iter_pos(azimuth_reference, apos_reference - 5, 10)
            rpos_iter = get_iter_range(rpos_reference, 10, maxrange)

            velocity_refcomp_array = np.zeros((len(rpos_iter) * len(apos_iter))) + np.NaN
            flag_refcomp_array = np.zeros((len(rpos_iter) * len(apos_iter))) + np.NaN

            cnt = -1
            for na in apos_iter:
                for nr in rpos_iter:
                    cnt += 1
                    velocity_refcomp_array[cnt] = velocity_reference[na, nr]
                    flag_refcomp_array[cnt] = flag_reference[na, nr]

            if np.sum(flag_refcomp_array >= 1) < 1:
                # No comparison possible
                continue

            velocity_refcomp_array = velocity_refcomp_array[(flag_refcomp_array >= 1)]
            vmean = np.nanmean(velocity_refcomp_array)
            vstd = np.nanstd(velocity_refcomp_array)
            pos = (velocity_refcomp_array >= vmean - vstd) & (velocity_refcomp_array <= vmean + vstd)
            compare_vel = np.nanmedian(velocity_refcomp_array[pos])

            if not is_good_velocity(compare_vel, current_vel, vnyq, alpha=0.4):
                vtrue = unfold(compare_vel, current_vel)
                if is_good_velocity(compare_vel, vtrue, vnyq, alpha=0.4):
                    velocity_slice[nazi, ngate] = vtrue
                    flag_slice[nazi, ngate] = 3

    return velocity_slice, flag_slice
