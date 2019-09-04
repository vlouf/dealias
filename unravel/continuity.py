"""
Module containing all the functions used for dealiasing. These functions use
radial-to-radial continuity, gate-to-gate continuity, box check, least square
continuity, ...

JIT-friendly is my excuse for a lot of function containing loops or
structure controls to make the function compatible with the Just-In-Time (JIT)
compiler of numba while they are sometimes shorter pythonic ways to do things.

@title: continuity
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 08/03/2018
"""

# Other Libraries
import numpy as np

from numba import jit, int64, float64
from scipy.stats import linregress


@jit(nopython=True)
def unfold(v1, v2, vnyq, half_nyq=False):
    """
    Compare two velocities, look at all possible unfolding value (up to a period
    of 7 times the nyquist) and find the unfolded velocity that is the closest
    the to reference.

    Parameters:
    ===========
    v1: float
        Reference velocity
    v2: float
        Velocity to unfold
    vnyq: float
        Nyquist velocity
    half_nyq: bool
        Deprecated argument, should not be used.

    Returns:
    ========
        vtrue: float
            Dealiased velocity.
    """
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

# @jit(nopython=True)
# def unfold(vref, v, vnq, half_nyq=False):
#     delv = v - vref
#     vshift = vnq * 2
#     if delv <= vnq:
#         vout = v
#     else:
#         vout = v - int((delv + np.sign(delv) * vnq) / vshift) * vshift
#     return vout


@jit(nopython=True)
def is_good_velocity(vel1, vel2, vnyq, alpha=0.8):
    """
    Compare two velocities, and check if they are comparable to each other.

    Parameters:
    ===========
    vel1: float
        Reference velocity
    vel2: float
        Velocity to unfold
    vnyq: float
        Nyquist velocity
    alpha: float
        Coefficient for which the nyquist velocity periodicity range is
        considered valid.

    Returns:
    ========
    True/False
    """
    return np.abs(vel2 - vel1) < alpha * vnyq


@jit(nopython=True)
def get_iter_pos(azi, st, nb=180):
    """
    Return a sequence of integers from start (inclusive) to stop (start + nb)
    by step of 1 for iterating over the azimuth (handle the case that azimuth
    360 is in fact 0, i.e. a cycle).
    JIT-friendly function (this is why the function looks much longer than the
    pythonic way of doing this).

    Parameters:
    ===========
    azi: ndarray<float>
        Azimuth.
    st: int
        Starting point.
    nb: int
        Number of unitary steps.

    Returns:
    ========
    out: ndarray<int>
        Array containing the position from start to start + nb, i.e.
        azi[out[0]] <=> st
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
    Similar as get_iter_pos, but this time for creating an array of iterative
    indices over the radar range. JIT-friendly function.

    Parameters:
    ===========
    pos_center: int
        Starting point
    nb_gate: int
        Number of gates to iter to.
    maxrange: int
        Length of the radar range, i.e. maxrange = len(r)

    Returns:
    ========
    Array of iteration indices.
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


@jit(int64(float64, float64, float64, float64), nopython=True)
def take_decision(velocity_reference, velocity_to_check, vnyq, alpha):
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
    elif is_good_velocity(velocity_reference, velocity_to_check, vnyq, alpha=alpha) or (np.sign(velocity_reference) ==
                                                                                        np.sign(velocity_to_check)):
        return 1
    else:
        return 2


@jit(nopython=True)
def correct_clockwise(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq, window_len=3, alpha=0.4):
    """
    Dealias using strict radial-to-radial continuity. The previous 3 radials are
    used as reference. Clockwise means that we loop over increasing azimuth
    (which is in fact counterclockwise, but let's try not to be confusing).
    This function will look at unprocessed velocity only.
    In this version of the code, if no radials are found in continuity, then we
    we use the gate to gate continuity.

    Parameters:
    ===========
    r: ndarray
        Radar scan range.
    azi: ndarray
        Radar scan azimuth.
    vel: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    myquadrant: ndarray <int>
        Position of azimuth to iter upon.
    nyquist_velocity: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    maxgate = len(r)
    # the number 3 is because we use the previous 3 radials as reference.
    for nbeam in myquadrant[window_len:]:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nbeam, ngate] != 0:
                continue

            # We want the previous 3 radials.
            npos = nbeam - window_len
            # Unfolded velocity
            velref = final_vel[get_iter_pos(azi, npos, window_len), ngate]
            flagvelref = flag_vel[get_iter_pos(azi, npos, window_len), ngate]

            # Folded velocity
            vel1 = vel[nbeam, ngate]

            if np.sum((flagvelref == 1) | (flagvelref == 2)) <= 1:
                continue

            mean_vel_ref = np.mean(velref[(flagvelref == 1) | (flagvelref == 2)])

            decision = take_decision(mean_vel_ref, vel1, vnyq, alpha=alpha)

            # If loose, skip this test.
            if ngate != 0 and window_len <= 3:
                npos = ngate - 1
                mean_vel_ref2 = final_vel[nbeam, npos]

                decision2 = take_decision(mean_vel_ref2, vel1, vnyq, alpha=alpha)
                if decision != decision2:
                    continue

            if decision == -3:
                flag_vel[nbeam, ngate] = -3
                continue
            elif decision == 1:
                final_vel[nbeam, ngate] = vel1
                flag_vel[nbeam, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, vel1, vnyq)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def correct_counterclockwise(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq,
                             window_len=3, alpha=0.4):
    """
    Dealias using strict radial-to-radial continuity. The next 3 radials are
    used as reference. Counterclockwise means that we loop over decreasing
    azimuths (which is in fact clockwise... I know, it's confusing).
    This function will look at unprocessed velocity only.
    In this version of the code, if no radials are found in continuity, then we
    we use the gate to gate continuity.

    Parameters:
    ===========
    r: ndarray
        Radar scan range.
    azi: ndarray
        Radar scan azimuth.
    vel: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    myquadrant: ndarray <int>
        Position of azimuth to iter upon.
    vnyq: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    maxgate = len(r)

    for nbeam in myquadrant:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nbeam, ngate] != 0:
                continue

            # We want the next 3 radials.
            npos = nbeam + 1
            # Unfolded velocity.
            velref = final_vel[get_iter_pos(azi, npos, window_len), ngate]
            flagvelref = flag_vel[get_iter_pos(azi, npos, window_len), ngate]

            # Folded velocity
            vel1 = vel[nbeam, ngate]

            if np.sum((flagvelref == 1) | (flagvelref == 2)) <= 1:
                continue

            mean_vel_ref = np.mean(velref[(flagvelref == 1) | (flagvelref == 2)])

            decision = take_decision(mean_vel_ref, vel1, vnyq, alpha=alpha)

            # If loose, skip this test.
            if ngate != 0 and window_len <= 3:
                npos = ngate - 1
                mean_vel_ref2 = final_vel[nbeam, npos]

                decision2 = take_decision(mean_vel_ref2, vel1, vnyq, alpha=alpha)
                if decision != decision2:
                    continue

            if decision == -3:
                flag_vel[nbeam, ngate] = -3
                continue
            elif decision == 1:
                final_vel[nbeam, ngate] = vel1
                flag_vel[nbeam, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, vel1, vnyq)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def correct_range_onward(vel, final_vel, flag_vel, vnyq, window_len=6, alpha=0.4):
    """
    Dealias using strict gate-to-gate continuity. The directly previous gate
    is used as reference. This function will look at unprocessed velocity only.

    Parameters:
    ===========
    vel: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vnyq: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    maxazi, maxrange = final_vel.shape
    for nbeam in range(maxazi):
        for ngate in range(1, maxrange):
            if flag_vel[nbeam, ngate] != 0:
                continue

            vel1 = vel[nbeam, ngate]
            npos = ngate - 1
            velref = final_vel[nbeam, npos]
            flagvelref = flag_vel[nbeam, npos]

            if flagvelref <= 0:
                if ngate < window_len:
                    continue

                velref_vec = final_vel[nbeam, (ngate - window_len):ngate]
                flagvelref_vec = flag_vel[nbeam, (ngate - window_len):ngate]
                if np.sum(flagvelref_vec > 0) == 0:
                    continue

                velref = np.nanmean(velref_vec[flagvelref_vec > 0])

            decision = take_decision(velref, vel1, vnyq, alpha=alpha)

            if decision == 1:
                final_vel[nbeam, ngate] = vel1
                flag_vel[nbeam, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1, vnyq)
                if is_good_velocity(velref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def correct_range_backward(vel, final_vel, flag_vel, vnyq, window_len=6, alpha=0.4):
    """
    Dealias using strict gate-to-gate continuity. The directly next gate (going
    backward, i.e. from the outside to the center) is used as reference.
    This function will look at unprocessed velocity only.

    Parameters:
    ===========
    vel: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vnyq: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    for nbeam in range(vel.shape[0]):
        start_vec = np.where(flag_vel[nbeam, :] == 1)[0]
        if len(start_vec) == 0:
            continue

        start_gate = start_vec[-1]
        for ngate in np.arange(start_gate - (window_len + 1), window_len, -1):
            if flag_vel[nbeam, ngate] != 0:
                continue

            vel1 = vel[nbeam, ngate]
            npos = ngate + 1
            velref = final_vel[nbeam, npos]
            flagvelref = flag_vel[nbeam, npos]

            if flagvelref <= 0:
                if ngate + window_len >= vel.shape[1]:
                    # Out of range.
                    continue

                velref_vec = final_vel[nbeam, ngate:(ngate + window_len)]
                flagvelref_vec = flag_vel[nbeam, ngate:(ngate + window_len)]
                if np.sum(flagvelref_vec > 0) == 0:
                    continue

                velref = np.nanmean(velref_vec[flagvelref_vec > 0])

            decision = take_decision(velref, vel1, vnyq, alpha=alpha)

            if decision == 1:
                final_vel[nbeam, ngate] = vel1
                flag_vel[nbeam, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1, vnyq)
                if is_good_velocity(velref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def correct_closest_reference(azimuth, vel, final_vel, flag_vel, vnyq, alpha=0.4):
    """
    Dealias using the closest cluster of value already processed. Once the
    closest correct value is found, a take a window of 10 radials and 40 gates
    around that point and use the median as of those points as a reference.
    This function will look at unprocessed velocity only.

    Parameters:
    ===========
    azi: ndarray
        Radar scan azimuth.
    vel: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vnyq: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    window_azi = 10
    window_gate = 40
    maxazi, maxrange = final_vel.shape

    for nbeam in range(maxazi):
        posazi_good, posgate_good = np.where(flag_vel > 0)
        for ngate in range(0, maxrange):
            if flag_vel[nbeam, ngate] != 0:
                continue

            vel1 = vel[nbeam, ngate]

            distance = (posazi_good - nbeam) ** 2 + (posgate_good - ngate) ** 2
            if len(distance) == 0:
                continue

            closest = np.argmin(distance)
            nbeam_close = posazi_good[closest]
            ngate_close = posgate_good[closest]

            iter_azi = get_iter_pos(azimuth, nbeam_close - window_azi // 2, window_azi)
            iter_range = get_iter_range(ngate_close, window_gate, maxrange)

            vel_ref_vec = np.zeros((len(iter_azi) * len(iter_range), ), dtype=float64) + np.NaN

            # Numba doesn't support 2D slice, that's why I loop over things.
            pos = -1
            for na in iter_azi:
                pos += 1
                vel_ref_vec[pos] = np.nanmean(final_vel[na, iter_range[0]: iter_range[-1]][flag_vel[na, iter_range[0]: iter_range[-1]] > 0])
            velref = np.nanmedian(vel_ref_vec)

            decision = take_decision(velref, vel1, vnyq, alpha=alpha)

            if decision == 1:
                final_vel[nbeam, ngate] = vel1
                flag_vel[nbeam, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1, vnyq)
                if is_good_velocity(velref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2
                else:
                    final_vel[nbeam, ngate] = velref
                    flag_vel[nbeam, ngate] = 3

    return final_vel, flag_vel


@jit(nopython=True)
def correct_box(azi, vel, final_vel, flag_vel, vnyq, window_range=20,
                window_azimuth=10, strategy='vertex', alpha=0.4):
    """
    This module dealiases velocities based on the median of an area of corrected
    velocities preceding the gate being processed. This module is similar to
    the dealiasing technique from Bergen et al. (1988).

    Parameters:
    ===========
    azi: ndarray
        Radar scan azimuth.
    vel: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vnyq: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    if strategy == 'vertex':
        azi_window_offset = window_azimuth
    else:
        azi_window_offset = window_azimuth // 2

    maxazi, maxrange = final_vel.shape
    for nbeam in range(maxazi):
        for ngate in np.arange(maxrange - 1, -1, -1):
            if flag_vel[nbeam, ngate] != 0:
                continue

            myvel = vel[nbeam, ngate]

            npos_azi = get_iter_pos(azi, nbeam - azi_window_offset, window_azimuth)
            npos_range = get_iter_range(ngate, window_range, maxrange)

            flag_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN
            vel_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN

            # I know a slice would be better, but this is for jit to work.
            cnt = -1
            for na in npos_azi:
                for nr in npos_range:
                    cnt += 1
                    if (na, nr) == (nbeam, ngate):
                        continue
                    vel_ref_vec[cnt] = final_vel[na, nr]
                    flag_ref_vec[cnt] = flag_vel[na, nr]

            if np.sum(flag_ref_vec >= 1) == 0:
                continue

            mean_vel_ref = np.nanmean(vel_ref_vec[flag_ref_vec >= 1])

            decision = take_decision(mean_vel_ref, myvel, vnyq, alpha=alpha)

            if decision <= 0:
                continue

            if decision == 1:
                final_vel[nbeam, ngate] = myvel
                flag_vel[nbeam, ngate] = 1
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, myvel, vnyq)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def box_check(azi, final_vel, flag_vel, vnyq, window_range=80,
              window_azimuth=20, alpha=0.4, strategy='vertex'):
    """
    Check if all individual points are consistent with their surrounding
    velocities based on the median of an area of corrected velocities preceding
    the gate being processed. This module is similar to the dealiasing technique
    from Bergen et al. (1988). This function will look at ALL points.

    Parameters:
    ===========
    azi: ndarray
        Radar scan azimuth.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vnyq: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array NEW value: 3->had to be corrected.
    """
    if strategy == 'vertex':
        azi_window_offset = window_azimuth
    else:
        azi_window_offset = window_azimuth // 2

    maxazi, maxrange = final_vel.shape
    for nbeam in range(maxazi):
        for ngate in np.arange(maxrange - 1, -1, -1):
            if flag_vel[nbeam, ngate] <= 0:
                continue

            myvel = final_vel[nbeam, ngate]

            npos_azi = get_iter_pos(azi, nbeam - azi_window_offset, window_azimuth)
            npos_range = get_iter_range(ngate, window_range, maxrange)

            flag_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN
            vel_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN

            cnt = -1
            for na in npos_azi:
                for nr in npos_range:
                    cnt += 1
                    if (na, nr) == (nbeam, ngate):
                        continue
                    vel_ref_vec[cnt] = final_vel[na, nr]
                    flag_ref_vec[cnt] = flag_vel[na, nr]

            if np.sum(flag_ref_vec >= 1) == 0:
                continue

            true_vel = vel_ref_vec[flag_ref_vec >= 1]
            mvel = np.nanmean(true_vel)
            svel = np.nanstd(true_vel)
            myvelref = np.nanmedian(true_vel[(true_vel >= mvel - svel) & (true_vel <= mvel + svel)])

            if not is_good_velocity(myvelref, myvel, vnyq, alpha=alpha):
                final_vel[nbeam, ngate] = myvelref
                flag_vel[nbeam, ngate] = 3

    return final_vel, flag_vel


@jit
def radial_least_square_check(r, azi, vel, final_vel, flag_vel, vnyq, alpha=0.4):
    """
    Dealias a linear regression of gates inside each radials.
    This function will look at PROCESSED velocity only. This function cannot be
    fully JITed due to the use of the scipy function linregress.

    Parameters:
    ===========
    r: ndarray
        Radar range
    azi: ndarray
        Radar scan azimuth.
    vel: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vnyq: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    maxazi, maxrange = final_vel.shape
    for nbeam in range(maxazi):
        myvel = final_vel[nbeam, :]
        myvel[flag_vel[nbeam, :] <= 0] = np.NaN

        if len(myvel[~np.isnan(myvel)]) < 2:
            continue

        slope, intercept, _, _, _ = linregress(r[~np.isnan(myvel)], myvel[~np.isnan(myvel)])

        fmin = intercept + slope * r - 0.4 * vnyq
        fmax = intercept + slope * r + 0.4 * vnyq
        vaffine = intercept + slope * r

        for ngate in range(maxrange):
            if flag_vel[nbeam, ngate] <= 0:
                continue

            myvel = final_vel[nbeam, ngate]

            if myvel >= fmin[ngate] and myvel <= fmax[ngate]:
                continue

            mean_vel_ref = vaffine[ngate]
            decision = take_decision(mean_vel_ref, myvel, vnyq, alpha=alpha)

            if decision <= 0:
                continue

            if decision == 1:
                final_vel[nbeam, ngate] = myvel
                flag_vel[nbeam, ngate] = 1
            elif decision == 2:
                myvel = vel[nbeam, ngate]
                vtrue = unfold(mean_vel_ref, myvel, vnyq)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel


@jit
def least_square_radial_last_module(r, azi, final_vel, vnyq, alpha=0.4):
    """
    Similar as radial_least_square_check.
    """
    maxazi, maxrange = final_vel.shape

    for nbeam in range(maxazi):
        myvel = final_vel[nbeam, :]

        if len(myvel[~np.isnan(myvel)]) < 10:
            continue

        slope, intercept, _, _, _ = linregress(r[~np.isnan(myvel)], myvel[~np.isnan(myvel)])

        fmin = intercept + slope * r - 0.4 * vnyq
        fmax = intercept + slope * r + 0.4 * vnyq
        vaffine = intercept + slope * r

        for ngate in range(maxrange):
            myvel = final_vel[nbeam, ngate]
            if np.isnan(myvel):
                continue

            if myvel >= fmin[ngate] and myvel <= fmax[ngate]:
                continue

            mean_vel_ref = vaffine[ngate]
            decision = take_decision(mean_vel_ref, myvel, vnyq, alpha=alpha)

            if decision <= 0:
                continue

            if decision == 1:
                final_vel[nbeam, ngate] = myvel
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, myvel, vnyq)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue

    return final_vel


@jit(nopython=True)
def unfolding_3D(r, elevation_reference, azimuth_reference, elevation_slice, azimuth_slice,
                 velocity_reference, flag_reference, velocity_slice, flag_slice, vnyq,
                 theta_3db=1, alpha=0.4):
    """
    Dealias using 3D continuity. This function will look at the velocities from
    one sweep (the reference) to the other (the slice).
    Parameters:
    ===========
    r: ndarray
        Radar range
    elevation_reference: float
        Elevation angle of the reference sweep.
    azimuth_reference: ndarray
        Azimuth of the reference sweep.
    elevation_slice: float
        Elevation angle of the sweep to dealias.
    azimuth_slice: ndarray
        Azimuth of the sweep to dealias.
    velocity_reference: ndarray <azimuth, r>
        Velocity of the reference sweep.
    flag_reference:
        Flag array of the reference
    velocity_slice: ndarray <azimuth, r>
        Velocity of the sweep to dealias.
    flag_slice:
        Flag array of the sweep to dealias.
    vnyq: float
        Nyquist velocity.
    loose: bool
        Being loose in the dealiasing.
    theta_3db: float
        Beamwidth.
    Returns:
    ========
    velocity_slice: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_slice: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    window_azimuth = 5
    window_range = 10

    ground_range_reference = r * np.cos(elevation_reference * np.pi / 180)
    ground_range_slice = r * np.cos(elevation_slice * np.pi / 180)

    altitude_reference_max = r * np.sin((elevation_reference + theta_3db) * np.pi / 180)
    altitude_slice_min = r * np.sin((elevation_slice - theta_3db) * np.pi / 180)

    maxazi, maxrange = velocity_slice.shape
    for nbeam in range(maxazi):
        for ngate in range(maxrange):
            if flag_slice[nbeam, ngate] <= 0:
                continue

            if altitude_reference_max[ngate] < altitude_slice_min[ngate]:
                break

            current_vel = velocity_slice[nbeam, ngate]

            rpos_reference = np.argmin(np.abs(ground_range_reference - ground_range_slice[ngate]))
            apos_reference = np.argmin(np.abs(azimuth_reference - azimuth_slice[nbeam]))

            apos_iter = get_iter_pos(azimuth_reference, apos_reference - window_azimuth // 2,
                                     window_azimuth)
            rpos_iter = get_iter_range(rpos_reference, window_range, maxrange)

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

            pos = (velocity_refcomp_array >= vmean - vstd) & \
                  (velocity_refcomp_array <= vmean + vstd)
            compare_vel = np.nanmedian(velocity_refcomp_array[pos])

            if not is_good_velocity(compare_vel, current_vel, vnyq, alpha=alpha):
                vtrue = unfold(compare_vel, current_vel, vnyq)
                if is_good_velocity(compare_vel, vtrue, vnyq, alpha=alpha):
                    velocity_slice[nbeam, ngate] = vtrue
                    flag_slice[nbeam, ngate] = 3

    return velocity_slice, flag_slice
