"""
Module containing all the functions used for dealiasing. These functions use
radial-to-radial continuity, gate-to-gate continuity, box check, least square
continuity, ...

JIT-friendly is my excuse for a lot of function containing loops or
structure controls to make the function compatible with the Just-In-Time (JIT)
compiler of numba while they are sometimes shorter pythonic ways to do things.

@title: continuity
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 15/03/2021

.. autosummary::
    :toctree: generated/

    linregress
    unfold
    is_good_velocity
    iter_azimuth
    iter_range
    take_decision
    correct_clockwise
    correct_counterclockwise
    correct_range_onward
    correct_range_backward
    correct_linear_interp
    correct_closest_reference
    correct_box
    _vectorized_stride
    _box_check_v2
    box_check
    radial_least_square_check
    least_square_radial_last_module
    unfolding_3D
"""

from . import cfg

import numpy as np

try:
    from numba import jit, jit_module, int64, float64
    HAVE_NUMBA = True
except Exception as e:
    print("numba import failed")
    HAVE_NUMBA = False
    float64 = float

def linregress(x, y):
    """
    Linear regression is an approach for predicting a response using a single
    feature. It is assumed that the two variables are linearly related. Hence,
    we try to find a linear function that predicts the response value(y) as
    accurately as possible as a function of the feature or independent
    variable(x).

    Parameters:
    ===========
        x: ndarray <vector>
        y: ndarray <vector>

    Returns:
    ========
        slope
        intecept
    """
    # number of observations/points
    n = len(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    slope = SS_xy / SS_xx
    intercept = m_y - slope * m_x

    return slope, intercept


def unfold(v1, v2, vnyq):
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

    Returns:
    ========
    return voff[pos]
        vtrue: float
            Dealiased velocity.
    """
    n = np.arange(0, 7, 2)
    if v1 > 0:
        voff = v1 + (n * vnyq - np.abs(v1 - v2))
    else:
        voff = v1 - (n * vnyq - np.abs(v1 - v2))

    pos = np.argmin(np.abs(voff - v1))
    return voff[pos]


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


def iter_azimuth(azi, idx_start, window_len):
    """
    Return a sequence of indices that are circling around for the azimuth.

    Parameters:
    ===========
    azi: ndarray<float>
        Azimuth.
    idx_start: int
        Starting point.
    window_len: int
        Window size.

    Returns:
    ========
    out: ndarray<int>
        Array containing the position from start to start + nb, i.e.
        azi[out[0]] <=> st
    """
    nbeam = len(azi)
    return np.arange(idx_start, idx_start + window_len) % nbeam


def iter_range(pos_center, window_len, maxgate):
    """
    Similar as iter_azimuth, but this time for creating an array of iterative
    indices over the radar range. JIT-friendly function.

    Parameters:
    ===========
    pos_center: int
        Starting point
    window_len: int
        Number of gates to iter to.
    maxgate: int
        Length of the radar range, i.e. maxgate = len(r)

    Returns:
    ========
    Array of iteration indices.
    """
    half_range = window_len // 2
    if pos_center < half_range:
        st_pos = 0
    else:
        st_pos = pos_center - half_range

    if pos_center + half_range >= maxgate:
        end_pos = maxgate
    else:
        end_pos = pos_center + half_range

    return np.arange(st_pos, end_pos)


# @jit(int64(float64, float64, float64, float64), nopython=True, cache=True)
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
    elif is_good_velocity(velocity_reference, velocity_to_check, vnyq, alpha=alpha) or (
            abs(velocity_reference) > cfg.SIGN_COMPARE_EPSILON and
            abs(velocity_to_check) > cfg.SIGN_COMPARE_EPSILON and
        np.sign(velocity_reference) == np.sign(velocity_to_check)
    ):
        return 1
    else:
        return 2


def correct_clockwise(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq, window_len=3, alpha=0.8):
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
    flag_threshold = window_len // 10
    if flag_threshold == 0:
        flag_threshold = 1
    elif flag_threshold > 10:
        flag_threshold = 10

    if cfg.SHOW_PROGRESS:
        print("correct_clockwise alpha:", alpha, f"win-len:{window_len}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    # the number 3 is because we use the previous 3 radials as reference.
    for nbeam in myquadrant[window_len:]:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nbeam, ngate] != 0:
                continue

            # We want the previous 3 radials.
            npos = nbeam - window_len
            # Unfolded velocity
            velref = final_vel[iter_azimuth(azi, npos, window_len), ngate]
            flagvelref = flag_vel[iter_azimuth(azi, npos, window_len), ngate]

            # Folded velocity
            vel1 = vel[nbeam, ngate]

            if np.sum(flagvelref > 0) < flag_threshold:
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


def correct_counterclockwise(r, azi, vel, final_vel, flag_vel, myquadrant, vnyq, window_len=3, alpha=0.8):
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
    flag_threshold = window_len // 10
    if flag_threshold == 0:
        flag_threshold = 1
    elif flag_threshold > 10:
        flag_threshold = 10

    if cfg.SHOW_PROGRESS:
        print("correct_counterclockwise alpha:", alpha, f"win-len:{window_len}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    if cfg.CORRECT_CLOCK_ITER:
        myquadrant = myquadrant[window_len:]

    for nbeam in myquadrant:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nbeam, ngate] != 0:
                continue

            # We want the next 3 radials.
            npos = nbeam + 1
            # Unfolded velocity.
            velref = final_vel[iter_azimuth(azi, npos, window_len), ngate]
            flagvelref = flag_vel[iter_azimuth(azi, npos, window_len), ngate]

            # Folded velocity
            vel1 = vel[nbeam, ngate]

            if np.sum(flagvelref > 0) < flag_threshold:
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


def correct_range_onward(vel, final_vel, flag_vel, vnyq, window_len=6, alpha=0.8):
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
    flag_threshold = window_len // 10
    if flag_threshold == 0:
        flag_threshold = 1
    elif flag_threshold > 10:
        flag_threshold = 10

    if cfg.SHOW_PROGRESS:
        print("correct_range_onward alpha:", alpha, f"win-len:{window_len}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

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

                # trailing reference window does not include current value
                velref_vec = final_vel[nbeam, (ngate - window_len) : ngate]
                flagvelref_vec = flag_vel[nbeam, (ngate - window_len) : ngate]
                if np.sum(flagvelref_vec > 0) < flag_threshold:
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


def correct_range_backward(vel, final_vel, flag_vel, vnyq, window_len=6, alpha=0.8):
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
    flag_threshold = window_len // 10
    if flag_threshold == 0:
        flag_threshold = 1
    elif flag_threshold > 10:
        flag_threshold = 10

    if cfg.SHOW_PROGRESS:
        print("correct_range_backward alpha:", alpha, f"win-len:{window_len}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    for nbeam in range(vel.shape[0]):
        if cfg.CORRECT_RANGE_CONSISTENT:
            gate_range = np.arange(vel.shape[1] - 2, -1, -1)
        else:
            start_vec = np.where(flag_vel[nbeam, :] == 1)[0]
            if len(start_vec) == 0:
                continue

            start_gate = start_vec[-1]
            gate_range = np.arange(start_gate - (window_len + 1), window_len, -1)


        for ngate in gate_range:
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

                # trailing reference window does not include current value
                velref_vec = final_vel[nbeam, ngate + 1 : ngate + window_len + 1]
                flagvelref_vec = flag_vel[nbeam, ngate + 1 : ngate + window_len + 1]
                if np.sum(flagvelref_vec > 0) < flag_threshold:
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


def correct_linear_interp(velocity, final_vel, flag_vel, vnyq, r_step=200, alpha=0.8):
    """
    Dealias using data close to the radar as reference for the most distant
    points left to dealiase.

    Parameters:
    ===========
    velocity: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vnyq: float
        Nyquist velocity.
    r_step: int
        Number of gates used to compute reference.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    maxazi, maxrange = final_vel.shape

    if cfg.SHOW_PROGRESS:
        print("correct_linear_interp (extrapolate) alpha:", alpha, f"window:{r_step}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    for nbeam in range(maxazi):
        if not np.any((flag_vel[nbeam, r_step:] == 0)):
            # There is nothing left to process for this azimuth.
            continue

        pos = flag_vel[nbeam, :r_step] > 0
        if np.sum(pos) == 0:
            # There's nothing that can be used as reference.
            continue

        v_selected = final_vel[nbeam, :r_step][pos]
        vmoy = np.mean(v_selected)

        if np.any((v_selected > 0)):
            vmoy_plus = np.nanmean(v_selected[v_selected > 0])
        else:
            vmoy_plus = np.NaN
        if np.any((v_selected < 0)):
            vmoy_minus = np.nanmean(v_selected[v_selected < 0])
        else:
            vmoy_minus = np.NaN

        if np.isnan(vmoy_plus) and np.isnan(vmoy_minus):
            continue

        for ngate in range(r_step, maxrange):
            if flag_vel[nbeam, ngate] != 0:
                continue
            current_vel = velocity[nbeam, ngate]

            if vmoy >= 0:
                decision = take_decision(vmoy_plus, current_vel, vnyq, alpha=alpha)
                vtrue = unfold(vmoy_plus, current_vel, vnyq)
            else:
                decision = take_decision(vmoy_minus, current_vel, vnyq, alpha=alpha)
                vtrue = unfold(vmoy_minus, current_vel, vnyq)

            if decision == 1:
                final_vel[nbeam, ngate] = current_vel
                flag_vel[nbeam, ngate] = 1
            elif decision == 2:
                final_vel[nbeam, ngate] = vtrue
                flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel

def circle_distance(a, b, circumference):
    """Distance between azimuths a and b on a circle."""
    return np.minimum(np.abs(a - b), np.abs(a - b + circumference))

def correct_closest_reference(azimuth, vel, final_vel, flag_vel, vnyq, alpha=0.8):
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

    if cfg.SHOW_PROGRESS:
        print("correct_closest alpha:", alpha, f"win-azi:{window_azi} win-bin:{window_gate}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    for nbeam in range(maxazi):
        posazi_good, posgate_good = np.where(flag_vel > 0)
        for ngate in range(0, maxrange):
            if flag_vel[nbeam, ngate] != 0:
                continue

            vel1 = vel[nbeam, ngate]

            distance = (circle_distance(posazi_good, nbeam, maxazi) ** 2 +
                        (posgate_good - ngate) ** 2)
            if len(distance) == 0:
                continue

            closest = np.argmin(distance)
            nbeam_close = posazi_good[closest]
            ngate_close = posgate_good[closest]

            npos_range = iter_range(ngate_close, window_gate, maxrange)
            vel_ref_vec = np.zeros(window_azi) + np.NaN

            # Numba doesn't support 2D slice, that's why I loop over things.
            pos = -1
            npos_range_end = npos_range[-1] + 1
            for na in iter_azimuth(azimuth, nbeam_close - window_azi // 2, window_azi):
                pos += 1
                vel_ref_vec[pos] = np.nanmean(
                    final_vel[na, npos_range[0] : npos_range_end][flag_vel[na, npos_range[0] : npos_range_end] > 0]
                )
            velref = np.nanmedian(vel_ref_vec)

            decision = take_decision(velref, vel1, vnyq, alpha=alpha)

            processed = False
            if (decision == 1) or (decision == 0):
                final_vel[nbeam, ngate] = vel1
                flag_vel[nbeam, ngate] = 1
                processed = True

            elif decision == 2:
                vtrue = unfold(velref, vel1, vnyq)
                if is_good_velocity(velref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2
                    processed = True

            if processed and cfg.CLOSEST_NO_DELAY:
                # add newly processed index to pos_good arrays
                # (nopython disallows foo.append(bar))
                posazi_good = np.append(posazi_good, [nbeam])
                posgate_good = np.append(posgate_good, [ngate])

    return final_vel, flag_vel


def correct_box(
    azi, vel, final_vel, flag_vel, vnyq, window_range=20, window_azimuth=10, strategy="surround", alpha=0.8
):
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
    if strategy == "vertex":
        azi_window_offset = window_azimuth
    else:
        azi_window_offset = window_azimuth // 2

    if cfg.SHOW_PROGRESS:
        print("correct_box alpha:", alpha, f"win-azi:{window_azimuth} win-bin:{window_range}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    maxazi, maxrange = final_vel.shape
    for nbeam in np.arange(maxazi - 1, -1, -1):
        for ngate in np.arange(maxrange - 1, -1, -1):
            if flag_vel[nbeam, ngate] != 0:
                continue

            myvel = vel[nbeam, ngate]
            npos_range = iter_range(ngate, window_range, maxrange)

            flag_ref_vec = np.zeros((len(npos_range) * window_azimuth)) + np.NaN
            vel_ref_vec = np.zeros((len(npos_range) * window_azimuth)) + np.NaN

            # I know a slice would be better, but this is for jit to work.
            cnt = -1
            for na in iter_azimuth(azi, nbeam - azi_window_offset, window_azimuth):
                for nr in npos_range:
                    cnt += 1
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


def radial_least_square_check(r, azi, vel, final_vel, flag_vel, vnyq, alpha=0.8):
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
    velbeam_arr = np.zeros(maxrange, dtype=float64)

    COUNT_MIN = 2

    if cfg.SHOW_PROGRESS:
        print("radial_least_sq alpha:", alpha, f"count-min:{COUNT_MIN}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    for nbeam in range(maxazi):
        velbeam_arr = final_vel[nbeam, :]
        if cfg.LEAST_SQ_NO_NAN:
            is_processed_cond = flag_vel[nbeam, :] > 0
        else:
            velbeam_arr[flag_vel[nbeam, :] <= 0] = np.NaN
            is_processed_cond = ~np.isnan(velbeam_arr)

        if len(is_processed_cond) < COUNT_MIN:
            continue

        slope, intercept = linregress(r[is_processed_cond], velbeam_arr[is_processed_cond])

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
                # if we haven't overwritten the already-processed values above,
                # there's no need to update/overwrite
                if not cfg.LEAST_SQ_NO_NAN:
                    final_vel[nbeam, ngate] = myvel
                    flag_vel[nbeam, ngate] = 1
            elif decision == 2:
                myvel = vel[nbeam, ngate]
                vtrue = unfold(mean_vel_ref, myvel, vnyq)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel


def least_square_radial_last_module(r, azi, final_vel, flag_vel, vnyq, alpha=0.8):
    """
    Similar as radial_least_square_check.
    """
    maxazi, maxrange = final_vel.shape
    velbeam_arr = np.zeros(maxrange, dtype=float64)

    COUNT_MIN = 10

    if cfg.SHOW_PROGRESS:
        print("radial_least_sq_last alpha:", alpha, f"count-min:{COUNT_MIN}")
    if not cfg.DO_ACT:
        return final_vel

    # NB: if we only set values provisionally (ie !cfg.LEAST_SQ_NOT_PROVISIONAL)
    # they will NOT be checked by the next stage (box_check)

    for nbeam in range(maxazi):
        velbeam_arr = final_vel[nbeam, :]
        if len(velbeam_arr[~np.isnan(velbeam_arr)]) < COUNT_MIN:
            continue

        slope, intercept = linregress(r[~np.isnan(velbeam_arr)], velbeam_arr[~np.isnan(velbeam_arr)])

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
                # final_vel is good (unchanged)
                if cfg.LEAST_SQ_NOT_PROVISIONAL and flag_vel[nbeam, ngate] == 0:
                    flag_vel[nbeam, ngate] = 1
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, myvel, vnyq)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    if cfg.LEAST_SQ_NOT_PROVISIONAL or flag_vel[nbeam, ngate] > 0:
                        flag_vel[nbeam, ngate] = 2

    return final_vel


def unfolding_3D(
    r_swref,
    azi_swref,
    elev_swref,
    vel_swref,
    flag_swref,
    r_slice,
    azi_slice,
    elev_slice,
    velocity_slice,
    flag_slice,
    original_velocity,
    vnyq,
    window_azi=20,
    window_range=80,
    alpha=0.8,
):
    """
    Dealias using 3D continuity. This function will look at the velocities from
    one sweep (the reference) to the other (the slice).
    Parameters:
    ===========
    r_swref: ndarray
        Range-coordinate of the reference sweep.
    elev_swref: float
        Elevation-coordinate of the reference sweep.
    azi_swref: ndarray
        Azimuth-coordinate the reference sweep.
    vel_swref: ndarray <azimuth, r>
        Velocity of the reference sweep.
    flag_swref:
        Flag array of the reference
    r_slice: ndarray
        Range-coordinate of the sweep to dealias.
    azi_slice: ndarray
        Azimuth of the sweep to dealias.
    elev_slice: float
        Elevation angle of the sweep to dealias.
    velocity_slice: ndarray <azimuth, r>
        2D-dealiased velocity of the sweep to dealias in 3D.
    flag_slice:
        Flag array of the sweep to dealias.
    original_velocity: ndarray <azimuth, r>
        Original aliased velocity field of the sweep to dealias.
    vnyq: float
        Nyquist velocity.
    window_azi: int
        Window size in the azimuth direction
    window_range: int
        Window size in the range direction.

    Returns:
    ========
    velocity_slice: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_slice: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vel_used_as_ref: ndarray <azimuth, range>
        Velocity field used as reference (debugging purposes only).
    processing_flag: ndarray <azimuth, range>
        Flag array that track the decisions made by the algorithm (debugging
        purposes only).
    """
    vel_used_as_ref = np.zeros(velocity_slice.shape)
    processing_flag = np.zeros(velocity_slice.shape) - 3

    maxazi, maxrange = velocity_slice.shape
    ref_range = vel_swref.shape[1]
    # TODO: handle differing ranges
    if ref_range != maxrange:
        print(f"WARNING: unfolding_3D: range counts differ {maxrange} {ref_range}")

    gr_swref = r_swref * np.cos(elev_swref * np.pi / 180)
    gr_slice = r_slice * np.cos(elev_slice * np.pi / 180)

    if cfg.SHOW_PROGRESS:
        print("unfolding_3d alpha:", alpha, f"win-azi:{window_azi} win-bin:{window_range}")
    if not cfg.DO_ACT:
        return velocity_slice, flag_slice, None, None

    for nbeam in range(maxazi):
        for ngate in range(maxrange):
            if flag_slice[nbeam, ngate] == -3:
                # No data here.
                processing_flag[nbeam, ngate] = -2
                continue

            current_vel = velocity_slice[nbeam, ngate]

            rpos_reference = np.argmin(np.abs(gr_swref - gr_slice[ngate]))
            apos_reference = np.argmin(np.abs(azi_swref - azi_slice[nbeam]))

            rpos_iter = iter_range(rpos_reference, window_range, ref_range)

            velocity_refcomp_array = np.zeros((len(rpos_iter) * window_azi)) + np.NaN
            flag_refcomp_array = np.zeros((len(rpos_iter) * window_azi)) - 3

            cnt = -1
            for na in iter_azimuth(azi_swref, apos_reference - window_azi // 2, window_azi):
                for nr in rpos_iter:
                    cnt += 1
                    velocity_refcomp_array[cnt] = vel_swref[na, nr]
                    flag_refcomp_array[cnt] = flag_swref[na, nr]

            if np.sum(flag_refcomp_array != -3) < 1:
                # No comparison possible all gates in the reference are missing.
                processing_flag[nbeam, ngate] = -1
                continue

            compare_vel = np.nanmedian(velocity_refcomp_array[(flag_refcomp_array >= 1)])
            vel_used_as_ref[nbeam, ngate] = compare_vel

            if is_good_velocity(compare_vel, current_vel, vnyq, alpha=alpha):
                processing_flag[nbeam, ngate] = 0
                # The current velocity is in agreement with the lower tilt velocity.
                continue

            ogvel = original_velocity[nbeam, ngate]
            if is_good_velocity(compare_vel, ogvel, vnyq, alpha=alpha):
                # The original velocity was good
                velocity_slice[nbeam, ngate] = ogvel
                flag_slice[nbeam, ngate] = 1
                processing_flag[nbeam, ngate] = 1
            else:
                vtrue = unfold(compare_vel, ogvel, vnyq)
                if is_good_velocity(compare_vel, vtrue, vnyq, alpha=alpha):
                    # New dealiased velocity value found
                    velocity_slice[nbeam, ngate] = vtrue
                    flag_slice[nbeam, ngate] = 2
                    processing_flag[nbeam, ngate] = 2

    return velocity_slice, flag_slice, vel_used_as_ref, processing_flag


def _box_check_v2(refvel, final_vel, flag_vel, vnyq, alpha):
    maxazi, maxrange = final_vel.shape
    for nbeam in range(maxazi):
        for ngate in range(maxrange):
            if flag_vel[nbeam, ngate] <= 0:
                continue

            myvel = final_vel[nbeam, ngate]
            myvelref = refvel[nbeam, ngate]
            if np.isnan(myvelref):
                continue

            if not is_good_velocity(myvelref, myvel, vnyq, alpha=alpha):
                final_vel[nbeam, ngate] = myvelref
                flag_vel[nbeam, ngate] = 3

    return final_vel, flag_vel


if HAVE_NUMBA:
    jit_module(nopython=True, error_model="numpy")


def box_check(final_vel, flag_vel, vnyq, window_range=80, window_azimuth=40, alpha=0.8):
    """
    Check if all individual points are consistent with their surrounding
    velocities based on the median of an area of corrected velocities preceding
    the gate being processed. This module is similar to the dealiasing technique
    from Bergen et al. (1988). This function will look at ALL points.

    Parameters:
    ===========
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

    if cfg.SHOW_PROGRESS:
        print("box_check (cross) alpha:", alpha, f"win-azi:{window_azimuth} win-bin:{window_range}")

    def _vectorized_stride(array, max_time, window, positive_only=True):
        """
        Adapted from:
        https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
        """
        start = 1 - sub_window_size + 1
        if positive_only and start < 0:
            start = 0

        sub_windows = (
            start
            + np.expand_dims(np.arange(sub_window_size), 0)
            + np.expand_dims(np.arange(max_time + 1), 0).T
        )

        if sub_windows.max() > max_time + 1:
            sub_windows = (sub_windows + 1) % max_time

        return array[sub_windows]

    vel_range = final_vel.copy()
    vel_azi = vel_range.copy().T
    
    vectorized_azi = _vectorized_stride(vel_azi, vel_azi.shape[0] - 2, window_azimuth, positive_only=False)
    vectorized_range = _vectorized_stride(vel_range, vel_range.shape[0] - 2, window_range)

    smooth_azi = np.c_[np.mean(vectorized_azi, axis=1).T, np.zeros(vel_azi.shape[1])]
    smooth_range = np.c_[np.mean(vectorized_range, axis=1).T, np.zeros(vel_range.shape[1])].T
    refvel = 0.5 * smooth_azi + 0.5 * smooth_range

    return _box_check_v2(refvel, final_vel.copy(), flag_vel, vnyq, alpha)
