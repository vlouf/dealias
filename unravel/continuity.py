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
@date: 11/01/2024

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
    box_check
    radial_least_square_check
    least_square_radial_last_module
    unfolding_3D
"""

from typing import Tuple, Union

import numpy as np
from numba import jit, jit_module, int64, float64

from . import cfg
from .cfg import log


def linregress(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Perform a linear regression on two vectors x and y.

    Parameters:
    ===========
        x: ndarray <vector>
        y: ndarray <vector>

    Returns:
    ========
        slope: float
        intercept: float
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


def unfold(v1: float, v2: float, vnyq: float) -> float:
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


def is_good_velocity(vel1: float, vel2: float, vnyq: float, alpha: float = 0.8) -> bool:
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


def iter_azimuth(azi: np.ndarray, idx_start: int, window_len: int) -> np.ndarray:
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


def iter_range(pos_center: int, window_len: int, maxgate: int) -> np.ndarray:
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


def take_decision(velocity_reference: float, velocity_to_check: float, vnyq: float, alpha: float) -> int:
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
    # don't sign-match with near-zero values.  we could probably make this larger
    SIGN_COMPARE_EPSILON = 1e-6

    if np.isnan(velocity_to_check):
        return -3
    elif np.isnan(velocity_reference):
        return 0
    elif is_good_velocity(velocity_reference, velocity_to_check, vnyq, alpha=alpha) or (
        abs(velocity_reference) > SIGN_COMPARE_EPSILON
        and abs(velocity_to_check) > SIGN_COMPARE_EPSILON
        and np.sign(velocity_reference) == np.sign(velocity_to_check)
    ):
        return 1
    else:
        return 2


def correct_clockwise(
    r: np.ndarray,
    azi: np.ndarray,
    vel: np.ndarray,
    final_vel: np.ndarray,
    flag_vel: np.ndarray,
    myquadrant: np.ndarray,
    vnyq: float,
    window_len: int = 3,
    alpha: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
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

    log("correct_clockwise alpha:", alpha, f"win-len:{window_len}")
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


def correct_counterclockwise(
    r: np.ndarray,
    azi: np.ndarray,
    vel: np.ndarray,
    final_vel: np.ndarray,
    flag_vel: np.ndarray,
    myquadrant: np.ndarray,
    vnyq: float,
    window_len: int = 3,
    alpha: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
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

    log("correct_counterclockwise alpha:", alpha, f"win-len:{window_len}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    for nbeam in myquadrant[window_len:]:
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


def correct_range_onward(
    vel: np.ndarray, final_vel: np.ndarray, flag_vel: np.ndarray, vnyq: float, window_len: int = 6, alpha: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
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

    log("correct_range_onward alpha:", alpha, f"win-len:{window_len}")
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


def correct_range_backward(
    vel: np.ndarray, final_vel: np.ndarray, flag_vel: np.ndarray, vnyq: float, window_len: int = 6, alpha: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
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

    log("correct_range_backward alpha:", alpha, f"win-len:{window_len}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    maxazi, maxrange = vel.shape
    for nbeam in range(maxazi):
        for ngate in range(maxrange - 2, -1, -1):
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


def correct_linear_interp(
    velocity: np.ndarray,
    final_vel: np.ndarray,
    flag_vel: np.ndarray,
    vnyq: float,
    r_step: int = 200,
    alpha: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
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

    log("correct_linear_interp (extrapolate) alpha:", alpha, f"window:{r_step}")
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
            vmoy_plus = np.nan
        if np.any((v_selected < 0)):
            vmoy_minus = np.nanmean(v_selected[v_selected < 0])
        else:
            vmoy_minus = np.nan

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

            # NB/TODO: no test/take_decision of unfolded velocity!

            # for mark-good (1):
            # - we mark as good if not actually unfolded
            # - 1e-2: VRAD typically encoded using gain of around 0.1, so smaller differences should be ignored
            if decision == 1 or (decision == 2 and np.isclose(current_vel, vtrue, atol=1e-2)):
                final_vel[nbeam, ngate] = current_vel
                flag_vel[nbeam, ngate] = 1
            elif decision == 2:
                final_vel[nbeam, ngate] = vtrue
                flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel


def circle_distance(a: np.ndarray, b: np.ndarray, circumference: float) -> np.ndarray:
    """Distance between azimuths a and b on a circle.

    NB: will work with numpy values or arrays."""
    return np.minimum(np.abs(a - b), np.abs(a - b + circumference))



def correct_box(
    azi: np.ndarray,
    vel: np.ndarray,
    final_vel: np.ndarray,
    flag_vel: np.ndarray,
    vnyq: float,
    window_range: int = 20,
    window_azimuth: int = 10,
    strategy: str = "surround",
    alpha: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
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

    log("correct_box alpha:", alpha, f"win-azi:{window_azimuth} win-bin:{window_range}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    maxazi, maxrange = final_vel.shape
    for nbeam in np.arange(maxazi - 1, -1, -1):
        for ngate in np.arange(maxrange - 1, -1, -1):
            if flag_vel[nbeam, ngate] != 0:
                continue

            myvel = vel[nbeam, ngate]
            npos_range = iter_range(ngate, window_range, maxrange)

            flag_ref_vec = np.zeros((len(npos_range) * window_azimuth)) + np.nan
            vel_ref_vec = np.zeros((len(npos_range) * window_azimuth)) + np.nan

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


def radial_least_square_check(
    r: np.ndarray,
    azi: np.ndarray,
    vel: np.ndarray,
    final_vel: np.ndarray,
    flag_vel: np.ndarray,
    vnyq: float,
    alpha: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
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

    log("radial_least_sq alpha:", alpha, f"count-min:{COUNT_MIN}")
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    for nbeam in range(maxazi):
        velbeam_arr = final_vel[nbeam, :]
        is_processed_cond = flag_vel[nbeam, :] > 0
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
                # if check is good, keep existing value and flag
                continue

            if decision == 2:
                myvel = vel[nbeam, ngate]

                # if previously unfolded, maybe the original value was good
                if is_good_velocity(mean_vel_ref, myvel, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = myvel
                    flag_vel[nbeam, ngate] = 1
                    continue

                vtrue = unfold(mean_vel_ref, myvel, vnyq)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2

    return final_vel, flag_vel


def least_square_radial_last_module(
    r: np.ndarray, azi: np.ndarray, final_vel: np.ndarray, flag_vel: np.ndarray, vnyq: float, alpha: float = 0.8
) -> np.ndarray:
    """
    Similar as radial_least_square_check.
    """
    maxazi, maxrange = final_vel.shape
    velbeam_arr = np.zeros(maxrange, dtype=float64)

    COUNT_MIN = 10

    log("radial_least_sq_last alpha:", alpha, f"count-min:{COUNT_MIN}")
    if not cfg.DO_ACT:
        return final_vel

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
                # final_vel is good (unchanged): update flag if needed
                if flag_vel[nbeam, ngate] == 0:
                    flag_vel[nbeam, ngate] = 1
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, myvel, vnyq)
                if is_good_velocity(mean_vel_ref, vtrue, vnyq, alpha=alpha):
                    final_vel[nbeam, ngate] = vtrue
                    flag_vel[nbeam, ngate] = 2

    return final_vel




jit_module(nopython=True, error_model="numpy", cache=True)


def box_check(
    azi, final_vel, flag_vel, vnyq, window_range=80, window_azimuth=None, alpha=0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """Call box_check_conv (default) or box_check_v1 (cross filter)."""

    if cfg.USE_BOX_CHECK_V1:
        if window_azimuth is None:
            window_azimuth = 40
        return box_check_v1(final_vel, flag_vel, vnyq, window_range, window_azimuth, alpha)

    if window_azimuth is None:
        window_azimuth = 20
    return box_check_conv(azi, final_vel, flag_vel, vnyq, window_range, window_azimuth, alpha)


def box_check_conv(
    azi: np.ndarray,
    final_vel: np.ndarray,
    flag_vel: np.ndarray,
    vnyq: float,
    window_range: int = 80,
    window_azimuth: int = 20,
    alpha: float = 0.8,
    strategy: str = "surround",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every processed gate, re-fold it if it disagrees with the windowed mean of
    its (window_azimuth x window_range) neighbourhood by >= alpha * vnyq.

    The reference is computed with cumulative sums (O(rays * gates)), with azimuth
    wrapped via iter_azimuth and range clipped at the edges via iter_range.  The
    reference is built from a snapshot of the field at entry (same as box_check_v1),
    so a same-pass correction is not used as a reference for later gates.
    """
    maxazi, maxrange = final_vel.shape
    half = window_range // 2
    W = window_azimuth
    azi_window_offset = W if strategy == "vertex" else W // 2

    log("box_check (conv) alpha:", alpha, f"win-azi:{W} win-bin:{window_range}")

    valid = flag_vel >= 1
    V = np.where(valid, final_vel, 0.0).astype(np.float64)
    M = valid.astype(np.float64)

    # Range box-sum, window [max(0, g-half), min(maxrange, g+half)) -- clipped at
    # the range edges, matching iter_range().
    def _range_sum(x):
        cs = np.empty((maxazi, maxrange + 1), dtype=np.float64)
        cs[:, 0] = 0.0
        np.cumsum(x, axis=1, out=cs[:, 1:])
        g = np.arange(maxrange)
        lo = np.maximum(0, g - half)
        hi = np.minimum(maxrange, g + half)
        return cs[:, hi] - cs[:, lo]

    # Azimuth box-sum, window [b-offset, b-offset+W) wrapped, matching iter_azimuth().
    def _azi_sum(x):
        pad = np.concatenate([x, x[:W]], axis=0)
        cs = np.empty((maxazi + W + 1, maxrange), dtype=np.float64)
        cs[0, :] = 0.0
        np.cumsum(pad, axis=0, out=cs[1:])
        st = (np.arange(maxazi) - azi_window_offset) % maxazi
        return cs[st + W] - cs[st]

    vel_sum = _azi_sum(_range_sum(V))
    cnt = _azi_sum(_range_sum(M))

    with np.errstate(invalid="ignore", divide="ignore"):
        refvel = np.where(cnt > 0, vel_sum / cnt, np.nan)

    # Re-fold gates that disagree with the neighbourhood reference by >= alpha * vnyq.
    # Reference is computed from the snapshot at entry.
    correct = valid & np.isfinite(refvel) & (np.abs(refvel - final_vel) >= alpha * vnyq)
    final_vel[correct] = refvel[correct].astype(final_vel.dtype)
    flag_vel[correct] = 3

    return final_vel, flag_vel


def _windowed_masked_mean(values: np.ndarray, mask: np.ndarray, half: int, W: int, offset: int):
    """Separable masked windowed sum/count over a 2D field.

    Window = azimuth [a-offset, a-offset+W) wrapped (matching iter_azimuth) x range
    [max(0, r-half), min(R, r+half)) clipped (matching iter_range). Returns the
    windowed sum of `values` where `mask` and the windowed count of `mask`.
    """
    A, R = values.shape
    V = np.where(mask, values, 0.0).astype(np.float64)
    M = mask.astype(np.float64)

    def _range_sum(x):
        cs = np.empty((A, R + 1), dtype=np.float64)
        cs[:, 0] = 0.0
        np.cumsum(x, axis=1, out=cs[:, 1:])
        g = np.arange(R)
        return cs[:, np.minimum(R, g + half)] - cs[:, np.maximum(0, g - half)]

    def _azi_sum(x):
        pad = np.concatenate([x, x[:W]], axis=0)
        cs = np.empty((A + W + 1, R), dtype=np.float64)
        cs[0, :] = 0.0
        np.cumsum(pad, axis=0, out=cs[1:])
        st = (np.arange(A) - offset) % A
        return cs[st + W] - cs[st]

    vel_sum = _azi_sum(_range_sum(V))
    cnt = _azi_sum(_range_sum(M))
    return vel_sum, cnt


def _unfold_vec(v1: np.ndarray, v2: np.ndarray, vnyq: float) -> np.ndarray:
    """Vectorised equivalent of unfold(): for each element pick the n in {0,2,4,6}
    that brings v2 closest to v1."""
    d = np.abs(v1 - v2)
    best = None
    best_diff = None
    for n in (0, 2, 4, 6):
        voff = np.where(v1 > 0, v1 + (n * vnyq - d), v1 - (n * vnyq - d))
        diff = np.abs(voff - v1)
        if best is None:
            best, best_diff = voff, diff
        else:
            take = diff < best_diff
            best = np.where(take, voff, best)
            best_diff = np.where(take, diff, best_diff)
    return best


def unfolding_3D(
    r_swref: np.ndarray,
    azi_swref: np.ndarray,
    elev_swref: float,
    vel_swref: np.ndarray,
    flag_swref: np.ndarray,
    r_slice: np.ndarray,
    azi_slice: np.ndarray,
    elev_slice: float,
    velocity_slice: np.ndarray,
    flag_slice: np.ndarray,
    original_velocity: np.ndarray,
    vnyq: float,
    window_azi: int = 20,
    window_range: int = 80,
    alpha: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, Union[None, np.ndarray], Union[None, np.ndarray]]:
    """
    Inter-sweep dealiasing: compare each gate of the slice against the windowed mean
    of the geometrically-matched window in the reference sweep.

    The reference is precomputed with cumulative sums (O(ref_rays * ref_gates)), then
    each slice gate is looked up via azimuth/range geometric mapping.  The mean equals
    the median wherever the reference window is internally consistent (coherent echo);
    they differ only in inconsistent (noise) windows.
    """
    maxazi, maxrange = velocity_slice.shape
    ref_azi, ref_range = vel_swref.shape
    half = window_range // 2
    W = window_azi
    offset = W // 2

    log("unfolding_3d alpha:", alpha, f"win-azi:{W} win-bin:{window_range}")

    gr_swref = r_swref * np.cos(elev_swref * np.pi / 180)
    gr_slice = r_slice * np.cos(elev_slice * np.pi / 180)

    # Windowed masked mean of the reference sweep (same geometry as the exact gather).
    vel_sum, cnt = _windowed_masked_mean(vel_swref, flag_swref >= 1, half, W, offset)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_mean = np.where(cnt > 0, vel_sum / cnt, np.nan)

    # Geometric mapping slice -> reference (azimuth per beam, range per gate).
    apos = np.argmin(circle_distance(azi_swref[None, :], azi_slice[:, None], 360.0), axis=1)
    rpos = np.argmin(np.abs(gr_swref[None, :] - gr_slice[:, None]), axis=1)

    compare = ref_mean[apos[:, None], rpos[None, :]]
    have_ref = cnt[apos[:, None], rpos[None, :]] >= 1

    velocity_slice = velocity_slice.copy()
    flag_slice = flag_slice.copy()

    proc = (flag_slice != -3) & have_ref & np.isfinite(compare)
    cur = velocity_slice
    og = original_velocity
    thr = alpha * vnyq

    good_cur = proc & (np.abs(compare - cur) < thr)
    # Agreement with lower tilt: only upgrade unprocessed gates' flag (value kept).
    flag_slice[good_cur & (flag_slice == 0)] = 1

    rem = proc & ~(np.abs(compare - cur) < thr)
    good_og = rem & (np.abs(compare - og) < thr)
    velocity_slice[good_og] = og[good_og]
    flag_slice[good_og] = 1

    rem2 = rem & ~(np.abs(compare - og) < thr)
    vtrue = _unfold_vec(compare, og, vnyq)
    good_vt = rem2 & (np.abs(compare - vtrue) < thr)
    velocity_slice[good_vt] = vtrue[good_vt]
    flag_slice[good_vt] = 2

    return velocity_slice, flag_slice, None, None


def _box_check_impl(refvel, final_vel, flag_vel, vnyq, alpha) -> Tuple[np.ndarray, np.ndarray]:
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


def box_check_v1(
    final_vel, flag_vel, vnyq, window_range=80, window_azimuth=40, alpha=0.8
) -> Tuple[np.ndarray, np.ndarray]:
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

    def reflect_idx(i, imax):
        """Reflect index at index bounds."""
        if i < 0:
            return -1 - i
        if i >= imax:
            # equivalently: imax - 1 - (i % imax)
            return 2 * imax - 1 - i
        return i

    def _vectorized_stride(array, window, positive_only=True):
        """
        Adapted from:
        https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
        """
        # bound for windowing index
        count0 = array.shape[0]

        # centre window
        start = -(window // 2)

        # create permutation matrix of window indices
        # eg (-1 + [0, 1, 2]) + [0, 1, 2].T
        # ->      [-1, 0, 1]  + [0, 1, 2].T
        # -> [    [-1, 0, 1], [0, 1, 2], [1, 2, 3]]
        sub_windows = (start + np.expand_dims(np.arange(window), 0)) + np.expand_dims(np.arange(count0), 0).T

        # handle index overruns
        if positive_only:  # eg for range indices
            # NB: we should really truncate the window at the edges, but as a
            # compromise we reflect (and some values double-up)
            flat = sub_windows.reshape(-1)
            for i in range(flat.shape[0]):
                flat[i] = reflect_idx(flat[i], count0)
        else:  # wrap (eg for azi indices)
            sub_windows = sub_windows % count0

        return array[sub_windows]

    log("box_check (cross) alpha:", alpha, f"win-azi:{window_azimuth} win-bin:{window_range}")

    vel_azi = final_vel.copy()
    vel_range = final_vel.copy().T

    vectorized_azi = _vectorized_stride(vel_azi, window_azimuth, positive_only=False)
    vectorized_range = _vectorized_stride(vel_range, window_range)

    # NB: windows will be invalidated by a single nan
    smooth_azi = np.mean(vectorized_azi, axis=1)
    smooth_range = np.mean(vectorized_range, axis=1).T

    refvel = 0.5 * smooth_azi + 0.5 * smooth_range

    return _box_check_impl(refvel, final_vel.copy(), flag_vel, vnyq, alpha)


def correct_closest_reference(
    azimuth: np.ndarray,
    vel: np.ndarray,
    final_vel: np.ndarray,
    flag_vel: np.ndarray,
    vnyq: float,
    alpha: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each unprocessed (flag == 0) gate, find the nearest already-dealiased gate
    via a Euclidean distance transform (azimuth treated as circular), use the windowed
    mean of the dealiased field around that gate as a reference, and apply the standard
    unfold/accept decision.

    The nearest-neighbour reference is taken from the set of good gates at entry (not
    updated within the sweep), which is appropriate because this stage acts on residual
    undealiased gates that are overwhelmingly noise.
    """
    from scipy import ndimage

    log("correct_closest alpha:", alpha)
    if not cfg.DO_ACT:
        return final_vel, flag_vel

    maxazi, maxrange = final_vel.shape
    good = flag_vel > 0
    if not good.any():
        return final_vel, flag_vel

    # Windowed masked mean of the dealiased field (window 10 azi x 40 range,
    # matching the exact routine's window_azi=10 / window_gate=40).
    vel_sum, cnt = _windowed_masked_mean(final_vel, good, half=20, W=10, offset=5)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_field = np.where(cnt > 0, vel_sum / cnt, np.nan)

    # Nearest good gate for every gate, with azimuth treated as circular (tile x3).
    g3 = np.concatenate([good, good, good], axis=0)
    _, (ia3, ir3) = ndimage.distance_transform_edt(~g3, return_indices=True)
    ia = ia3[maxazi : 2 * maxazi] % maxazi
    ir = ir3[maxazi : 2 * maxazi]

    velref = ref_field[ia, ir]

    proc = (flag_vel == 0) & np.isfinite(vel)
    isfin = np.isfinite(velref)
    is_good = np.abs(velref - vel) < alpha * vnyq
    same_sign = (np.abs(velref) > 1e-6) & (np.abs(vel) > 1e-6) & (np.sign(velref) == np.sign(vel))

    # decision 0 (no reference) or 1 (consistent): keep the original velocity.
    keep = proc & (~isfin | is_good | same_sign)
    final_vel[keep] = vel[keep]
    flag_vel[keep] = 1

    # decision 2 (folded): try to unfold against the reference.
    dec2 = proc & isfin & ~(is_good | same_sign)
    vtrue = _unfold_vec(velref, vel, vnyq)
    ok = dec2 & (np.abs(velref - vtrue) < alpha * vnyq)
    final_vel[ok] = vtrue[ok].astype(final_vel.dtype)
    flag_vel[ok] = 2

    return final_vel, flag_vel
