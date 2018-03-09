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
