"""
Module 2: Initialize the unfolding.

@title: initialisation
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 29/01/2018

!!!! CAREFUL: vnyq is worth half the nyquist velocity. !!!!

Call this function: initialize_unfolding
"""
# Other Libraries
import numpy as np

from numba import jit

# Custom
from .utils import *


def is_good_velocity(vel1, vel2, vnyq, alpha=0.8):
    return np.abs(vel2 - vel1) < alpha * vnyq


@jit(nopython=True)
def _check_initialisation(final_vel, flag_vel, vnyq):
    """
    Check if the initial reference radials are properly unfolded and change the
    flag array value accordingly.

    Parameters:
    ===========
    final_vel: array <azimuth, range>
        Unfolded velocities.
    flag_vel: array <azimuth, range>
        Flag array for velocity processing (0: unprocessed, 1:processed, 2:unfolded, -3: missing)
    vnyq: float
        Half-nyquist velocity.

    Returns:
    ========
    flag_vel: array <azimuth, range>
        Updated flag array for velocity processing (0: unprocessed, 1:processed, 2:unfolded, -3: missing)
    """
    # Repeating the function so that @jit works.
    def is_good_velocity(vel1, vel2, vnyq, alpha=0.8):
        return np.abs(vel2 - vel1) < alpha * vnyq

    for nazi in range(0, final_vel.shape[0]):
        if not np.any(flag_vel[nazi, :] == 2):
            continue
        for ngate in range(1, final_vel.shape[1]):
            if flag_vel[nazi, ngate] < 0:
                continue

            vel1 = final_vel[nazi, ngate]

            npos = ngate - 1
            vel0 = final_vel[nazi, npos]
            flag0 = flag_vel[nazi, npos]
            while (flag0 <= 0) & (npos > 0):
                npos -= 1
                vel0 = final_vel[nazi, npos]
                flag0 = flag_vel[nazi, npos]

            if is_good_velocity(vel0, vel1, vnyq):
                flag_vel[nazi, ngate] = 1
            else:
                flag_vel[nazi, ngate] = 0

    return flag_vel


def _unfold_reference_radials(azi, vel, final_vel, flag_vel, azi_ref_pos, vnyq=13.3):
    """
    Unfold the reference radials.

    Parameters:
    ===========
    vel: array <azimuth, range>
        Raw velocity.
    final_vel: array <azimuth, range>
        Unfolded velocities.
    flag_vel: array <azimuth, range>
        Flag array for velocity processing (0: unprocessed, 1:processed, 2:unfolded, -3: missing)
    azi_ref_pos: int
        Azimuth reference position.
    vnyq: float
        Half-nyquist velocity.

    Returns:
    ========
    final_vel: array <azimuth, range>
        Updated unfolded velocities.
    flag_vel: array <azimuth, range>
        Updated flag array for velocity processing (0: unprocessed, 1:processed, 2:unfolded, -3: missing)
    """
    # Initial reference radials.
    ref_vel = vel[azi_ref_pos, :]
    final_vel[azi_ref_pos, :] = ref_vel.filled(0)
    flag_vel[azi_ref_pos, ~ref_vel.mask] = 1
    flag_ref = flag_vel[azi_ref_pos, :]
    ref_vel = ref_vel.filled(0)

    # Unfold the first 3 radials (very strict process here: if not valid then don't exists!).
    # Defining the processing flag 0: unproc, 1: proc, 2: part_proc, -3: missing/wrong.
    for nazi in get_iter_pos(azi, azi_ref_pos + 1, 3):
        for ngate in range(0, vel.shape[1]):
            vel0 = ref_vel[ngate]
            vel1 = vel[nazi, ngate]
            flag0 = flag_ref[ngate]

            # Missing reference data
            if flag0 == -3:
                final_vel[nazi, ngate] = vel1
                continue

            # Missing data to be processed
            if np.ma.is_masked(vel1):
                flag_vel[nazi, ngate] = -3
                continue

            if is_good_velocity(vel0, vel1, vnyq=vnyq):
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
            else:
                flag_vel[nazi, ngate] = -3

    return final_vel, flag_vel


def initialize_unfolding(r, azi, azi_start_pos, azi_end_pos, vel, vnyq=13.3):
    """
    Initialize the unfolding procedure and unfold the reference radials..

    Parameters:
    ===========

    Returns:
    ========
    final_vel: array <azimuth, range>
        Unfolded velocities.
    flag_vel: array <azimuth, range>
        Flag array for velocity processing (0: unprocessed, 1:processed, 2:unfolded, -3: missing)
    """
    # Initialize stuff.
    final_vel = np.zeros(vel.shape)
    flag_vel = np.zeros(vel.shape, dtype=int)
    flag_vel[vel.mask] = -3

    #  Initial 3 radials.
    final_vel, flag_vel = _unfold_reference_radials(azi, vel, final_vel, flag_vel, azi_start_pos)
    final_vel, flag_vel = _unfold_reference_radials(azi, vel, final_vel, flag_vel, azi_end_pos)

    # Compute the normalised integrated velocity along each radials.
    n = np.sum(np.abs(vel), axis=1) / np.max(np.abs(vel), axis=1)
    d = np.sum(~vel.mask, axis=1)
    yall = n / d
    # Magic happens.
    for a in np.where(yall < 0.4)[0]:
        if any(flag_vel[a, :] == 1):
            continue
        final_vel[a, :] = vel[a, :]
        flag_vel[a, ~vel[a, :].mask] = 2

    vmax = np.nanmax(vel, axis=1)
    vmin = np.nanmax(vel, axis=1)

    # Looking for all radials that never come close of the nyquist.
    pos0 = (vmax < 0.8 * vnyq) & (vmin > 0.8 * -vnyq)

    final_vel[pos0, :] = vel[pos0, :].filled(np.NaN)
    flag_vel[pos0, :] = 1

    flag_vel = _check_initialisation(final_vel, flag_vel, vnyq)

    return final_vel, flag_vel
