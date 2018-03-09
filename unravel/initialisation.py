"""
Module 2: Initialize the unfolding.

@title: initialisation
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 29/01/2018

Call this function: initialize_unfolding
"""
# Other Libraries
import numpy as np

from numba import jit, int64, float64

# Custom
from .continuity import take_decision, unfold, is_good_velocity


@jit(nopython=True)
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
    maxazi, maxrange = vel.shape
    final_vel = np.zeros((maxazi, maxrange), dtype=float64)
    flag_vel = np.zeros((maxazi, maxrange), dtype=int64)
    vmin = np.zeros((maxazi, ))
    vmax = np.zeros((maxazi, ))
    nsum = np.zeros((maxazi, ))
    dnum = np.zeros((maxazi, ))

    for nazi in range(maxazi):
        vmax[nazi] = np.nanmax(np.abs(vel[nazi, :]))
        nsum[nazi] = np.nansum(np.abs(vel[nazi, :]))

        for ngate in range(maxrange):
            if np.isnan(vel[nazi, ngate]):
                flag_vel[nazi, ngate] = -3
            else:
                dnum[nazi] += 1

    # Compute the normalised integrated velocity along each radials.
    normed_sum = nsum / vmax
    yall = normed_sum / dnum

    threshold = 0.2
    # Looking if the threshold is not too strict.
    iter_radials = np.where(yall < threshold)[0]
    while (len(iter_radials) < 10) and (threshold < 0.8):
        threshold += 0.1
        iter_radials = np.where(yall < threshold)[0]

    # Magic happens.
    for pos_good in iter_radials:
        myvel = vel[pos_good, :]
        if np.sum(np.abs(myvel) >= 0.8 * vnyq) > 3:
            continue

        for ngate in range(3, len(myvel) - 3):
            velref0 = np.nanmedian(myvel[ngate - 3:ngate])
            velref1 = np.nanmedian(myvel[ngate + 1:ngate + 4])
            decision = take_decision(velref0, velref1, vnyq)
            if decision != 1:
                continue

            if np.isnan(velref0):
                velref = velref1
            elif np.isnan(velref1):
                velref = velref0
            else:
                velref = (velref0 + velref1) / 2

            decision = take_decision(velref, myvel[ngate], vnyq)
            if decision == 0:
                continue
            elif decision == 1:
                final_vel[pos_good, ngate] = myvel[ngate]
                flag_vel[pos_good, ngate] = 1
            elif decision == 2:
                vtrue = unfold(myvel[ngate - 1], myvel[ngate])
                if is_good_velocity(myvel[ngate - 1], vtrue, vnyq, alpha=0.4):
                    final_vel[pos_good, ngate] = vtrue
                    flag_vel[pos_good, ngate] = 2

    return final_vel, flag_vel
