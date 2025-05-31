"""
Module initialize the unfolding.

@title: initialisation
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 31/05/2025

.. autosummary::
    :toctree: generated/

    find_last_good_vel
    flipud
    first_pass
    initialize_unfolding
"""

from typing import Tuple

import numpy as np
from numba import jit_module
from numba import uint32

# Custom
from . import cfg
from .cfg import log
from .continuity import take_decision, unfold, is_good_velocity


def find_last_good_vel(j: int, n: int, azipos: np.ndarray, vflag: np.ndarray, nfilter: int) -> int:
    """
    Looking for the last good (i.e. processed) velocity in a slice.

    Parameters:
    ===========
    j: int
        Position in dimension 1
    n: int
        Position in dimension 2
    azipos: array
        Array of available position in dimension 1
    vflag: ndarray <azimuth, range>
        Flag array (-3: missing, 0: unprocessed, 1: processed, 2: processed and unfolded)
    nfilter: int
        Size of filter.

    Returns:
    ========
        idx_ref: int
            Last valid value in slice. Returns -999 if no valid velocity is found.
    """
    i = 0
    while i < nfilter:
        i += 1
        idx_ref = j - i
        idx = azipos[idx_ref]
        vflag_ref = vflag[idx, n]
        if vflag_ref > 0:
            return idx_ref
    return -999


def flipud(arr: np.ndarray) -> np.ndarray:
    """
    Numpy's flipud function is not supported by numba for some reasons...
    So here it is.
    """
    return arr[::-1, :]  # Soooo complex!


def first_pass(
    azi_start_pos: int,
    velocity: np.ndarray,
    final_vel: np.ndarray,
    vflag: np.ndarray,
    vnyquist: float,
    delta_vmax: float,
    nfilter: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    First pass: continuity check along the azimuth, starting at azi_start_pos.

    Parameters:
    ===========
    azi_start_pos: int
        Starting position in array alongside dimension 0.
    velocity: ndarray <azimuth, range>
        Velocity to dealias
    final_vel: ndarray <azimuth, range>
        Result array
    vflag: ndarray <azimuth, range>
        Flag array (-3: missing, 0: unprocessed, 1: processed, 2: processed and unfolded)
    vnyquist: float
        Nyquist velocity.
    vshift: float
        Shift expected when folding, i.e. 2*vnyquist (in general)
    delta_vmax: float
        Maximum velocity difference tolerated between two contiguous gates.
    nfilter: int
        Size of filtering window

    Returns:
    ========
    final_vel: array <azimuth, range>
        Unfolded velocities.
    flag_vel: array <azimuth, range>
        Flag array for velocity processing (0: unprocessed, 1:processed, 2:unfolded, -3: missing)
    """
    log("init-clock alpha:", delta_vmax / vnyquist, f"radial:{azi_start_pos}")
    if not cfg.DO_ACT:
        return final_vel, vflag

    num_azi, ngate = velocity.shape
    azipos = np.zeros((2 * num_azi), dtype=uint32)
    azipos[:num_azi] = np.arange(num_azi)
    azipos[num_azi:] = np.arange(num_azi)

    for mypass in range(2):
        if mypass == 1:
            velocity = flipud(velocity)
            final_vel = flipud(final_vel)
            vflag = flipud(vflag)
            azi_start_pos = num_azi - azi_start_pos - 1

        for j in range(azi_start_pos, azi_start_pos + num_azi // 2):
            for n in range(ngate):
                # Build slice for comparison
                j_idx = np.arange(j + 1, j + nfilter + 1)
                j_idx[j_idx >= num_azi] -= num_azi

                idx_selected = vflag[j_idx, n]
                if np.all((idx_selected == -3)):
                    # All values missing in slice.
                    continue
                # Selection of velocity to dealias (j_idx is an array)
                v_selected = velocity[j_idx, n]

                # Searching reference
                idx_ref = j
                # Looking for last processed value for reference, within a nfilter distance.
                if vflag[azipos[j], n] <= 0:
                    idx_ref = find_last_good_vel(j, n, azipos, vflag, nfilter)
                    if idx_ref == -999:
                        continue

                # Reference velocity
                vref = final_vel[azipos[int(idx_ref)], n]

                # Dealiasing slice
                for k in range(len(v_selected)):
                    if idx_selected[k] == -3:
                        continue

                    vk = v_selected[k]
                    dv1 = np.abs(vk - vref)
                    if dv1 < delta_vmax:
                        # No need to dealias
                        final_vel[j_idx[k], n] = vk
                        vflag[j_idx[k], n] = 1
                    else:
                        vk_unfld = unfold(vref, vk, vnyquist)
                        dvk = np.abs(vk_unfld - vref)

                        if dvk < delta_vmax:
                            final_vel[j_idx[k], n] = vk_unfld
                            vflag[j_idx[k], n] = 2

    velocity = flipud(velocity)
    final_vel = flipud(final_vel)
    vflag = flipud(vflag)

    return final_vel, vflag


def initialize_unfolding(
    azi_start_pos: int, azi_end_pos: int, vel: np.ndarray, flag_vel: np.ndarray, vnyq: float = 13.3
) -> Tuple[np.ndarray, np.ndarray]:
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
    maxazi = vel.shape[0]
    final_vel = vel.copy()

    iter_radials_init = np.array([azi_start_pos - 1, azi_start_pos, azi_start_pos + 1])
    iter_radial_list = [iter_radials_init]
    if azi_end_pos != azi_start_pos:
        iter_radials_last = np.array([azi_end_pos - 1, azi_end_pos, azi_end_pos + 1])
        iter_radial_list.append(iter_radials_last)

    alpha = 0.4  # for unfolding in take_decision()

    log("init-unfold-radial alpha:", alpha, f"radials:{azi_start_pos} {azi_end_pos}")
    if not cfg.DO_ACT:
        return vel, flag_vel

    thresholded_vel = np.abs(vel) >= 0.6 * vnyq  # Precompute the thresholded array

    for iter_radials in iter_radial_list:
        iter_radials[iter_radials >= maxazi] -= maxazi
        iter_radials[iter_radials < 0] += maxazi

        # Magic happens.
        is_bad = 0
        for pos_good in iter_radials:
            myvel = vel[pos_good, :]
            if np.sum(thresholded_vel[pos_good, :]) > 3:  # Use precomputed array
                is_bad += 1
                continue

            for ngate in range(3, len(myvel) - 3):
                velref0 = np.nanmedian(myvel[ngate - 3 : ngate])
                velref1 = np.nanmedian(myvel[ngate + 1 : ngate + 4])
                decision = take_decision(velref0, velref1, vnyq, alpha=alpha)
                if decision != 1:
                    continue

                if np.isnan(velref0):
                    velref = velref1
                elif np.isnan(velref1):
                    velref = velref0
                else:
                    velref = (velref0 + velref1) / 2

                decision = take_decision(velref, myvel[ngate], vnyq, alpha=alpha)
                if decision == 0:
                    continue
                elif decision == 1:
                    final_vel[pos_good, ngate] = myvel[ngate]
                    flag_vel[pos_good, ngate] = 1
                elif decision == 2:
                    vtrue = unfold(velref, myvel[ngate], vnyq)
                    if is_good_velocity(velref, vtrue, vnyq, alpha=alpha):
                        final_vel[pos_good, ngate] = vtrue
                        flag_vel[pos_good, ngate] = 2

        if is_bad > len(iter_radials) - 1:
            for pos_good in iter_radials:
                myvel = vel[pos_good, :]

                for ngate in range(3, len(myvel) - 3):
                    if np.abs(myvel[ngate]) >= 0.5 * vnyq:
                        continue

                    velref0 = np.nanmedian(myvel[ngate - 3 : ngate])
                    velref1 = np.nanmedian(myvel[ngate + 1 : ngate + 4])
                    decision = take_decision(velref0, velref1, vnyq, alpha=alpha)
                    if decision != 1:
                        continue

                    if np.isnan(velref0):
                        velref = velref1
                    elif np.isnan(velref1):
                        velref = velref0
                    else:
                        velref = (velref0 + velref1) / 2

                    decision = take_decision(velref, myvel[ngate], vnyq, alpha=alpha)
                    if decision == 0:
                        continue
                    elif decision == 1:
                        final_vel[pos_good, ngate] = myvel[ngate]
                        flag_vel[pos_good, ngate] = 1
                    elif decision == 2:
                        vtrue = unfold(velref, myvel[ngate], vnyq)
                        if is_good_velocity(velref, vtrue, vnyq, alpha=alpha):
                            final_vel[pos_good, ngate] = vtrue
                            flag_vel[pos_good, ngate] = 2

    return final_vel, flag_vel


# Compiles all functions in the module using Numba's JIT for performance optimization.
jit_module(nopython=True, error_model="numpy", cache=True)
