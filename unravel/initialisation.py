"""
Module 2: Initialize the unfolding.

@title: initialisation
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 29/01/2018
"""
# Other Libraries
import numpy as np

from numba import jit, int64, float64
from numba import uint32

# Custom
from .continuity import take_decision, unfold, is_good_velocity


@jit(nopython=True)
def find_last_good_vel(j, n, azipos, vflag, nfilter):
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
            Last valid value in slice.
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


@jit(nopython=True)
def first_pass(azi_start_pos, velocity, final_vel, vflag, vnyquist, vshift, delta_vmax, nfilter=5):
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
    nazi, ngate = velocity.shape
    azipos = np.zeros((2 * nazi), dtype=uint32)
    azipos[:nazi] = np.arange(nazi)
    azipos[nazi:] = np.arange(nazi)
    
    for j in range(azi_start_pos, azi_start_pos + nazi // 2) :
        for n in range(ngate):               
            # Build slice for comparison
            j_idx = np.arange(j + 1, j + nfilter + 1)
            j_idx[j_idx >= nazi] -= nazi

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
#                 if vflag[azipos[int(idx_ref)], n] <= 0:
#                     continue
            
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
                        
    return final_vel, vflag


@jit(nopython=True)
def initialize_unfolding(r, azi, azi_start_pos, azi_end_pos, vel, flag_vel, vnyq=13.3):
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
    final_vel = np.zeros(vel.shape, dtype=float64)    
    
    iter_radials = np.array([azi_start_pos - 1, azi_start_pos, 
                             azi_start_pos + 1, azi_end_pos - 1, 
                             azi_end_pos, azi_end_pos + 1])    
    iter_radials[iter_radials >= maxazi] -= maxazi

    # Magic happens.
    is_bad = 0
    for pos_good in iter_radials:
        myvel = vel[pos_good, :]
        if np.sum(np.abs(myvel) >= 0.6 * vnyq) > 3:
            is_bad += 1
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
                vtrue = unfold(myvel[ngate - 1], myvel[ngate], vnyq)
                if is_good_velocity(myvel[ngate - 1], vtrue, vnyq, alpha=0.4):
                    final_vel[pos_good, ngate] = vtrue
                    flag_vel[pos_good, ngate] = 2

    if is_bad > len(iter_radials) - 1:
        for pos_good in iter_radials:
            myvel = vel[pos_good, :]

            for ngate in range(3, len(myvel) - 3):
                if np.abs(myvel[ngate]) >= 0.5 * vnyq:
                    continue

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
                    vtrue = unfold(myvel[ngate - 1], myvel[ngate], vnyq)
                    if is_good_velocity(myvel[ngate - 1], vtrue, vnyq, alpha=0.4):
                        final_vel[pos_good, ngate] = vtrue
                        flag_vel[pos_good, ngate] = 2    
    
    return final_vel, flag_vel