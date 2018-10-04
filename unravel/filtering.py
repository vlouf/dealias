"""
Codes for creating and manipulating gate filters.

@title: filtering
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 20/01/2018

.. autosummary::
    :toctree: generated/

    velocity_texture
    do_gatefilter
"""
import pyart

# Other Libraries
import numpy as np
from numba import jit


def do_gatefilter(radar, vel_name, dbz_name):
    """
    Generate a GateFilter that remove all bad data.
    """
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_outside(dbz_name,  5, 65)
    gf_desp = pyart.correct.despeckle_field(radar, dbz_name, gatefilter=gf)

    return gf_desp


@jit(nopython=True)
def unfold(v, vref, vnq, vshift):
    delv = v - vref

    if(np.abs(delv) < vnq):
        unfld = v
    else:
        unfld = v - int((delv + np.sign(delv) * vnq) / vshift) * vshift
    return unfld


@jit(nopython=True)
def filter_data(velocity, vflag, vnyquist, vshift, delta_vmax, nfilter=10):     
    nazi, ngate = velocity.shape
    for j in range(0, 360):
        for n in range(0, ngate):
            if vflag[j, n] == -3:
                continue

            vmoy = 0
            vmoy_plus = 0
            vmoy_minus = 0

            n1 = n
            n2 = n1 + nfilter
            n2 = np.min(np.array([ngate, n2]))

            idx_selected = vflag[j, n1: n2]
            if np.all((idx_selected == -3)):
                continue

            v_selected = velocity[j, n1: n2][idx_selected != -3]
            vmoy = np.mean(v_selected)

            if np.any((v_selected > 0)):
                vmoy_plus = np.mean(v_selected[v_selected > 0])
            else:
                vmoy_plus = np.NaN
            if np.any((v_selected < 0)):
                vmoy_minus = np.mean(v_selected[v_selected < 0])
            else:
                vmoy_minus = np.NaN

            k = 0
            nselect = np.sum(idx_selected != -3)
            for k in range(nselect):
                vk = v_selected[k]
                dv1 = np.abs(vk - vmoy)
                if dv1 >= delta_vmax:
                    if vmoy >= 0:                        
                        vk_unfld = unfold(vk, vmoy_plus, vnyquist, vshift)
                        dvk = np.abs(vk - vmoy_plus)                        
                    else:                        
                        vk_unfld = unfold(vk, vmoy_minus, vnyquist, vshift)
                        dvk = np.abs(vk - vmoy_minus)                        

                    dvkm = np.abs(vk_unfld - vmoy)
                    if dvkm < delta_vmax or dvk < delta_vmax:
                        velocity[j, n + k] = vk_unfld
                    else:
                        vflag[j, n + k] = -3          
                        
    return velocity, vflag