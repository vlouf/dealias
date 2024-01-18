"""
Codes for creating and manipulating gate filters.

@title: filtering
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 08/09/2020

.. autosummary::
    :toctree: generated/

    do_gatefilter
    unfold
    filter_data
"""

from . import cfg

# Other Libraries
import numpy as np
from numba import jit


def do_gatefilter(radar, dbz_name: str):
    """
    Generate a GateFilter that remove all bad data.

    Parameters:
    ===========
    radar: pyart.core.Radar
        Radar pyart object.
    dbz_name: str
        Reflectivity field name.

    Returns:
    ========
    gf_desp: pyart.filters.GateFilter
        GateFilter object.
    """
    import pyart

    gf = pyart.filters.GateFilter(radar)
    gf.exclude_outside(dbz_name, -15, 70)
    gf_desp = pyart.correct.despeckle_field(radar, dbz_name, gatefilter=gf)

    return gf_desp


@jit(nopython=True)
def unfold(v: float, vref: float, vnq: float, vshift: float) -> float:
    """
    Unfold velocity.

    Parameters:
    ===========
    v: float
        Velocity to unfold.
    vref: float
        Reference velocity.
    vnq: float
        Nyquist velocity.
    vshift: float
        Allowed shift (twice the Nyquist co-interval.)

    Returns:
    ========
    unfld: float
        Unfolded velocity.
    """
    delv = v - vref

    if np.abs(delv) < vnq:
        unfld = v
    else:
        unfld = v - int((delv + np.sign(delv) * vnq) / vshift) * vshift
    return unfld


@jit(nopython=True)
def filter_data(velocity, vflag, vnyquist, vshift, alpha, nfilter=10):
    """
    Filter data (despeckling) using MAD and first quick attempt at unfolding 
    velocity.

    Parameters:
    ===========
    velocity: ndarray
        Velocity field.
    vflag: ndarray
        Flag array.
    vnyquist: float
        Nyquist velocity.
    vshift: float
        Allowed shift.
    alpha: float
        Trusted velocity difference Nyquist multiplier.
    nfilter: int
        Window size.
    
    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    """
    nrays = velocity.shape[0]
    ngate = velocity.shape[1]
    delta_vmax = vnyquist * alpha;

    if cfg.SHOW_PROGRESS:       # nopython cfg().show_progress
        print("filter_data MAD alpha:", alpha)
    if not cfg.DO_ACT:          # nopython cfg().do_act
        return velocity, vflag

    for j in range(0, nrays):
        for n in range(0, ngate):
            if vflag[j, n] == -3:
                continue

            vmoy = 0
            vmoy_plus = 0
            vmoy_minus = 0

            n1 = n
            n2 = n1 + nfilter
            n2 = np.min(np.array([ngate, n2]))

            idx_selected = vflag[j, n1:n2]
            if np.all((idx_selected == -3)):
                continue

            v_selected = velocity[j, n1:n2][idx_selected != -3]
            vmoy = np.median(v_selected)

            if np.any((v_selected > 0)):
                vmoy_plus = np.median(v_selected[v_selected > 0])
            else:
                vmoy_plus = np.NaN
            if np.any((v_selected < 0)):
                vmoy_minus = np.median(v_selected[v_selected < 0])
            else:
                vmoy_minus = np.NaN

            for k in range(n1, n2):
                if vflag[j, k] == -3:
                    continue
                vk = velocity[j, k]
                dv1 = np.abs(vk - vmoy)
                if dv1 >= delta_vmax:
                    if vmoy >= 0:
                        vk_unfld = unfold(vk, vmoy_plus, vnyquist, vshift)
                        dvk = np.abs(vk_unfld - vmoy_plus)
                    else:
                        vk_unfld = unfold(vk, vmoy_minus, vnyquist, vshift)
                        dvk = np.abs(vk_unfld - vmoy_minus)

                    dvkm = np.abs(vk_unfld - vmoy)
                    if dvkm < delta_vmax or dvk < delta_vmax:
                        velocity[j, k] = vk_unfld

    return velocity, vflag
