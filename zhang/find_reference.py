"""
Module 1: Finding reference.

@title: find_reference
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 29/01/2018
"""

# Other Libraries
import numpy as np


def get_static_rays(vel):
    """
    Compute the number of static gate (close to 0 m/s) and returns the best
    azimuths.

    To be a reference radial, four criteria must be met. First, they are minima
    in the curve of the normalized average of the absolute values of the
    measured velocities at all the valid gates along each radial.

    Parameter:
    ==========
    nvel: array [azi, range]
        velocity field.

    Returns:
    ========
    minpos: array
        Sorted array of the best azimuths.
    """
    try:
        nvel = vel.filled(np.NaN)
    except AttributeError:
        nvel = vel

    # Criterion 1: Top third of valid gates
    sum_good = np.sum(~np.isnan(nvel), axis=1)
    valid_pos = (sum_good / np.max(sum_good) > 2 / 3)

    n = np.sum(np.abs(vel), axis=1) / np.max(np.abs(vel), axis=1)
    d = np.sum(~np.isnan(nvel), axis=1)
    yall = n / d

    minpos = np.argsort(yall)[valid_pos]

    return minpos


def get_opposite_azimuth(myazi, tolerance=20):
    """
    Get the opposite azimuth plus/minus a tolerance.

    To be a reference radial, four criteria must be met.
    Second, these two initial reference radials should be separated by
    approximately 180.

    Parameters:
    ===========
    myazi: int
        Azimuth angle.
    tolerance: int
        Range of tolerance for the azimuth.

    Returns:
    ========
    minazi: int
        Opposite angle minimun range
    maxazi: int
        Opposite angle maximum range
    """
    azi_range = 180 - tolerance
    minazi = myazi + azi_range
    maxazi = myazi - azi_range
    if minazi > 360:
        minazi -= 360
    if maxazi < 0:
        maxazi += 360

    return [minazi, maxazi]


def get_valid_rays(vel):
    """
    Compute the quantity of valid gates for each rays, and returns the best
    azimuths.

    To be a reference radial, four criteria must be met.
    The fourth criterion is that the number of data points for the radial with
    the minimum sum must contain at least two-thirds of the average number
    of valid gates in all the radials in all azimuths


    Parameter:
    ==========
    nvel: array [azi, range]
        velocity field.

    Returns:
    ========
    extpos: array
        Sorted array of the best azimuths.
    """
    try:
        nvel = vel.filled(np.NaN)
    except AttributeError:
        nvel = vel
    # Criterion 1: Top third of valid gates
    sum_good = np.sum(~np.isnan(nvel), axis=1)
    valid_pos = (sum_good / np.max(sum_good) > 2 / 3)

    y2all = (np.max(np.abs(vel), axis=1) - np.min(np.abs(vel), axis=1))
#     y2 = y2all[valid_pos]

    extpos = np.argsort(y2all)[valid_pos]

    return extpos


def find_reference_radials(azi, vel, debug=False):
    """
    A reference radial is one that exhibits little or no aliasing. The most
    likely position for this to occur is where the wind direction is almost
    orthogonal to the direction the antenna is pointing. Also, the average value
    of the absolute value of that radial's Doppler velocity will be at a minimum.

    To be a reference radial, four criteria must be met. First, they are minima
    in the curve of the normalized average of the absolute values of the
    measured velocities at all the valid gates along each radial.
    Second, these two initial reference radials should be separated by
    approximately 180.

    Parameter:
    ==========
    azi
    vel
    rhohv

    Returns:
    ========
    minpos: array
        Sorted array of the best azimuths.
    """
    pos_valid = get_valid_rays(vel)
    pos_static = get_static_rays(vel)

    # Finding intersects of criteria 1 to 3.
    weight_valid = np.arange(0, len(pos_valid), 1)
    weight_static = np.arange(0, len(pos_static), 1)

    total_weight = np.zeros(len(pos_valid)) + np.NaN
    for cnt, (one_valid, one_valid_weight) in enumerate(zip(pos_valid, weight_valid)):
        try:
            one_static_weight = weight_static[one_valid == pos_static][0]
        except IndexError:
            one_static_weight = 9999

        total_weight[cnt] = one_static_weight + one_valid_weight

    pos1 = pos_valid[np.argmin(total_weight)]

    # Finding the 2nd radial of reference
    pos2 = pos1 + len(azi) // 2
    if pos2 > len(azi):
        pos2 -= len(azi)

#     try:
#         ref2_range_min, ref2_range_max = get_opposite_azimuth(azi[pos1])
#         if ref2_range_min < ref2_range_max:
#             goodpos = np.where((azi >= ref2_range_min) & (azi <= ref2_range_max))[0]
#         else:
#             goodpos = np.where((azi >= ref2_range_min) | (azi <= ref2_range_max))[0]

#         rslt = [(a, total_weight[a == pos_valid][0]) for a in goodpos if a in pos_valid]
#         opposite_pos, opposite_weight = zip(*rslt)
#         pos2 = opposite_pos[np.argmin(opposite_weight)]
#     except Exception:
#         pos2 = pos1 + len(azi) // 2
#         if pos2 > len(azi):
#             pos2 -= len(azi)
    if debug:
        print(f"References are azimuths {azi[pos1]} and {azi[pos2]}, i.e. azimuthal positions {pos1} and {pos2}.")

    return pos1, pos2


def get_quadrant(azi, posang1, posang2, full=False):
    """
    Compute the 4 part of the quadrant based on the 2 reference radials
    Quadrant 1 : reference radial 1 -> clockwise
    Quadrant 2 : reference radial 1 -> counter-clockwise
    Quadrant 3 : reference radial 2 -> clockwise
    Quadrant 4 : reference radial 2 -> counter-clockwise
    """
    ang1, ang2 = posang1, posang2
    maxazipos = len(azi)

    def get_sl(a, b, clock=1):
        if clock == 1:
            if a < b:
                return list(range(a, b + 1))
            else:
                return [*range(a, maxazipos), *range(0, b + 1)]
        else:
            if a > b:
                return list(range(a, b - 1, -1))
            else:
                return [*range(a, -1, -1), *range(maxazipos - 1, b - 1, -1)]

    if ang1 > ang2:
        dist1 = ang1 - ang2
        dist2 = maxazipos - dist1
        mid1 = ang1 + dist1 // 2
        if mid1 >= maxazipos:
            mid1 -= maxazipos

        mid2 = ang1 - dist2 // 2
        if mid2 < 0:
            mid2 += maxazipos

        quad = [None] * 4
        if not full:
            quad[0] = get_sl(ang1, mid1, 1)
            quad[1] = get_sl(ang1, mid2, -1)
            quad[2] = get_sl(ang2, mid2, 1)
            quad[3] = get_sl(ang2, mid1, -1)
        else:
            quad[0] = get_sl(ang1, ang2, 1)
            quad[1] = get_sl(ang1, ang2, -1)
            quad[2] = get_sl(ang2, ang1, 1)
            quad[3] = get_sl(ang2, ang1, -1)

    else:
        dist1 = ang2 - ang1
        dist2 = maxazipos - dist1
        mid1 = ang1 + dist1 // 2
        if mid1 >= maxazipos:
            mid1 -= maxazipos

        mid2 = ang1 - dist2 // 2
        if mid2 < 0:
            mid2 += maxazipos

        quad = [None] * 4
        if not full:
            quad[0] = get_sl(ang1, mid1, 1)
            quad[1] = get_sl(ang1, mid2, -1)
            quad[2] = get_sl(ang2, mid2, 1)
            quad[3] = get_sl(ang2, mid1, -1)
        else:
            quad[0] = get_sl(ang1, ang2, 1)
            quad[1] = get_sl(ang1, ang2, -1)
            quad[2] = get_sl(ang2, ang1, 1)
            quad[3] = get_sl(ang2, ang1, -1)

    return quad
