"""
Module 1: Finding reference.

@title: find_reference
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 08/09/2020

.. autosummary::
    :toctree: generated/

    find_reference_radials
    get_quadrant
"""

from .cfg import cfg

# Other Libraries
import numpy as np


def find_reference_radials(azimuth, velocity):
    """
    A beam is valid if it contains at least 10 valid gates (not NaN).
    We seek beams that contain the most valid gate, defined by being
    more than the total mean of valid gate per beam. Of these selected
    beams, we are looking for the beam with the minimum absolute mean velocity.

    Parameters:
    ===========
    azimuth: ndarray
        Azimuth array.
    velocity: ndarray
        Velocity field.

    Returns:
    ========
    start_beam: int
        Index of first reference
    end_beam: int
        Index of end reference
    """

    def find_min_quadrant(azi, vel, nvalid_gate_qd, nsum_moy):
        return azi[nvalid_gate_qd >= nsum_moy][np.argmin(np.nanmean(np.abs(vel), axis=1)[nvalid_gate_qd >= nsum_moy])]

    nvalid_gate = np.sum(~np.isnan(velocity), axis=1)
    nvalid_gate[nvalid_gate < 10] = 0
    nsum_tot = np.sum(~np.isnan(velocity[nvalid_gate > 0, :]))
    nvalid_beam = len(azimuth[nvalid_gate > 0])

    nsum_moy = nsum_tot // nvalid_beam
    if nsum_moy > 0.7 * velocity.shape[1]:
        nsum_moy = int(0.7 * velocity.shape[1])

    if cfg().show_progress:
        print(f"find_reference_radials vtotal:{nsum_tot} vbeams:{nvalid_beam} total_mean:{nsum_tot // nvalid_beam} beam_thresh:{nsum_moy} valid[0]:{nvalid_gate[0]} valid[1]:{nvalid_gate[1]}")

    try:
        start_beam = find_min_quadrant(azimuth, velocity, nvalid_gate, nsum_moy)
    except ValueError:
        start_beam = azimuth[np.argmin(np.nanmean(np.abs(velocity), axis=1))]

    nb = np.zeros((4,))
    for i in range(4):
        pos = (azimuth >= i * 90) & (azimuth < (i + 1) * 90)
        try:
            nb[i] = find_min_quadrant(azimuth[pos], velocity[pos, :], nvalid_gate[pos], nsum_moy)
        except ValueError:
            nb[i] = 9999

    opposition = start_beam + 180
    if opposition >= 360:
        opposition -= 360

    end_beam = nb[np.argmin(np.abs(nb - opposition))]

    if cfg().show_progress:
        print(f"find_reference_radials radials:{start_beam:.1f} {end_beam:.1f}")

    return start_beam, end_beam


def get_quadrant(azimuth, azi_start_pos, azi_end_pos):
    """
    Get the 2 mid-points to end the unfolding azimuthal continuity and divide
    the scan in 4 pieces.

    Parameters:
    ===========
    azimuth: ndarray
        Azimuth array.
    azi_start_pos: int
        Index of first reference.
    azi_end_pos: int
        Index of second reference.

    Returns:
    ========
    quad: List<4>
        Index list of the 4 quandrants.
    """
    nbeam = len(azimuth)
    if azi_start_pos > azi_end_pos:
        iter_plus = np.append(np.arange(azi_start_pos, nbeam), np.arange(0, azi_end_pos + 1))
        iter_minus = np.arange(azi_end_pos, azi_start_pos + 1)[::-1]
    else:
        iter_plus = np.arange(azi_start_pos, azi_end_pos)
        iter_minus = np.append(np.arange(azi_end_pos, nbeam), np.arange(0, azi_start_pos + 1))[::-1]

    quad = [None] * 4
    quad[0] = iter_plus[: len(iter_plus) // 2]
    quad[1] = iter_minus[: len(iter_minus) // 2]
    quad[2] = iter_plus[len(iter_plus) // 2 :][::-1]
    quad[3] = iter_minus[len(iter_minus) // 2 :][::-1]

    return quad
