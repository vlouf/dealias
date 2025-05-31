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

from typing import List, Tuple

# Other Libraries
import numpy as np

from .cfg import log


def find_reference_radials(velocity: np.ndarray) -> Tuple[int, int]:
    """
    A beam is valid if it contains at least 10 valid gates (not NaN).
    We seek beams that contain the most valid gate, defined by being
    more than the total mean of valid gate per beam. Of these selected
    beams, we are looking for the beam with the minimum absolute mean velocity.

    Parameters:
    ===========
    velocity: ndarray
        Velocity field.

    Returns:
    ========
    start_beam: int
        Index of first reference
    end_beam: int
        Index of end reference
    """

    def find_min_quadrant(azi: np.ndarray, vel: np.ndarray, nvalid_gate_qd: float, nsum_moy: float) -> int:
        return azi[nvalid_gate_qd >= nsum_moy][np.argmin(np.nanmean(np.abs(vel), axis=1)[nvalid_gate_qd >= nsum_moy])]

    def circular_diff(a: float, b: float, mod=360.0) -> float:
        """
        called as circular_diff(a, b, 360) gives the positive difference between
        two angles (in degrees).  cribbed from
        https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
        """
        return (mod / 2) - abs(abs(a - b) - (mod / 2))

    # create azimuth indices
    azi_count = velocity.shape[0]
    azimuth = np.r_[0:azi_count]

    nvalid_gate = np.sum(~np.isnan(velocity), axis=1)
    nvalid_gate[nvalid_gate < 10] = 0
    nsum_tot = np.sum(~np.isnan(velocity[nvalid_gate > 0, :]))
    nvalid_beam = len(azimuth[nvalid_gate > 0])

    nsum_moy = nsum_tot // nvalid_beam
    if nsum_moy > 0.7 * velocity.shape[1]:
        nsum_moy = int(0.7 * velocity.shape[1])

    log(
        f"find_reference_radials vtotal:{nsum_tot} vbeams:{nvalid_beam} total_mean:{nsum_tot // nvalid_beam} beam_thresh:{nsum_moy} valid[0]:{nvalid_gate[0]} valid[1]:{nvalid_gate[1]}"
    )

    try:
        start_beam = find_min_quadrant(azimuth, velocity, nvalid_gate, nsum_moy)
    except ValueError:
        start_beam = azimuth[np.argmin(np.nanmean(np.abs(velocity), axis=1))]

    # find other beam

    SECTOR_COUNT = 4

    # put start_beam in centre of a sector
    azi0 = (start_beam + (azi_count / (2 * SECTOR_COUNT))) % azi_count
    sector_edge = lambda sector: (azi0 + (sector * azi_count) / SECTOR_COUNT) % azi_count

    nb = np.zeros((SECTOR_COUNT,), int)
    for i in range(SECTOR_COUNT):
        sector_start = sector_edge(i)
        sector_end = sector_edge(i + 1)

        if sector_start <= sector_end:
            pos = (azimuth >= sector_start) & (azimuth < sector_end)
        else:
            pos = (azimuth >= sector_start) | (azimuth < sector_end)

        try:
            nb[i] = find_min_quadrant(azimuth[pos], velocity[pos, :], nvalid_gate[pos], nsum_moy)
        except ValueError:
            nb[i] = start_beam  # the worst we can do

    start_diff = lambda a: circular_diff(start_beam, a)
    start_diff_vec = np.vectorize(start_diff)
    end_beam = nb[np.argmax(start_diff_vec(nb))]

    log(f"find_reference_radials radials:{start_beam} {end_beam}")

    # NB: maybe end_beam == start_beam
    return start_beam, end_beam


def get_quadrant(azimuth: np.ndarray, azi_start_pos: int, azi_end_pos: int) -> List[np.ndarray]:
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

    quad = []
    quad.append(iter_plus[: len(iter_plus) // 2])
    quad.append(iter_minus[: len(iter_minus) // 2])
    quad.append(iter_plus[len(iter_plus) // 2 :][::-1])
    quad.append(iter_minus[len(iter_minus) // 2 :][::-1])

    return quad
