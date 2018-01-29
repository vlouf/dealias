"""
Module: utils

Bunch of shared useful functions.

@title: utils
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 29/01/2018
"""

# Other Libraries
import numpy as np


def get_azi_end_pos(azi, azi_start_pos, nb=180):
    end_azi = azi[azi_start_pos] + nb
    if end_azi >= len(azi):
        end_azi -= len(azi)
    if end_azi < 0:
        end_azi += len(azi)
    pos = np.argmin(np.abs(azi - end_azi))
    return pos


def get_iter_pos(azi, st, nb=180):
    if st < 0:
        st += len(azi)
    if st >= len(azi):
        st -= len(azi)
    ed = get_azi_end_pos(azi, st, nb)
    posazi = np.arange(0, len(azi))
    if nb > 0:
        if ed < st:
            mypos = np.append(posazi[st:], posazi[:ed])
        else:
            mypos = posazi[st:ed]
    else:
        if ed > st:
            mypos = np.append(posazi[st::-1], posazi[-1:ed:-1])
        else:
            mypos = posazi[st:ed:-1]

    return mypos


def get_iter_range(r, st):
    rpos = np.arange(len(r))
    if st > 0:
        inward = rpos[st:]
        outward = rpos[(st - 1)::-1]
    else:
        inward = rpos
        outward = None
    return inward, outward
