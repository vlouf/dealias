# Python Standard Library
import os
import glob
import warnings

# Other python libraries.
import pyart
import netCDF4
import numpy as np
import matplotlib.pyplot as pl

from numba import jit
from scipy.stats import linregress

# Local
import filtering
import find_reference


def dealias_process(radar):
    # Filter bad data.
    gf = filtering.do_gatefilter(radar)

    # Extract velocity field, range and azimuth.
    sl = radar.get_slice(0)
    r = radar.range['data'].copy()
    azi = radar.azimuth['data'][sl].copy()
    velocity = radar.fields['VEL']['data'].copy()

    vel = np.ma.masked_where(gf.gate_excluded, velocity)[sl]
    nvel = vel.filled(np.NaN)

    maxazi = len(azi)
    maxgate = len(r)

    # Find reference radials and make the quadrant.
    azi_start_pos, azi_end_pos = find_reference.find_reference_radials(azi, vel)
    quad = find_reference.get_quadrant(azi, azi_start_pos, azi_end_pos)

    final_vel, flag_vel = initialize_unfolding(r, azi, azi_start_pos, azi_end_pos, vel)
    plot_radar(final_vel, flag_vel)
