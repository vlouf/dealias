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

# Other Libraries
import pyart
import numpy as np


def velocity_texture(radar, vel_name):
    """
    Compute velocity texture using new Bobby Jackson function in Py-ART.

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    vel_name: str
        Name of the (original) Doppler velocity field.

    Returns:
    ========
    vdop_vel: dict
        Velocity texture.
    """

    try:
        v_nyq_vel = radar.instrument_parameters['nyquist_velocity']['data'][0]
    except Exception:
        vdop_art = radar.fields[vel_name]['data']
        v_nyq_vel = np.max(np.abs(vdop_art))

    vel_dict = pyart.retrieve.calculate_velocity_texture(radar, vel_name, nyq=v_nyq_vel)

    return vel_dict


def do_gatefilter(radar, vel_name, dbz_name, zdr_name=None, rho_name=None):
    """
    Generate a GateFilter that remove all bad data.
    """

    tvel = velocity_texture(radar, vel_name)
    radar.add_field("TVEL", tvel)

    gf = pyart.filters.GateFilter(radar)
    if zdr_name is not None:
        gf.exclude_outside(zdr_name, -3.0, 7.0)

    gf.exclude_outside(dbz_name, -40.0, 80.0)
    gf.exclude_above("TVEL", 4)

    if rho_name is not None:
        gf.include_above(rho_name, 0.8)

    gf_desp = pyart.correct.despeckle_field(radar, "DBZ", gatefilter=gf)

    return gf_desp
