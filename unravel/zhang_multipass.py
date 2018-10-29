"""
Driver for the implementation of Zhang et al. (2006) technique using UNRAVEL modules.
 - J. Zhang and S. Wang, "An automated 2D multipass Doppler radar velocity
   dealiasing scheme," J. Atmos. Ocean. Technol., vol. 23, no. 9, pp. 1239–1248,
   2006.
"""

# Python Standard Library
import time
import traceback

# Other python libraries.
import numpy as np

# Local
from . import continuity
from . import filtering
from . import initialisation
from . import find_reference


def _zhang_2d(r, azimuth, velocity, elev_angle, nyquist_velocity, debug=False, inherit_flag=None, 
              inherit_azi_start=None, inherit_azi_end=None):
    """
    Dealiasing processing for 2D slice of the Doppler radar velocity field.

    Parameters:
    ===========
    r: ndarray
        Radar scan range.
    azimuth: ndarray
        Radar scan azimuth.
    velocity: ndarray <azimuth, r>
        Aliased Doppler velocity field.
    elev_angle: float
        Elevation angle of the velocity field.
    nyquist_velocity: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array (-3: No data, 0: Unprocessed, 1: Processed - no change -,
                    2: Processed - dealiased.)
    """
    # Make sure velocity is not a masked array.
    try:
        velocity = velocity.filled(np.NaN)
    except Exception:
        pass

    # Parameters from Michel Chong
    vshift = 2 * nyquist_velocity  # By how much the velocity shift when folding    
    delta_vmax = 0.5 * nyquist_velocity  # The authorised change in velocity from one gate to the other.

    # Pre-processing, filtering noise.
    flag_vel = np.zeros(velocity.shape, dtype=int)
    flag_vel[np.isnan(velocity)] = -3
    velocity, flag_vel = filtering.filter_data(velocity, flag_vel, nyquist_velocity, vshift, delta_vmax)
    velocity[flag_vel == -3] = np.NaN

    st_time = time.time()  # tic

    tot_gate = velocity.shape[0] * velocity.shape[1]
    nmask_gate = np.sum(np.isnan(velocity))
    if debug:
        print(f"There are {tot_gate - nmask_gate} gates to dealias at elevation {elev_angle}.")

    start_beam, end_beam = find_reference.find_reference_radials(azimuth, velocity)
    azi_start_pos = np.argmin(np.abs(azimuth - start_beam))
    azi_end_pos = np.argmin(np.abs(azimuth - end_beam))
    # quadrant = find_reference.get_quadrant(azimuth, azi_start_pos, azi_end_pos)

    dealias_vel, flag_vel = initialisation.initialize_unfolding(r, azimuth, azi_start_pos, azi_end_pos, velocity, flag_vel, vnyq=nyquist_velocity)

    vel = velocity.copy()
    vel[azi_start_pos, :] = dealias_vel[azi_start_pos, :]
    delta_vmax = 0.75*nyquist_velocity
    
    dealias_vel, flag_vel = initialisation.first_pass(azi_start_pos, vel, dealias_vel, flag_vel, nyquist_velocity, vshift, delta_vmax)

    dealias_vel, flag_vel = continuity.correct_range_onward_loose(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity, 60)
    dealias_vel, flag_vel = continuity.correct_range_backward_loose(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity, 60)
    
    dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity, dealias_vel, flag_vel,
                                                         np.arange(len(azimuth)), nyquist_velocity, 30)
    dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity, dealias_vel, flag_vel,
                                                                np.arange(len(azimuth))[::-1], nyquist_velocity, 30)

    dealias_vel, flag_vel = continuity.correct_range_onward_loose(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity, 120)
    dealias_vel, flag_vel = continuity.correct_range_backward_loose(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity, 120)
    
    dealias_vel[flag_vel == 0] = velocity[flag_vel == 0]
    
    return dealias_vel, flag_vel, azi_start_pos, azi_end_pos


def unravel(radar, velname="VEL", dbzname="DBZ", gatefilter=None, nyquist_velocity=None, debug=False):
    """
    Process driver.    

    Parameters:
    ===========
    input_file: str
        Input radar file to dealias. File must be compatible with Py-ART.
    gatefilter: Object GateFilter
        GateFilter for filtering noise. If not provided it will automaticaly
        compute it with help of the dual-polar variables.
    vel_name: str
        Name of the velocity field.
    dbz_name: str
        Name of the reflectivity field.
    Returns:
    ========
    ultimate_dealiased_velocity: ndarray
        Dealised velocity field.
    """
    # Filter
    if gatefilter is None:
        gatefilter = filtering.do_gatefilter(radar, velname, dbzname)

    if nyquist_velocity is None:
        nyquist_velocity = radar.instrument_parameters['nyquist_velocity']['data'][0]

    # Start with first reference.
    slice_number = 0
    myslice = radar.get_slice(slice_number)

    r = radar.range['data'].copy()
    velocity = radar.fields[velname]['data'].copy()
    azimuth_reference = radar.azimuth['data'][myslice]
    elevation_reference = radar.elevation['data'][myslice].mean()

    velocity_reference = np.ma.masked_where(gatefilter.gate_excluded, velocity)[myslice]

    # Dealiasing first sweep.
    final_vel, flag_vel, azi_s, azi_e = _zhang_2d(r, azimuth_reference, velocity_reference,
                                                              elevation_reference, nyquist_velocity, debug=False)

    velocity_reference = final_vel.copy()
    flag_reference = flag_vel.copy()

    ultimate_dealiased_velocity = np.zeros(radar.fields[velname]['data'].shape)
    ultimate_dealiased_velocity[myslice] = final_vel.copy()

    for slice_number in range(1, radar.nsweeps):
        if debug:
            print(slice_number)
        myslice = radar.get_slice(slice_number)
        azimuth_slice = radar.azimuth['data'][myslice]
        elevation_slice = radar.elevation['data'][myslice].mean()

        if len(azimuth_slice) < 60:
            print(f"Problem with slice #{slice_number}, only {len(azimuth_slice)} radials.")
            continue

        vel = np.ma.masked_where(gatefilter.gate_excluded, velocity)[myslice]
        velocity_slice = vel.filled(np.NaN)

        flag_slice = np.zeros_like(velocity_slice) + 1
        flag_slice[np.isnan(velocity_slice)] = -3

        final_vel, flag_vel, azi_s, azi_e = _zhang_2d(r, azimuth_slice, velocity_slice,
                                                                  elevation_slice, nyquist_velocity,
                                                                  debug=debug, inherit_flag=flag_slice,
                                                                  inherit_azi_start=azi_s, inherit_azi_end=azi_e)

        ultimate_dealiased_velocity[myslice] = final_vel.copy()

    ultimate_dealiased_velocity = np.ma.masked_where(gatefilter.gate_excluded,
                                                     ultimate_dealiased_velocity)

    return ultimate_dealiased_velocity
