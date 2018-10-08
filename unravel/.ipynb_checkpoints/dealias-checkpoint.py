"""
Driver script for the dealiasing module. You want to call process_3D.

Even though most of the things here are new, it is still based (or inspired) by
the works of:
 - J. Zhang and S. Wang, "An automated 2D multipass Doppler radar velocity
   dealiasing scheme," J. Atmos. Ocean. Technol., vol. 23, no. 9, pp. 1239–1248,
   2006.
 - G. He, G. Li, X. Zou, and P. S. Ray, "A velocity dealiasing scheme for
   synthetic C-band data from China’s new generation weather radar system
   (CINRAD)," J. Atmos. Ocean. Technol., vol. 29, no. 9, pp. 1263–1274, 2012.
 - G. Li, G. He, X. Zou, and P. S. Ray, "A velocity dealiasing scheme for
   C-band weather radar systems," Adv. Atmos. Sci., vol. 31, no. 1, pp. 17–26,
   2014.

@title: dealias
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 05/04/2018
@date:04/10/2018

    count_proc
    dealiasing_process_2D
    process_3D
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


def count_proc(myflag, debug=False):
    """
    Count how many gates are left to dealias.

    Parameters:
    ===========
    myflag: ndarray (int)
        Processing flag array.
    debug: bool
        Print switch.

    Returns:
    ========
    perc: float
        Percentage of gates processed.
    """
    count = np.sum(myflag == 0)
    total = myflag.size
    perc = (total - count) / total * 100
    if debug:
        print(f"Still {count} gates left to dealias. {perc:0.1f}% done.")
    return perc


def dealiasing_process_2D(r, azimuth, velocity, elev_angle, nyquist_velocity,
                          debug=False, inherit_flag=None, inherit_azi_start=None,
                          inherit_azi_end=None):
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
    gamma = 0.5
    delta_vmax = gamma * nyquist_velocity  # The authorised change in velocity from one gate to the other.

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
    quadrant = find_reference.get_quadrant(azimuth, azi_start_pos, azi_end_pos)
    
    dealias_vel, flag_vel = initialisation.initialize_unfolding(r, azimuth, azi_start_pos, azi_end_pos, velocity, flag_vel, vnyq=nyquist_velocity)
    
    vel = velocity.copy()
    vel[azi_start_pos, :] = dealias_vel[azi_start_pos, :]
    delta_vmax = 0.75*nyquist_velocity
    
    # Clockwise
    dealias_vel, flag_vel = initialisation.first_pass(azi_start_pos, vel, dealias_vel, flag_vel, nyquist_velocity, vshift, delta_vmax)
    
    # Counterclockwise, i.e. array flipped
    stazi = len(azimuth) - azi_start_pos - 1
    dealias_vel, flag_vel = initialisation.first_pass(stazi, np.flipud(vel), np.flipud(dealias_vel), np.flipud(flag_vel), nyquist_velocity, vshift, delta_vmax)    
    dealias_vel = np.flipud(dealias_vel)
    flag_vel = np.flipud(flag_vel)
        
    # Range
    dealias_vel, flag_vel = continuity.correct_range_onward(velocity, dealias_vel, flag_vel, nyquist_velocity)
    dealias_vel, flag_vel = continuity.correct_range_backward(velocity, dealias_vel, flag_vel, nyquist_velocity)
    
    if count_proc(flag_vel, False) < 100:
        azimuth_iteration = np.arange(azi_start_pos, azi_start_pos + len(azimuth))
        azimuth_iteration[azimuth_iteration >= len(azimuth)] -= len(azimuth)
        dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity, dealias_vel, flag_vel, 
                                                             azimuth_iteration, nyquist_velocity, 6)
        dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity, dealias_vel, flag_vel, 
                                                                    azimuth_iteration, nyquist_velocity, 6)

    if count_proc(flag_vel, False) < 100:
        dealias_vel, flag_vel = continuity.correct_range_onward_loose(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_range_backward_loose(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity)

    # Box error check with respect to surrounding velocities
    dealias_vel, flag_vel = continuity.correct_box(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity, 5, 2)
    dealias_vel, flag_vel = continuity.correct_box(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity, 20, 10)
    dealias_vel, flag_vel = continuity.correct_box(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity, 40, 20)
    dealias_vel, flag_vel = continuity.correct_box(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity, 80, 40)

    # Dealiasing with a circular area of points around the unprocessed ones (no point left unprocessed after this step).
    if elev_angle <= 6:
        # Least squares error check in the radial direction
        dealias_vel, flag_vel = continuity.radial_least_square_check(r, azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity)
        # No flag.
        dealias_vel = continuity.least_square_radial_last_module(r, azimuth, dealias_vel, nyquist_velocity)
        
    # Looking for the closest reference..
    if count_proc(flag_vel, False) < 100:
        dealias_vel, flag_vel = continuity.correct_closest_reference(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity)

    dealias_vel, flag_vel = continuity.box_check(azimuth, dealias_vel, flag_vel, nyquist_velocity)

    if debug:
        print(f"2D fields processed in {time.time() - st_time:0.2f} seconds for {elev_angle:0.2f} elevation.")

    try:
        dealias_vel = dealias_vel.filled(np.NaN)
    except Exception:
        pass

    return dealias_vel, flag_vel, azi_start_pos, azi_end_pos


def process_3D(radar, velname="VEL", dbzname="DBZ", gatefilter=None, nyquist_velocity=None,
               debug=False, do_3D=True):
    """
    Process driver.
    Full dealiasing process 2D + 3D.

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
    zdr_name: str
        Name of the differential reflectivity field.
    rho_name: str
        Name of the cross correlation ratio field.

    Returns:
    ========
    ultimate_dealiased_velocity: ndarray
        Dealised velocit
        y field.
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
    final_vel, flag_vel, azi_s, azi_e = dealiasing_process_2D(r, azimuth_reference, velocity_reference,
                                                              elevation_reference, nyquist_velocity, debug=False)

    velocity_reference = final_vel.copy()
    flag_reference = flag_vel.copy()

    ultimate_dealiased_velocity = np.zeros(radar.fields[velname]['data'].shape)
    ultimate_dealiased_velocity[myslice] = final_vel.copy()

    for slice_number in range(1, radar.nsweeps):
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

        final_vel, flag_vel, azi_s, azi_e = dealiasing_process_2D(r, azimuth_slice, velocity_slice,
                                                                  elevation_slice, nyquist_velocity,
                                                                  debug=debug, inherit_flag=flag_slice,
                                                                  inherit_azi_start=azi_s, inherit_azi_end=azi_e)

        if do_3D:
            velocity_slice, flag_slice = continuity.unfolding_3D(r, elevation_reference,
                                                                 azimuth_reference,
                                                                 elevation_slice,
                                                                 azimuth_slice,
                                                                 velocity_reference,
                                                                 flag_reference,
                                                                 final_vel, flag_vel,
                                                                 nyquist_velocity)

            azimuth_reference = azimuth_slice.copy()
            velocity_reference = final_vel.copy()
            flag_reference = flag_vel.copy()
            elevation_reference = elevation_slice

        ultimate_dealiased_velocity[myslice] = final_vel.copy()

    ultimate_dealiased_velocity = np.ma.masked_where(gatefilter.gate_excluded,
                                                     ultimate_dealiased_velocity)

    return ultimate_dealiased_velocity
