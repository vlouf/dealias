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
@date: 05/04/2018

    count_proc
    dealiasing_process_2D
    process_3D
"""
# Python Standard Library
import time

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


def dealiasing_process_2D(r, azimuth, velocity, elev_angle, nyquist_velocity, debug=False, inherit_flag=None, inherit_azi_start=None, inherit_azi_end=None):
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
        Flag array (-3: No data, 0: Unprocessed, 1: Processed - no change -, 2: Processed - dealiased.)
    """
    st_time = time.time()  # tic
    try:
        velocity_nomask = velocity.filled(np.NaN)
    except Exception:
        velocity_nomask = velocity.copy()

    tot_gate = velocity_nomask.shape[0] * velocity_nomask.shape[1]
    nmask_gate = np.sum(np.isnan(velocity_nomask))
    print(f"There are {tot_gate - nmask_gate} gates to dealias at elevation {elev_angle}.")

    # Dealiasing based upon previously corrected velocities starting from two reference
    # radials, approximately 180° apart, where the wind is nearly orthogonal to the radar beam..
    try:
        azi_start_pos, azi_end_pos = find_reference.find_reference_radials(azimuth, velocity, debug)
    except ValueError:
        if inherit_azi_start is not None:
            azi_start_pos = inherit_azi_start
            azi_end_pos = inherit_azi_end
        else:
            azi_start_pos = 0
            azi_end_pos = len(azimuth) // 2

    # Looking for midpoints between the two reference radials. (4 quadrants to iter to).
    quadrant = find_reference.get_quadrant(azimuth, azi_start_pos, azi_end_pos)
    # Initialize unfolding, verifying reference radials.
    try:
        dealias_vel, flag_vel = initialisation.initialize_unfolding(r, azimuth, azi_start_pos,
                                                                    azi_end_pos, velocity_nomask, nyquist_velocity)
    except Exception:
        flag_vel = inherit_flag
        if inherit_flag is None:
            raise ValueError("No possible starting point found. Cannot dealias.")

        dealias_vel = velocity_nomask.copy()
        pos = flag_vel != 1
        flag_vel[pos] = 0
        dealias_vel[pos] = 0
        dealias_vel[np.isnan(dealias_vel)] = 0

    dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel, flag_vel,
                                                         quadrant[0], nyquist_velocity)
    dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                                flag_vel, quadrant[1], nyquist_velocity)
    dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel, flag_vel,
                                                         quadrant[2], nyquist_velocity)
    dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                                flag_vel, quadrant[3], nyquist_velocity)

    dealias_vel, flag_vel = continuity.correct_range_onward(velocity, dealias_vel, flag_vel, nyquist_velocity)
    dealias_vel, flag_vel = continuity.correct_range_backward(velocity, dealias_vel, flag_vel, nyquist_velocity)

    # One full sweep.
    dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                         flag_vel, np.arange(0, dealias_vel.shape[0]), nyquist_velocity)

    dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                                flag_vel, np.arange(dealias_vel.shape[0] - 1, 0, -1), nyquist_velocity)

    dealias_vel, flag_vel = continuity.correct_range_onward(velocity, dealias_vel, flag_vel, nyquist_velocity)
    dealias_vel, flag_vel = continuity.correct_range_backward(velocity, dealias_vel, flag_vel, nyquist_velocity)

    # Loose radial area dealiasing.
    dealias_vel, flag_vel = continuity.correct_range_onward_loose(azimuth, velocity_nomask,
                                                                  dealias_vel, flag_vel, nyquist_velocity)

    # Looking for the closest reference..
    dealias_vel, flag_vel = continuity.correct_closest_reference(azimuth, velocity_nomask,
                                                                 dealias_vel, flag_vel, nyquist_velocity)

    # Box error check with respect to surrounding velocities
    dealias_vel, flag_vel = continuity.correct_box(azimuth, velocity_nomask, dealias_vel,
                                                   flag_vel, nyquist_velocity)

    # Dealiasing with a circular area of points around the unprocessed ones (no point left unprocessed after this step).
    if elev_angle <= 6:
        # Least squares error check in the radial direction
        dealias_vel, flag_vel = continuity.radial_least_square_check(r, azimuth, velocity_nomask,
                                                                     dealias_vel, flag_vel,
                                                                     nyquist_velocity)

    # No flag.
    dealias_vel = continuity.least_square_radial_last_module(r, azimuth, dealias_vel,
                                                             nyquist_velocity)

    dealias_vel, flag_vel = continuity.box_check(azimuth, dealias_vel, flag_vel, nyquist_velocity)

    if debug:
        print(f"2D fields processed in {time.time() - st_time:0.2f} seconds for {elev_angle:0.2f} elevation.")

    try:
        dealias_vel = dealias_vel.filled(np.NaN)
    except Exception:
        pass

    return dealias_vel, flag_vel, azi_start_pos, azi_end_pos


def process_3D(radar, velname="VEL", dbzname="DBZ", zdrname="ZDR", rhohvname="RHOHV",
               gatefilter=None, nyquist_velocity=None, two_passes=False, debug=False):
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
        Dealised velocity field.
    """
    # Filter
    if gatefilter is None:
        gatefilter = filtering.do_gatefilter(radar, "VEL", "DBZ", zdr_name="ZDR", rho_name="RHOHV")

    if nyquist_velocity is None:
        nyquist_velocity = radar.instrument_parameters['nyquist_velocity']['data'][0]

    # Start with first reference.
    slice_number = 0
    myslice = radar.get_slice(slice_number)

    r = radar.range['data'].copy()
    velocity = radar.fields["VEL"]['data'].copy()
    azimuth_reference = radar.azimuth['data'][myslice]
    elevation_reference = radar.elevation['data'][myslice].mean()

    velocity_reference = np.ma.masked_where(gatefilter.gate_excluded, velocity)[myslice]

    # Dealiasing first sweep.
    final_vel, flag_vel, azi_s, azi_e = dealiasing_process_2D(r, azimuth_reference, velocity_reference,
                                                              elevation_reference, nyquist_velocity, debug=False)

    velocity_reference = final_vel.copy()
    flag_reference = flag_vel.copy()

    ultimate_dealiased_velocity = np.zeros(radar.fields["VEL"]['data'].shape)
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
                                                                  debug=False, inherit_flag=flag_slice,
                                                                  inherit_azi_start=azi_s, inherit_azi_end=azi_e)

        velocity_slice, flag_slice = continuity.unfolding_3D(r, elevation_reference,
                                                             azimuth_reference,
                                                             elevation_slice,
                                                             azimuth_slice,
                                                             velocity_reference,
                                                             flag_reference,
                                                             final_vel, flag_vel,
                                                             nyquist_velocity, loose=False)

        azimuth_reference = azimuth_slice.copy()
        velocity_reference = final_vel.copy()
        flag_reference = flag_vel.copy()
        elevation_reference = elevation_slice

        ultimate_dealiased_velocity[myslice] = final_vel.copy()

    ultimate_dealiased_velocity = np.ma.masked_where(gatefilter.gate_excluded,
                                                     ultimate_dealiased_velocity)

    return ultimate_dealiased_velocity
