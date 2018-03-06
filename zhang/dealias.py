"""
Main module.

@title: dealias
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 19/02/2018

    _multi_proc_buffer
    count_proc
    dealiasing_process_2D
    dealiasing_main_process
"""
# Python Standard Library
import time
# from multiprocessing import Pool

# Other python libraries.
# import pyart
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


def dealiasing_process_2D(r, azimuth, velocity, elev_angle, nyquist_velocity, debug=False):
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

    # Dealiasing based upon previously corrected velocities starting from two reference
    # radials, approximately 180° apart, where the wind is nearly orthogonal to the radar beam..
    azi_start_pos, azi_end_pos = find_reference.find_reference_radials(azimuth, velocity, debug)
    # Looking for midpoints between the two reference radials. (4 quadrants to iter to).
    quadrant = find_reference.get_quadrant(azimuth, azi_start_pos, azi_end_pos)
    # Initialize unfolding, verifying reference radials.
    dealias_vel, flag_vel = initialisation.initialize_unfolding(r, azimuth, azi_start_pos,
                                                                azi_end_pos, velocity_nomask, nyquist_velocity)

    # This is very strict continuity, we want to make the most of it.
    pass_nb = 0
    while pass_nb < 2:
        pass_nb += 1
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

        # Radial dealiasing inside the quadrants, starting from midpoints. With less strict reference findings.
        dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                                    flag_vel, quadrant[0][::-1], nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel, flag_vel,
                                                             quadrant[1][::-1], nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_counterclockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                                    flag_vel, quadrant[2][::-1], nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel, flag_vel,
                                                             quadrant[3][::-1], nyquist_velocity)

        dealias_vel, flag_vel = continuity.correct_range_onward(velocity, dealias_vel, flag_vel, nyquist_velocity)
        dealias_vel, flag_vel = continuity.correct_range_backward(velocity, dealias_vel, flag_vel, nyquist_velocity)

    # One full sweep.
    dealias_vel, flag_vel = continuity.correct_clockwise(r, azimuth, velocity_nomask, dealias_vel,
                                                         flag_vel, np.arange(dealias_vel.shape[0]), nyquist_velocity)
    # Loose radial area dealiasing.
    dealias_vel, flag_vel = continuity.correct_range_onward_loose(azimuth, velocity_nomask, dealias_vel, flag_vel, nyquist_velocity)

    # Looking for the closest reference..
    dealias_vel, flag_vel = continuity.correct_closest_reference(azimuth, velocity_nomask, dealias_vel, flag_vel, nyquist_velocity)

    # Box error check with respect to surrounding velocities
    dealias_vel, flag_vel = continuity.correct_box(azimuth, velocity_nomask, dealias_vel, flag_vel, nyquist_velocity)

    # Dealiasing with a circular area of points around the unprocessed ones (no point left unprocessed after this step).
    # Dealias_vel, flag_vel = continuity.radial_continuity_roi(azimuth, velocity, dealias_vel, flag_vel, nyquist_velocity)
    if elev_angle <= 6:
        # Least squares error check in the radial direction
        dealias_vel, flag_vel = continuity.radial_least_square_check(r, azimuth, velocity_nomask, dealias_vel, flag_vel, nyquist_velocity)

    # No flag.
    dealias_vel = continuity.least_square_radial_last_module(r, azimuth, dealias_vel, nyquist_velocity)

    dealias_vel, flag_vel = continuity.box_check(azimuth, dealias_vel, flag_vel, nyquist_velocity)

    if debug:
        print(f"2D fields processed in {time.time() - st_time:0.2f} seconds for {elev_angle:0.2f} elevation.")

    try:
        dealias_vel = dealias_vel.filled(np.NaN)
    except Exception:
        pass

    return dealias_vel, flag_vel


def process_3D(radar, velname="VEL", dbzname="DBZ", zdrname="ZDR", rhohvname="RHOHV",
               gatefilter=None, nyquist_velocity=None, two_passes=False, debug=False):
    """
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
    final_vel, flag_vel = dealiasing_process_2D(r, azimuth_reference, velocity_reference,
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

        # 3D dealiasing
        velocity_slice, flag_slice = continuity.unfolding_3D(r, elevation_reference, azimuth_reference,
                                                             elevation_slice, azimuth_slice, velocity_reference,
                                                             flag_reference, velocity_slice, flag_slice,
                                                             nyquist_velocity, loose=True)

        final_vel, flag_vel = dealiasing_process_2D(r, azimuth_slice, velocity_slice, elevation_slice,
                                                    nyquist_velocity, debug=False)

        if two_passes:
            velocity_slice, flag_slice = continuity.unfolding_3D(r, elevation_reference, azimuth_reference,
                                                                 elevation_slice, azimuth_slice, velocity_reference,
                                                                 flag_reference, final_vel, flag_vel,
                                                                 nyquist_velocity, loose=False)
        azimuth_reference = azimuth_slice.copy()
        velocity_reference = final_vel.copy()
        flag_reference = flag_vel.copy()

        ultimate_dealiased_velocity[myslice] = final_vel.copy()
    #     plot_radar(final_vel, flag_vel, slice_number)

    ultimate_dealiased_velocity = np.ma.masked_where(gatefilter.gate_excluded, ultimate_dealiased_velocity)

    return ultimate_dealiased_velocity


# def correct_azimuth(radar):
#     """
#     Sometimes, rapic files have multiple zeros. This corrects from wrong azimuths.
#     """
#     for sl in radar.iter_slice():
#         azi = radar.azimuth['data'][sl]
#         if azi[0] != 0:
#             continue
#         n0 = np.sum(azi == 0)
#         if n0 > 1:
#             try:
#                 for cnt in range(0, n0 + 2):
#                     if azi[cnt] != 0:
#                         break
#                 for cnt2 in range(0, cnt):
#                     val = azi[cnt] - (cnt - cnt2)
#                     if val < 0:
#                         val += 360
#                     azi[cnt2] = val
#             except Exception:
#                 continue
#     return None


# def dealiasing_main_process(input_file, gatefilter=None, debug=False, ncpus=16, vel_name="VEL",
#                             dbz_name="DBZ", zdr_name="ZDR", rho_name="RHOHV"):
#     """
#     Full dealiasing process 2D + 3D.
#
#     Parameters:
#     ===========
#     input_file: str
#         Input radar file to dealias. File must be compatible with Py-ART.
#     gatefilter: Object GateFilter
#         GateFilter for filtering noise. If not provided it will automaticaly
#         compute it with help of the dual-polar variables.
#     debug: boolean
#         Print debug messages.
#     ncpus: int
#         Number of process to use.
#     vel_name: str
#         Name of the velocity field.
#     dbz_name: str
#         Name of the reflectivity field.
#     zdr_name: str
#         Name of the differential reflectivity field.
#     rho_name: str
#         Name of the cross correlation ratio field.
#
#     Returns:
#     ========
#     ultimate_dealiased_velocity: ndarray
#         Dealised velocity field.
#     """
#     try:
#         radar = pyart.io.read(input_file)
#     except Exception:
#         raise
#
#     correct_azimuth(radar)
#
#     try:
#         radar.fields[vel_name]
#     except KeyError:
#         raise KeyError(f"Wrong name for the velocity field. {vel_name} not found.")
#
#     try:
#         nyquist_velocity = radar.instrument_parameters['nyquist_velocity']['data'][0]
#     except Exception:
#         nyquist_velocity = np.max(np.abs(radar.fields[vel_name]['data']))
#
#     ultimate_dealiased_velocity = np.zeros_like(radar.fields[vel_name]['data'])
#     if gatefilter is None:
#         gatefilter = filtering.do_gatefilter(radar, vel_name, dbz_name, zdr_name, rho_name)
#
#     st_time = time.time()
#
#     args_list = [(radar, gatefilter, nyquist_velocity, vel_name, debug, sln) for sln in range(radar.nsweeps)]
#     # In case we are not allowed to spawn deamonic processes
#     try:
#         with Pool(ncpus) as pool:
#             rslt = pool.starmap(_multi_proc_buffer, args_list)
#     except AssertionError:
#         rslt = [None] * len(args_list)
#         for cnt, myargs in enumerate(args_list):
#             rslt[cnt] = _multi_proc_buffer(*myargs)
#
#     if True:
#         print(f"2D fields processed in {time.time() - st_time:0.2f} seconds.")
#         print("Starting 3D processing.")
#
#     # Dealiasing 3D
#     myslice = radar.get_slice(0)
#     r = radar.range['data']
#     azimuth_reference = radar.azimuth['data'][myslice]
#     elevation_reference = radar.elevation['data'][myslice].mean()
#     velocity_reference, flag_reference, _ = rslt[0]
#
#     ultimate_dealiased_velocity[myslice] = velocity_reference.copy()
#
#     for (velocity_slice, flag_slice, slice_number) in rslt:
#         myslice = radar.get_slice(slice_number)
#         azimuth_slice = radar.azimuth['data'][myslice]
#         elevation_slice = radar.elevation['data'][myslice].mean()
#
#         # 3D dealiasing
#         velocity_slice, flag_slice = continuity.unfolding_3D(r, elevation_reference, azimuth_reference,
#                                                              elevation_slice, azimuth_slice, velocity_reference,
#                                                              flag_reference, velocity_slice, flag_slice, nyquist_velocity)
#
# #         velocity_slice = continuity.least_square_radial_last_module(r, azimuth_slice, velocity_slice, nyquist_velocity)
#         velocity_slice, flag_slice = continuity.box_check(azimuth_slice, velocity_slice, flag_slice, nyquist_velocity)
#
#         azimuth_reference = azimuth_slice.copy()
#         velocity_reference = velocity_slice.copy()
#         flag_reference = flag_slice.copy()
#
#         ultimate_dealiased_velocity[myslice] = velocity_reference.copy()
#
#     ultimate_dealiased_velocity = np.ma.masked_where(radar.fields[vel_name]['data'].mask, ultimate_dealiased_velocity)
#
#     return ultimate_dealiased_velocity
