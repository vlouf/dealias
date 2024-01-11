"""
Driver script for the dealiasing module.

@title: dealias
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 05/04/2018
@date: 11/01/2024

    _check_nyquist
    unmask_array
    dealiasing_process_2D
    dealias_long_range
    unravel_3D_pyart_multiproc
    unravel_3D_pyart
    unravel_3D_pyodim
"""
from typing import Union, Tuple

import pyart
import numpy as np
import dask.bag as db

from . import continuity
from . import filtering
from .core import Dealias


def _check_nyquist(radar: pyart.core.Radar, nyquist_velocity: float) -> np.ndarray:
    """
    If nyquist is not defined, then it will assume that it is the same
    nyquist for the whole sweep. If you want a different nyquist at each
    sweep then pass a list.

    Parameters:
    ===========
    radar: pyart.core.Radar
        Radar object
    nyquist_velocity: List or Scalar
        Nyquist velocity associated with each sweep.

    Returns:
    ========
    nyquist_list: List[float: nsweep]
        List of Nyquist velocity for each radar sweep.
    """
    if nyquist_velocity is None:
        nyquist_velocity = radar.instrument_parameters["nyquist_velocity"]["data"][0]
        nyquist_list = [nyquist_velocity] * radar.nsweeps
        if nyquist_velocity is None:
            raise ValueError("Nyquist velocity not found.")
    else:
        if np.isscalar(nyquist_velocity):
            nyquist_list = [nyquist_velocity] * radar.nsweeps
            pass
        else:
            if len(nyquist_velocity) != radar.nsweeps:
                raise IndexError("Nyquist velocity list size is different from the number of radar sweeps.")
            else:
                nyquist_list = nyquist_velocity

    return np.array(nyquist_list)


def unmask_array(x: Union[np.ndarray, np.ma.MaskedArray], fill_value=np.NaN) -> np.ndarray:
    try:
        x = x.filled(fill_value)
    except AttributeError:
        pass
    return x


def dealiasing_process_2D(
    r: np.ndarray,
    azimuth: np.ndarray,
    elevation: float,
    velocity: np.ndarray,
    nyquist_velocity: float,
    alpha: float = 0.6,
    debug: bool = False,
):
    """
    Dealiasing processing for 2D slice of the Doppler radar velocity field.

    Parameters:
    ===========
    r: ndarray
        Radar scan range.
    azimuth: ndarray
        Radar scan azimuth.
    elevation: float
        Elevation angle of the velocity field.
    velocity: ndarray <azimuth, r>
        Aliased Doppler velocity field.
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
    brake = None
    if not np.isscalar(elevation):
        raise TypeError("Elevation should be scalar, not an array.")
    if velocity.shape != (len(azimuth), len(r)):
        raise ValueError("The dimensions of the velocity field should be <azimuth, range>.")
    r = unmask_array(r)
    azimuth = unmask_array(azimuth)
    velocity = unmask_array(velocity)

    dealias_2D = Dealias(r, azimuth, elevation, velocity, nyquist_velocity, alpha)

    # Initialization
    dealias_2D.initialize()

    # Dealiasing modules
    dealias_2D.correct_range()
    for window in [6, 12]:
        dealias_2D.correct_range(window)
        dealias_2D.correct_clock(window)
        if dealias_2D.check_completed():
            brake = "range"
            break

    if not dealias_2D.check_completed():
        for window in [(5, 2), (20, 10), (40, 20)]:
            dealias_2D.correct_box(window)
            if dealias_2D.check_completed():
                brake = "box"
                break

    if not dealias_2D.check_completed():
        brake = "square"
        dealias_2D.correct_leastsquare()

    if not dealias_2D.check_completed():
        brake = "regression"
        dealias_2D.correct_linregress()

    if not dealias_2D.check_completed():
        brake = "closest"
        dealias_2D.correct_closest()

    # Checking modules.
    dealias_2D.check_leastsquare()
    dealias_2D.check_box()

    unfold_vel = dealias_2D.dealias_vel.copy()
    unfold_vel[dealias_2D.flag < 0] = np.NaN

    if debug:
        return unfold_vel, dealias_2D.flag, brake

    return unfold_vel, dealias_2D.flag


def dealias_long_range(
    r: np.ndarray,
    azimuth: np.ndarray,
    elevation: float,
    velocity: np.ndarray,
    nyquist_velocity: float,
    alpha: float = 0.6,
    debug: bool = False,
):
    """
    Dealiasing processing for 2D slice of the Doppler radar velocity field.

    Parameters:
    ===========
    r: ndarray
        Radar scan range.
    azimuth: ndarray
        Radar scan azimuth.
    elevation: float
        Elevation angle of the velocity field.
    velocity: ndarray <azimuth, r>
        Aliased Doppler velocity field.
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
    brake = None
    if not np.isscalar(elevation):
        raise TypeError("Elevation should be scalar, not an array.")
    if velocity.shape != (len(azimuth), len(r)):
        raise ValueError("The dimensions of the velocity field should be <azimuth, range>.")
    r = unmask_array(r)
    azimuth = unmask_array(azimuth)
    velocity = unmask_array(velocity)

    dealias_2D = Dealias(r, azimuth, elevation, velocity, nyquist_velocity, alpha)

    dealias_2D.initialize()
    dealias_2D.correct_range()
    for window in [6, 12, 24, 48, 96]:
        dealias_2D.correct_range(window)
        dealias_2D.correct_clock(window)
        if dealias_2D.check_completed():
            brake = "range"
            break

    if not dealias_2D.check_completed():
        for window in [(20, 20), (40, 40)]:
            dealias_2D.correct_box(window)
            if dealias_2D.check_completed():
                brake = "box"
                break

    if not dealias_2D.check_completed():
        brake = "regression"
        dealias_2D.correct_linregress()

    if not dealias_2D.check_completed():
        brake = "closest"
        dealias_2D.correct_closest()

    dealias_2D.check_box()

    unfold_vel = dealias_2D.dealias_vel.copy()
    unfold_vel[dealias_2D.flag < 0] = np.NaN

    if debug:
        return unfold_vel, dealias_2D.flag, brake

    return unfold_vel, dealias_2D.flag


def unravel_3D_pyart_multiproc(
    radar: pyart.core.Radar,
    velname: str = "VEL",
    dbzname: str = "DBZ",
    gatefilter: pyart.filters.GateFilter = None,
    nyquist_velocity: float = None,
    strategy: str = "default",
    alpha: float = 0.8,
    do_3d: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Process driver.
    Full dealiasing process 2D + 3D.

    Parameters:
    ===========
    radar: PyART Radar Object
        Py-ART radar object.
    gatefilter: Object GateFilter
        GateFilter for filtering noise. If not provided it will automaticaly
        compute it with help of the dual-polar variables.
    velname: str
        Name of the velocity field.
    dbzname: str
        Name of the reflectivity field.
    strategy: ['default', 'long_range']
        Using the default dealiasing strategy or the long range strategy.
    nyquist_velocity: float or list
        If it is a scalar, then it is assume as being the same nyquist for the
        whole volume. If it is a list, it is expected to be the value of nyquist
        for each sweep.

    Returns:
    ========
    unraveled_velocity: ndarray
        Dealised velocity field.
    """
    # Check arguments
    if strategy not in ["default", "long_range"]:
        raise ValueError("Dealiasing strategy not understood please choose 'default' or 'long_range'")
    if gatefilter is None:
        gatefilter = filtering.do_gatefilter(radar, dbzname)
    unraveled_velocity = np.zeros(radar.fields[velname]["data"].shape)
    nyquist_list = _check_nyquist(radar, nyquist_velocity)

    # Read the velocity field.
    velocity = unmask_array(radar.fields[velname]["data"])
    velocity[gatefilter.gate_excluded] = np.NaN

    # Build argument list for multiprocessing.
    args_list = []
    r = unmask_array(radar.range["data"])
    for slice_number in range(0, radar.nsweeps):
        nyquist_velocity = nyquist_list[slice_number]
        sweep = radar.get_slice(slice_number)
        azi = radar.azimuth["data"][sweep]
        elev = radar.elevation["data"][sweep].mean()
        velocity_slice = velocity[sweep]
        args_list.append((r, azi, elev, velocity_slice, nyquist_velocity, alpha))

    # Run the 2D dealiasing using multiprocessing 1 process per sweep.
    if strategy == "default":
        bag = db.from_sequence(args_list).starmap(dealiasing_process_2D)
    else:
        bag = db.from_sequence(args_list).starmap(dealias_long_range)
    rslt = bag.compute()

    # Run the 3D Unfolding using the first slice as reference.
    if do_3d:
        args_list = []
        sweep = radar.get_slice(0)
        azimuth_reference = unmask_array(radar.azimuth["data"][sweep])
        elevation_reference = unmask_array(radar.elevation["data"][sweep].mean())
        velocity_reference, flag_reference = rslt[0][0], rslt[0][1]
        unraveled_velocity[radar.get_slice(0)] = velocity_reference.copy()
        for slice_number in range(1, radar.nsweeps):
            nyquist_velocity = nyquist_list[slice_number]
            sweep = radar.get_slice(slice_number)
            azimuth_slice = unmask_array(radar.azimuth["data"][sweep])
            elevation_slice = unmask_array(radar.elevation["data"][sweep].mean())
            final_vel, flag_vel = rslt[slice_number][0], rslt[slice_number][1]

            final_vel, flag_slice, _, _ = continuity.unfolding_3D(
                r_swref=r,
                azi_swref=azimuth_reference,
                elev_swref=elevation_reference,
                vel_swref=velocity_reference,
                flag_swref=flag_reference,
                r_slice=r,
                azi_slice=azimuth_slice,
                elev_slice=elevation_slice,
                velocity_slice=final_vel,
                flag_slice=flag_vel,
                original_velocity=velocity[sweep],
                vnyq=nyquist_velocity,
                window_azi=6,
                window_range=10,
                alpha=alpha,
            )

            args_list.append((azimuth_slice, final_vel, flag_slice, nyquist_velocity, 20))
            azimuth_reference = azimuth_slice.copy()
            velocity_reference = final_vel.copy()
            flag_reference = flag_vel.copy()
            elevation_reference = elevation_slice

        # Multiproc box check and saved unravel velocity
        bag = db.from_sequence(args_list).starmap(continuity.box_check)
        nrslt = bag.compute()

        for n in range(len(nrslt)):
            sweep = radar.get_slice(n + 1)
            unraveled_velocity[sweep] = nrslt[n][0]
    else:
        for n in range(len(rslt)):
            sweep = radar.get_slice(n)
            unraveled_velocity[sweep] = rslt[n][0]

    return unraveled_velocity


def unravel_3D_pyart(
    radar: pyart.core.Radar,
    velname: str = "VEL",
    dbzname: str = "DBZ",
    gatefilter: pyart.filters.GateFilter = None,
    nyquist_velocity: float = None,
    strategy: str = "default",
    alpha: float = 0.8,
    do_3d: bool = True,
    debug: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Process driver.
    Full dealiasing process 2D + 3D.

    Parameters:
    ===========
    radar: PyART Radar Object
        Py-ART radar object.
    gatefilter: Object GateFilter
        GateFilter for filtering noise. If not provided it will automaticaly
        compute it with help of the dual-polar variables.
    velname: str
        Name of the velocity field.
    dbzname: str
        Name of the reflectivity field.
    strategy: ['default', 'long_range']
        Using the default dealiasing strategy or the long range strategy.
    nyquist_velocity: float or list
        If it is a scalar, then it is assume as being the same nyquist for the
        whole volume. If it is a list, it is expected to be the value of nyquist
        for each sweep.

    Returns:
    ========
    unraveled_velocity: ndarray
        Dealised velocity field.
    """
    pointbreak = []
    # Check arguments
    if strategy not in ["default", "long_range"]:
        raise ValueError("Dealiasing strategy not understood please choose 'default' or 'long_range'")
    if gatefilter is None:
        gatefilter = filtering.do_gatefilter(radar, dbzname)
    nyquist_list = _check_nyquist(radar, nyquist_velocity)

    # Read the velocity field.
    velocity = unmask_array(radar.fields[velname]["data"])
    velocity[gatefilter.gate_excluded] = np.NaN

    # Read coordinates and start with the first sweep.
    sweep = radar.get_slice(0)
    r = unmask_array(radar.range["data"])
    azimuth_reference = unmask_array(radar.azimuth["data"][sweep])
    elevation_reference = unmask_array(radar.elevation["data"][sweep].mean())
    velocity_reference = velocity[sweep]

    # Dealiasing first sweep.
    nyquist_velocity = nyquist_list[0]
    if strategy == "default":
        outargs = dealiasing_process_2D(
            r, azimuth_reference, elevation_reference, velocity_reference, nyquist_velocity, debug=debug, **kwargs
        )
    else:
        outargs = dealias_long_range(
            r, azimuth_reference, elevation_reference, velocity_reference, nyquist_velocity, **kwargs
        )
    if debug:
        velocity_reference, flag_reference, brake = outargs
        pointbreak.append(brake)
    else:
        velocity_reference, flag_reference = outargs

    unraveled_velocity = np.zeros(radar.fields[velname]["data"].shape)
    unraveled_velocity[sweep] = velocity_reference.copy()

    for slice_number in range(1, radar.nsweeps):
        nyquist_velocity = nyquist_list[slice_number]
        sweep = radar.get_slice(slice_number)
        azimuth_slice = unmask_array(radar.azimuth["data"][sweep])
        elevation_slice = unmask_array(radar.elevation["data"][sweep].mean())
        velocity_slice = velocity[sweep]

        flag_slice = np.zeros_like(velocity_slice) + 1
        flag_slice[np.isnan(velocity_slice)] = -3

        if strategy == "default":
            outargs = dealiasing_process_2D(
                r, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, debug=debug, **kwargs
            )
        else:
            outargs = dealias_long_range(r, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, **kwargs)
        if debug:
            final_vel, flag_vel, brake = outargs
            pointbreak.append(brake)
        else:
            final_vel, flag_vel = outargs

        final_vel = unmask_array(final_vel)
        if do_3d:
            final_vel, flag_slice, _, _ = continuity.unfolding_3D(
                r,
                azimuth_reference,
                elevation_reference,
                velocity_reference,
                flag_reference,
                r,
                azimuth_slice,
                elevation_slice,
                final_vel,
                flag_vel,
                velocity[sweep],
                nyquist_velocity,
                alpha=alpha,
            )

            final_vel, flag_slice = continuity.box_check(azimuth_slice, final_vel, flag_slice, nyquist_velocity)
            azimuth_reference = azimuth_slice.copy()
            velocity_reference = final_vel.copy()
            flag_reference = flag_vel.copy()
            elevation_reference = elevation_slice

        unraveled_velocity[sweep] = final_vel.copy()

    if debug:
        return unraveled_velocity, pointbreak

    return unraveled_velocity


def unravel_3D_pyodim(
    odim_file,
    vel_name="VRADH",
    output_vel_name="unraveled_velocity",
    load_all_fields=False,
    gatefilter=None,
    strategy="long_range",
    debug=False,
    **kwargs,
):
    """
    Support for ODIM H5 files and Nyquist changing with the elevation. The new
    scan strategy is using single-PRF, to avoid dual-PRF artifacts, but with
    a PRF (and thus Nyquist velocity) that changes at each sweep.

    Parameters:
    ===========
    odim_file: str
        ODIM H5 file name.
    vel_name: str
        Velocity field name.
    output_vel_name: str
        Output dealiased velocity name
    load_all_fields: bool
        Load all fields in the ODIM H5 files (for writing the H5 file on disk)
        or just the velocity field.
    gatefilter: NoneType
        Placeholder for the GateFilter argument like in unravel_3D_pyart.
        Feature not supported yet.
    strategy: ['default', 'long_range']
        Using the default dealiasing strategy or the long range strategy.

    Returns:
    ========
    radar_datasets: List
        List of xarray datasets. PyODIM output data model.
    """
    # NOTE: This function is made to handle a variable PRF, and thus a variable
    # Nyquist. We use the sweeps with the highest Nyquist in the lowest
    # elevation scans as reference and we dealise in 3D, down and up from that
    # sweep.
    if strategy not in ["default", "long_range"]:
        raise ValueError("Dealiasing strategy not understood please choose 'default' or 'long_range'")
    if gatefilter is not None:
        raise ValueError("gatefilter not supported with pyodim structure. Please use Py-ART instead.")
    if debug:
        print("Argument debug=True is not yet supported with ODIM files.")

    import pyodim

    if load_all_fields:
        radar_datasets = pyodim.read_odim(odim_file)
    else:
        radar_datasets = pyodim.read_odim(odim_file, include_fields=[vel_name])
    radar_datasets = [r.compute() for r in radar_datasets]

    # Looking for low-elevation sweep with the highest Nyquist velocity to use
    # as reference.
    elev_angles = [r["elevation"].values[0] for r in radar_datasets]
    nyquists = [r.attrs["NI"] for r in radar_datasets]
    nslice_ref = np.argmax(nyquists[: len(elev_angles) // 2])

    r_reference = radar_datasets[nslice_ref].range.values
    azimuth_reference = radar_datasets[nslice_ref].azimuth.values
    velocity_reference = radar_datasets[nslice_ref][vel_name].values
    elevation_reference = radar_datasets[nslice_ref]["elevation"].values[0]
    nyquist_velocity = radar_datasets[nslice_ref].attrs["NI"]

    # Dealiasing first sweep.
    if strategy == "default":
        final_vel, flag_vel = dealiasing_process_2D(
            r_reference, azimuth_reference, elevation_reference, velocity_reference, nyquist_velocity, **kwargs
        )
    else:
        final_vel, flag_vel = dealias_long_range(
            r_reference, azimuth_reference, elevation_reference, velocity_reference, nyquist_velocity, **kwargs
        )
    velocity_reference, flag_reference = final_vel.copy(), flag_vel.copy()
    radar_datasets[nslice_ref] = radar_datasets[nslice_ref].merge(
        {output_vel_name: (("azimuth", "range"), velocity_reference)}
    )

    # Processing sweeps by decreasing elevations from the nslice_ref sweeps
    if nslice_ref != 0:
        for sweep in np.arange(nslice_ref)[::-1]:
            r_slice = radar_datasets[sweep].range.values
            azimuth_slice = radar_datasets[sweep].azimuth.values
            velocity_slice = radar_datasets[sweep][vel_name].values
            elevation_slice = radar_datasets[sweep]["elevation"].values[0]
            nyquist_velocity = radar_datasets[sweep].attrs["NI"]

            if strategy == "default":
                final_vel, flag_vel = dealiasing_process_2D(
                    r_slice, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, **kwargs
                )
            else:
                final_vel, flag_vel = dealias_long_range(
                    r_slice, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, **kwargs
                )

            final_vel, flag_slice, _, _ = continuity.unfolding_3D(
                r_reference,
                azimuth_reference,
                elevation_reference,
                velocity_reference,
                flag_reference,
                r_slice,
                azimuth_slice,
                elevation_slice,
                final_vel,
                flag_vel,
                velocity_slice,
                nyquist_velocity,
            )

            azimuth_reference = azimuth_slice.copy()
            velocity_reference = final_vel.copy()
            flag_reference = flag_vel.copy()
            elevation_reference = elevation_slice
            r_reference = r_slice

            radar_datasets[sweep] = radar_datasets[sweep].merge({output_vel_name: (("azimuth", "range"), final_vel)})

        r_reference = radar_datasets[nslice_ref].range.values
        azimuth_reference = radar_datasets[nslice_ref].azimuth.values
        velocity_reference = radar_datasets[nslice_ref][output_vel_name].values
        elevation_reference = radar_datasets[nslice_ref]["elevation"].values[0]

    # Processing sweeps by increasing elevations from the nslice_ref sweeps
    for sweep in range(nslice_ref + 1, len(radar_datasets)):
        r_slice = radar_datasets[sweep].range.values
        azimuth_slice = radar_datasets[sweep].azimuth.values
        velocity_slice = radar_datasets[sweep][vel_name].values
        elevation_slice = radar_datasets[sweep]["elevation"].values[0]
        nyquist_velocity = radar_datasets[sweep].attrs["NI"]

        if strategy == "default":
            final_vel, flag_vel = dealiasing_process_2D(
                r_slice, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, **kwargs
            )
        else:
            final_vel, flag_vel = dealias_long_range(
                r_slice, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, **kwargs
            )

        final_vel, flag_slice, _, _ = continuity.unfolding_3D(
            r_reference,
            azimuth_reference,
            elevation_reference,
            velocity_reference,
            flag_reference,
            r_slice,
            azimuth_slice,
            elevation_slice,
            final_vel,
            flag_vel,
            velocity_slice,
            nyquist_velocity,
        )

        azimuth_reference = azimuth_slice.copy()
        velocity_reference = final_vel.copy()
        flag_reference = flag_vel.copy()
        elevation_reference = elevation_slice
        r_reference = r_slice

        radar_datasets[sweep] = radar_datasets[sweep].merge({output_vel_name: (("azimuth", "range"), final_vel)})

    return radar_datasets
