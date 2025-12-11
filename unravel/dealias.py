"""
Driver script for the dealiasing module.

@title: dealias
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 05/04/2018
@date: 11/12/2025

    _check_nyquist
    unmask_array
    dealiasing_process_2D
    dealias_long_range
    unravel_3D_pyart_multiproc
    unravel_3D_pyart
    unravel_3D_pyodim
"""

from typing import Union, Tuple, List

import pyart
import numpy as np
import xarray as xr
import dask.bag as db
from numpy.typing import NDArray

from . import continuity
from . import filtering
from .cfg import log, stage_check
from .core import Dealias
from .odim import write_odim_slice


def _check_nyquist(radar: pyart.core.Radar, nyquist_velocity: Union[None, List[float], float]) -> NDArray:
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


def unmask_array(x: Union[np.ndarray, np.ma.MaskedArray], fill_value=np.nan) -> np.ndarray:
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
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, str]]:
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
    alpha: float
        Threshold for the dealiased velocity. Default is 0.6.
    debug: bool
        If True, returns additional information about the processing stages.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array (-3: No data, 0: Unprocessed, 1: Processed - no change -,
                    2: Processed - dealiased.)
    completed: str
        Name of the last completed stage in the dealiasing process.
        Only returned if debug is True.
    """
    if not np.isscalar(elevation):
        raise TypeError("Elevation should be scalar, not an array.")
    if velocity.shape != (len(azimuth), len(r)):
        raise ValueError("The dimensions of the velocity field should be <azimuth, range>.")
    r = unmask_array(r)
    azimuth = unmask_array(azimuth)
    velocity = unmask_array(velocity)

    dealias_2D = Dealias(r, azimuth, elevation, velocity, nyquist_velocity, alpha)

    # Initialisation
    # stages 0, 1, 2, 3
    dealias_2D.initialize()

    # Dealiasing modules
    completed = ""
    if stage_check("range"):
        dealias_2D.correct_range()
    for window in [6, 12]:
        if stage_check("range", completed):
            dealias_2D.correct_range(window)
        if stage_check("clock", completed):
            dealias_2D.correct_clock(window)
        if not completed and dealias_2D.check_completed():
            completed = "range"

    for window in [(5, 2), (20, 10), (40, 20)]:
        if stage_check("box", completed):
            dealias_2D.correct_box(window)
            if dealias_2D.check_completed():
                completed = "box"

    if stage_check("lsquare", completed):
        dealias_2D.correct_leastsquare()
        if dealias_2D.check_completed():
            completed = "lsquare"

    if stage_check("regression", completed):
        dealias_2D.correct_linregress()
        if dealias_2D.check_completed():
            completed = "regression"

    if stage_check("closest", completed):
        dealias_2D.correct_closest()
        if dealias_2D.check_completed():
            completed = "closest"

    # Checking modules.
    if stage_check("check-lsquare"):
        dealias_2D.check_leastsquare()
    if stage_check("check-box"):
        dealias_2D.check_box()

    unfold_vel = dealias_2D.dealias_vel.copy()
    unfold_vel[dealias_2D.flag < 0] = np.nan

    if debug:
        return unfold_vel, dealias_2D.flag, completed

    return unfold_vel, dealias_2D.flag


def dealias_long_range(
    r: np.ndarray,
    azimuth: np.ndarray,
    elevation: float,
    velocity: np.ndarray,
    nyquist_velocity: float,
    alpha: float = 0.6,
    debug: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, str]]:
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
    alpha: float
        Threshold for the dealiased velocity. Default is 0.6.
    debug: bool
        If True, returns additional information about the processing stages.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array (-3: No data, 0: Unprocessed, 1: Processed - no change -,
                    2: Processed - dealiased.)
    completed: str
        Name of the last completed stage in the dealiasing process.
        Only returned if debug is True.
    """
    if not np.isscalar(elevation):
        raise TypeError("Elevation should be scalar, not an array.")
    if velocity.shape != (len(azimuth), len(r)):
        raise ValueError("The dimensions of the velocity field should be <azimuth, range>.")
    r = unmask_array(r)
    azimuth = unmask_array(azimuth)
    velocity = unmask_array(velocity)

    dealias_2D = Dealias(r, azimuth, elevation, velocity, nyquist_velocity, alpha)

    # Initialisation
    # stages 0, 1, 2, 3
    dealias_2D.initialize()

    # Dealiasing modules
    completed = ""
    if stage_check("range"):
        dealias_2D.correct_range()
    for window in [6, 12, 24, 48, 96]:
        if stage_check("range", completed):
            dealias_2D.correct_range(window)
        if stage_check("clock", completed):
            dealias_2D.correct_clock(window)
        if not completed and dealias_2D.check_completed():
            completed = "range"

    for window in [(20, 20), (40, 40)]:
        if stage_check("box", completed):
            dealias_2D.correct_box(window)
            if dealias_2D.check_completed():
                completed = "box"

    if stage_check("regression", completed):
        dealias_2D.correct_linregress()
        if dealias_2D.check_completed():
            completed = "regression"

    if stage_check("closest", completed):
        dealias_2D.correct_closest()
        if not dealias_2D.check_completed():
            completed = "closest"

    # Checking modules
    if stage_check("check-box"):
        dealias_2D.check_box()

    unfold_vel = dealias_2D.dealias_vel.copy()
    unfold_vel[dealias_2D.flag < 0] = np.nan

    if debug:
        return unfold_vel, dealias_2D.flag, completed

    return unfold_vel, dealias_2D.flag


def unravel_3D_pyart_multiproc(
    radar: pyart.core.Radar,
    velname: str = "VEL",
    dbzname: str = "DBZ",
    gatefilter: Union[pyart.filters.GateFilter, None] = None,
    nyquist_velocity: Union[float, None] = None,
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
        Nyquist velocity.
    alpha: float
        Threshold for the dealiased velocity. Default is 0.8.
    do_3d: bool
        If True, run the 3D unfolding process after the 2D dealiasing.

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
    velocity[gatefilter.gate_excluded] = np.nan

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
    #
    # NB: parallel stage counting in stage_check() won't work due to global state
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
    gatefilter: Union[pyart.filters.GateFilter, None] = None,
    nyquist_velocity: Union[float, None] = None,
    strategy: str = "default",
    alpha: float = 0.8,
    do_3d: bool = True,
    debug: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
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
    alpha: float
        Threshold for the dealiased velocity. Default is 0.8.
    do_3d: bool
        If True, run the 3D unfolding process after the 2D dealiasing.
    debug: bool
        If True, returns additional information about the processing stages.

    Returns:
    ========
    unraveled_velocity: ndarray
        Dealised velocity field.
    pointbreak: List[str]
        List of pointbreaks if debug is True, otherwise not returned.
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
    velocity[gatefilter.gate_excluded] = np.nan

    # Read coordinates and start with the first sweep.
    sweep = radar.get_slice(0)
    r = unmask_array(radar.range["data"])
    azimuth_reference = unmask_array(radar.azimuth["data"][sweep])
    elevation_reference = unmask_array(radar.elevation["data"][sweep]).mean()
    velocity_reference = velocity[sweep]

    # Dealiasing first sweep.
    nyquist_velocity = nyquist_list[0]
    if nyquist_velocity is None:
        raise ValueError("Nyquist velocity must not be None for dealiasing_process_2D.")
    if strategy == "default":
        outargs = dealiasing_process_2D(
            r, azimuth_reference, elevation_reference, velocity_reference, nyquist_velocity, debug=debug, **kwargs
        )
    else:
        outargs = dealias_long_range(
            r, azimuth_reference, elevation_reference, velocity_reference, nyquist_velocity, **kwargs
        )
    if len(outargs) == 3:
        velocity_reference, flag_reference, brake = outargs
    else:
        velocity_reference, flag_reference = outargs
        brake = None
    pointbreak.append(brake)

    unraveled_velocity = np.zeros(radar.fields[velname]["data"].shape)
    unraveled_velocity[sweep] = velocity_reference.copy()
    for slice_number in range(1, radar.nsweeps):
        nyquist_velocity = nyquist_list[slice_number]
        if nyquist_velocity is None:
            raise ValueError("Nyquist velocity must not be None for dealiasing_process_2D.")
        sweep = radar.get_slice(slice_number)
        azimuth_slice = unmask_array(radar.azimuth["data"][sweep])
        elevation_slice = unmask_array(radar.elevation["data"][sweep]).mean()
        velocity_slice = velocity[sweep]

        flag_slice = np.zeros_like(velocity_slice) + 1
        flag_slice[np.isnan(velocity_slice)] = -3

        if strategy == "default":
            outargs = dealiasing_process_2D(
                r, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, debug=debug, **kwargs
            )
        else:
            outargs = dealias_long_range(r, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, **kwargs)
        if len(outargs) == 3:
            final_vel, flag_vel, brake = outargs
        else:
            final_vel, flag_vel = outargs
            brake = None
        pointbreak.append(brake)

        final_vel = unmask_array(final_vel)
        if stage_check("3d") and do_3d:
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

            if stage_check("check-box"):
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
    odim_input: Union[str, List[xr.Dataset]],
    vel_name: str = "VRADH",
    output_vel_name: str = "unraveled_velocity",
    load_all_fields: bool = False,
    condition: Union[None, Tuple[str, str, float]] = None,
    strategy: str = "long_range",
    alpha: float = 0.6,
    debug: bool = False,
    read_write: bool = False,
    output_flag_name: Union[str, None] = None,
) -> List[xr.Dataset]:
    """
    Support for ODIM H5 files and Nyquist changing with the elevation. The new
    scan strategy is using single-PRF, to avoid dual-PRF artifacts, but with
    a PRF (and thus Nyquist velocity) that changes at each sweep.

    Parameters:
    ===========
    odim_input: Union[str, List[xr.Dataset]]
        Either an ODIM H5 file path (str) or a list of pre-loaded xarray datasets.
        Passing pre-loaded datasets allows for preprocessing (e.g., dual-PRF correction)
        before dealiasing.
    vel_name: str
        Velocity field name.
    output_vel_name: str
        Output dealiased velocity name
    load_all_fields: bool
        Load all fields in the ODIM H5 files (for writing the H5 file on disk)
        or just the velocity field. Only used when odim_input is a file path.
    condition: (tuple)
        (variable_name, "lower"/"above", threshold_value)
    strategy: ['default', 'long_range']
        Using the default dealiasing strategy or the long range strategy.
    alpha: float
        Threshold for the dealiased velocity. Default is 0.6.
    debug: bool
        If True, returns additional information about the processing stages.
    read_write: bool
        Write back to original file if True. Only applies when odim_input is a file path.

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
    if debug:
        print("Argument debug=True is not yet supported with ODIM files.")
    if output_flag_name is None:
        output_flag_name = f"{output_vel_name}_flag"

    import pyodim

    # Handle both file path and pre-loaded datasets
    h5file = None
    if isinstance(odim_input, str):
        # Input is a file path - read it
        if load_all_fields or condition is not None:
            (rsets, h5file) = pyodim.read_write_odim(odim_input, read_write=read_write)
        else:
            (rsets, h5file) = pyodim.read_write_odim(odim_input, read_write=read_write, include_fields=[vel_name])
        rsets = [r.compute() for r in rsets]
    elif isinstance(odim_input, list):
        # Input is pre-loaded datasets
        rsets = odim_input
        if read_write:
            raise ValueError("read_write=True is only supported when odim_input is a file path")
    else:
        raise TypeError("odim_input must be either a file path (str) or a list of xarray Datasets")

    # Filtering data with provided gatefilter.
    radar_datasets = rsets
    if condition:
        var, op, threshold = condition
        for idx, radar in enumerate(rsets):
            mask = radar[var] < threshold if op == "lower" else radar[var] > threshold
            # Use .copy() to avoid modifying the original field
            radar_datasets[idx] = radar.merge(
                {f"{vel_name}_clean": (radar[vel_name].dims, np.ma.masked_where(mask, radar[vel_name].values.copy()))}
            )
        vel_name = f"{vel_name}_clean"

    # Looking for low-elevation sweep with the highest Nyquist velocity to use
    # as reference.
    elev_angles = [r["elevation"].values[0] for r in radar_datasets]
    nyquists = [r.attrs["NI"] for r in radar_datasets]
    nslice_ref = np.argmax(nyquists[: len(elev_angles) // 2])

    # Dealiasing first sweep.
    radar_datasets[nslice_ref] = unravel_3D_pyodim_slice(
        radar_datasets[nslice_ref], None, vel_name, strategy, output_vel_name, output_flag_name, alpha, debug
    )

    # Processing sweeps by decreasing elevations from the nslice_ref sweeps
    if nslice_ref != 0:
        for sweep in np.arange(nslice_ref)[::-1]:
            radar_datasets[sweep] = unravel_3D_pyodim_slice(
                radar_datasets[sweep],
                radar_datasets[sweep + 1],
                vel_name,
                strategy,
                output_vel_name,
                output_flag_name,
                alpha,
                debug,
            )

    # Processing sweeps by increasing elevations from the nslice_ref sweeps
    for sweep in range(nslice_ref + 1, len(radar_datasets)):
        radar_datasets[sweep] = unravel_3D_pyodim_slice(
            radar_datasets[sweep],
            radar_datasets[sweep - 1],
            vel_name,
            strategy,
            output_vel_name,
            output_flag_name,
            alpha,
            debug,
        )

    if read_write:
        if h5file is None:
            raise ValueError("Cannot write back to file when datasets were pre-loaded")
        for sweep in range(len(radar_datasets)):
            write_odim_slice(
                h5file,
                radar_datasets[sweep],
                vel_name,
                output_vel_name,
                output_flag_name,
            )

    return radar_datasets


def unravel_3D_pyodim_slice(
    ds_sweep: xr.Dataset,
    ds_ref: Union[None, xr.Dataset],
    vel_name: str,
    strategy: str,
    output_vel_name: str,
    output_flag_name: str,
    alpha: float,
    debug: bool,
) -> xr.Dataset:
    """
    Process one slice/sweep/tilt of ODIM polar radar volume.
    This function performs the dealiasing process on a single slice of the
    radar volume, using the provided strategy and reference dataset if available.
    Parameters:
    ===========
    ds_sweep: xr.Dataset
        Dataset for the current slice/sweep/tilt.
    ds_ref: xr.Dataset
        Reference sweep dataset for 3D unfolding, if available.
    vel_name: str
        Name of the velocity field in the dataset.
    strategy: str
        Dealiasing strategy to use, either 'default' or 'long_range'.
    output_vel_name: str
        Name for the output dealiased velocity field.
    output_flag_name: str
        Name for the output flag field indicating dealiasing status.
    alpha: float
        Threshold for the dealiased velocity. Default is 0.8.
    debug: bool
        If True, returns additional information about the processing stages.

    Returns:
    ========
    ds_sweep: xr.Dataset
        Updated dataset sweep with dealiased velocity and flag fields added.
    """

    r_slice = ds_sweep.range.values
    azimuth_slice = ds_sweep.azimuth.values
    velocity_slice = ds_sweep[vel_name].values.copy()
    elevation_slice = ds_sweep["elevation"].values[0]
    nyquist_velocity = ds_sweep.attrs["NI"]

    log(f"tilt:{elevation_slice:.1f} {ds_sweep.attrs['id']} nv:{nyquist_velocity:.2f}")

    # TODO: pass alpha
    if strategy == "default":
        outputs = dealiasing_process_2D(
            r_slice, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, alpha, debug
        )
    else:
        outputs = dealias_long_range(
            r_slice, azimuth_slice, elevation_slice, velocity_slice, nyquist_velocity, alpha, debug
        )

    if len(outputs) == 3:
        final_vel, flag_vel, completed = outputs
        log(f"Completed stage: {completed}")
    else:
        final_vel, flag_vel = outputs

    if stage_check("3d") and ds_ref:
        r_reference = ds_ref.range.values
        azimuth_reference = ds_ref.azimuth.values
        elevation_reference = ds_ref["elevation"].values[0]
        velocity_reference = ds_ref[output_vel_name].values
        flag_reference = ds_ref[output_flag_name].values

        log(f"tilt:{elevation_slice:.1f} {ds_sweep.attrs['id']} nv:{nyquist_velocity:.2f}")
        log(f"ref:{ds_ref['elevation'].values[0]:.1f} {ds_ref.attrs['id']}  nv:{ds_ref.attrs['NI']}")

        final_vel, flag_vel, _, _ = continuity.unfolding_3D(
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

    if stage_check("check-box"):
        final_vel, flag_vel = continuity.box_check(azimuth_slice, final_vel, flag_vel, nyquist_velocity, 20)

    # write results back to dataset
    ds_sweep = ds_sweep.merge(
        {output_vel_name: (("azimuth", "range"), final_vel), output_flag_name: (("azimuth", "range"), flag_vel)}
    )

    return ds_sweep
