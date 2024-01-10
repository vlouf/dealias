# python3

import itertools

import h5py
import numpy as np

import pyodim

# this is used throughout UNRAVEL
FLAG_NODATA = -3

# flag encoding matching `cent_chain.xml` settings
#
# NB: floats not ints
FLAG_ENC_GAIN = 0.1
FLAG_ENC_OFFSET = 0.0
FLAG_ENC_NODATA = 255.0

# dealiased velocity encoding matching `cent_chain.xml` settings
#
# NB: we must provide a new encoding and datatype for dealiased velocity as
# the original encoding will often max out at one nyquist interval
#
# NB: floats not ints
VEL2_ENC_GAIN = 0.025
VEL2_ENC_OFFSET = -300.0
VEL2_ENC_NODATA = 0.0
VEL2_ENC_UNDETECT = 1.0
VEL2_ENC_DTYPE = 'uint16'

def rename_old_data(
        h5_tilt,
        data_name):
    """Rename old data to avoid name clashes."""
    for d_idx in itertools.count(1):
        if not f"data{d_idx}" in h5_tilt:
            break
        d_what = h5_tilt[f"data{d_idx}/what"]
        if d_what.attrs["quantity"].decode() == data_name:
            print(f"renaming {data_name} -> {data_name}_{d_idx}")
            write_odim_str_attrib(d_what, "quantity", f"{data_name}_{d_idx}")

def write_odim_slice(
        h5file,
        ds_sweep,
        vel_name: str,
        output_vel_name: str,
        output_flag_name: str):
    """Write one slice of corrected velocity data back to ODIM HDF5 file.

    @param h5file: h5py file handle
    @param ds_sweep: dataset for slice/tilt/sweep
    @param vel_name: name of original velocity data
    @param output_vel_name: name for updated velocity data
    @param output_flag_name: name for generated velocity flag data
    """

    tilt_id = ds_sweep.attrs["id"]

    vel_meta = ds_sweep[vel_name]
    vel_id = vel_meta.attrs["id"]

    h5_tilt = h5file[tilt_id]

    # duplicate velocity group for corrected velocity
    vel2_id = pyodim.copy_h5_data(h5_tilt, vel_id)
    vel2_h5 = h5_tilt[vel2_id]
    rename_old_data(h5_tilt, output_vel_name)

    print(f"writing {tilt_id}/{vel2_id} {output_vel_name} corrected velocity")

    # update metadata for corrected velocity
    vel2_h5["what"].attrs["gain"] = VEL2_ENC_GAIN
    vel2_h5["what"].attrs["offset"] = VEL2_ENC_OFFSET
    vel2_h5["what"].attrs["nodata"] = VEL2_ENC_NODATA
    vel2_h5["what"].attrs["undetect"] = VEL2_ENC_UNDETECT
    # set quantity
    pyodim.write_odim_str_attrib(vel2_h5["what"], "quantity", output_vel_name)

    # encode (reverse read transformation) and replace data array
    vel2_data = ds_sweep[output_vel_name].values

    # NB: round() matching bom-core odim write_pack()
    vel2_encoded = np.ma.filled(
        np.round((vel2_data - VEL2_ENC_OFFSET) / VEL2_ENC_GAIN), VEL2_ENC_NODATA)
    # NB: as we use a different datatype we must recreate the dataset
    # NB: conversion from floating-point to discrete levels
    del vel2_h5["data"]
    vel2_h5.create_dataset("data", dtype=VEL2_ENC_DTYPE, data=vel2_encoded)

    # duplicate velocity group for unravel velocity flags
    flag_id = pyodim.copy_h5_data(h5_tilt, vel_id)
    flag_h5 = h5_tilt[flag_id]
    rename_old_data(ds_sweep, output_flag_name)

    print(f"writing {tilt_id}/{flag_id} {output_flag_name} flags")

    # update metadata for flags
    flag_h5_what = flag_h5["what"]
    flag_h5_what.attrs["gain"] = FLAG_ENC_GAIN
    flag_h5_what.attrs["offset"] = FLAG_ENC_OFFSET
    flag_h5_what.attrs["nodata"]   = FLAG_ENC_NODATA
    flag_h5_what.attrs["undetect"] = FLAG_ENC_NODATA
    # set quantity
    pyodim.write_odim_str_attrib(flag_h5_what, "quantity", output_flag_name)

    # encode and overwrite data array in-place
    flag_data = ds_sweep[output_flag_name].values
    flag_ma = np.ma.masked_equal(flag_data, FLAG_NODATA)
    # NB: matching bom-core odim write_pack()
    flag_encoded = np.ma.filled(
        np.round((flag_ma - FLAG_ENC_OFFSET) / FLAG_ENC_GAIN), FLAG_ENC_NODATA)
    flag_h5["data"][...] = flag_encoded
