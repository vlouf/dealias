# python3

import itertools
import h5py
import numpy as np

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

def odim_str_type_id(text_bytes):
    """Generate ODIM-conformant string type ID."""
    # string type (h5py default is STRPAD STR_NULLPAD, ODIM spec is STRPAD STR_NULLTERM)
    type_id = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
    type_id.set_strpad(h5py.h5t.STR_NULLTERM)
    type_id.set_size(len(text_bytes) + 1)
    return type_id

def write_odim_str_attrib(group, attrib_name: str, text: str):
    """Write ODIM-conformant string attribute."""
    group_id = group.id
    text_bytes = text.encode('utf-8')
    type_id = odim_str_type_id(text_bytes)
    space = h5py.h5s.create(h5py.h5s.SCALAR)
    att_id = h5py.h5a.create(group_id, attrib_name.encode('utf-8'), type_id, space)
    text_array = np.array(text_bytes)
    att_id.write(text_array)

def h5_copy_data(h5_tilt, ds_sweep, orig_id):
    """Add a data array to `h5_tilt` by copying `orig_id`.

    @return new id
    """
    # increment data_count
    data_count = ds_sweep.attrs["data_count"] + 1
    ds_sweep.attrs["data_count"] = data_count

    # use data_count for data_id
    data_id = f"data{data_count}"

    # duplicate original
    h5_tilt.copy(orig_id, data_id)

    # return id
    return data_id

def rename_old_data(
        h5_tilt,
        # ds_sweep,
        data_name):
    """Rename old data to avoid name clashes."""
    for d_idx in itertools.count(1):
        if not f"data{d_idx}" in h5_tilt:
            break
        d_what = h5_tilt[f"data{d_idx}/what"]
        if d_what.attrs["quantity"].decode() == data_name:
            print(f"renaming {data_name} -> {data_name}_{d_idx}")
            del d_what.attrs["quantity"]
            write_odim_str_attrib(d_what, "quantity", f"{data_name}_{d_idx}")

def write_odim_slice(
        h5file,
        ds_sweep,
        vel_name,
        output_vel_name,
        output_flag_name):
    """Write data for one slice back to ODIM HDF5 file."""

    tilt_id = ds_sweep.attrs["id"]

    vel_meta = ds_sweep[vel_name]
    vel_id = vel_meta.attrs["id"]

    h5_tilt = h5file[tilt_id]

    # duplicate velocity group for corrected velocity
    vel2_id = h5_copy_data(h5_tilt, ds_sweep, vel_id)
    vel2_h5 = h5_tilt[vel2_id]
    rename_old_data(h5_tilt, output_vel_name)

    print(f"writing {tilt_id}/{vel2_id} {output_vel_name} corrected velocity")

    # update metadata for corrected velocity
    vel2_h5["what"].attrs["gain"] = VEL2_ENC_GAIN
    vel2_h5["what"].attrs["offset"] = VEL2_ENC_OFFSET
    vel2_h5["what"].attrs["nodata"] = VEL2_ENC_NODATA
    vel2_h5["what"].attrs["undetect"] = VEL2_ENC_UNDETECT
    # set quantity
    del vel2_h5["what"].attrs["quantity"]
    write_odim_str_attrib(vel2_h5["what"], "quantity", output_vel_name)

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
    flag_id = h5_copy_data(h5_tilt, ds_sweep, vel_id)
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
    del flag_h5_what.attrs["quantity"]
    write_odim_str_attrib(flag_h5_what, "quantity", output_flag_name)

    # encode and overwrite data array in-place
    flag_data = ds_sweep[output_flag_name].values
    flag_ma = np.ma.masked_equal(flag_data, FLAG_NODATA)
    # NB: matching bom-core odim write_pack()
    flag_encoded = np.ma.filled(
        np.round((flag_ma - FLAG_ENC_OFFSET) / FLAG_ENC_GAIN), FLAG_ENC_NODATA)
    flag_h5["data"][...] = flag_encoded
