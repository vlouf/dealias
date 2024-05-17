#!/usr/bin/python3
"""Run UNRAVEL velocity dealiasing on ODIM HDF5 file.

example usage:
 $ python3 -m unravel path/to/odim-file.pvol.h5

NOTE: we don't even use `pyart` in `unravel_3D_pyodim` however it's still
imported and it's noisy, so we recommend calling unravel with PYART_QUIET=1 in
the environment.
"""
import argparse
import os
import re
import shutil
import stat
import sys

from .cfg import Cfg
from .dealias import unravel_3D_pyodim

def configure(c):

    ## run logic ##

    # do we actually want to run / change things?
    c.set_do_act(True)

    # show progress?
    c.set_show_progress(True)

    ## variations ##

    # choose between box_check v1(cross window) and v2(box window)
    c.set_use_v1_box_check(True)

def usage():
    """Print usage and exit."""
    print(__doc__)
    sys.exit(1)

def main(args):
    """Run UNRAVEL."""

    cfg = Cfg()

    in_path = args.odim_file
    if args.max_stage:
        print(f"setting max stage: {args.max_stage}")
        cfg.set_max_stage(args.max_stage)

    print(f"Processing {in_path}")
    configure(cfg)

    out_path = in_path
    if cfg.do_act():
        out_path = re.sub(".pvol.h5", ".unravel.h5", out_path)
        out_path = re.sub("input/", "output/", out_path)
        if out_path == in_path:
            print(f"bad path in:{in_path} out:{out_path}")
            usage()

        print(f"Generating {out_path}")
        shutil.copy(in_path, out_path)

        # fix file mode if needed
        out_mode = os.stat(out_path).st_mode
        want_mode = stat.S_IREAD | stat.S_IWRITE
        if out_mode & want_mode != want_mode:
            os.chmod(out_path, out_mode | want_mode)

    # TODO: support other unravel entry points...
    unravel_3D_pyodim(
        out_path,
        strategy=args.strategy,
        output_vel_name="VRADDH",
        output_flag_name="V_FLG",
        read_write=cfg.do_act())

if __name__ == "__main__":
    parser = argparse.ArgumentParser("\n" + __doc__ + "\nunravel")
    parser.add_argument("odim_file")
    parser.add_argument("--strategy", default="default")
    parser.add_argument("--max-stage", type=int)
    args = parser.parse_args()

    main(args)
