#!/usr/bin/python3
"""Run UNRAVEL on ODIM HDF5 file.

USAGE:  PATH.pvol.h5

example usage:
 $ python3 -m unravel path/to/odim-file.pvol.h5

NOTE: we don't even use `pyart` in `unravel_3D_pyodim` however it's still
imported and it's noisy, so we recommend calling unravel with PYART_QUIET=1 in
the environment, eg:
 $ PYART_QUIET=1 python3 -m unravel path/to/odim-file.pvol.h5
"""

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

def main():
    """Run UNRAVEL."""

    if len(sys.argv) < 2:
        usage()
    in_path = sys.argv[1]

    print(f"Processing {in_path}")
    cfg = Cfg()
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

    # use "default" not "long_range" for testing as default uses all filters
    unravel_3D_pyodim(
        out_path,
        strategy="default",
        output_vel_name="VRADDH",
        output_flag_name="V_FLG",
        read_write=cfg.do_act())

if __name__ == "__main__":
    main()
