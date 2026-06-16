from .cfg import Cfg
from .dealias import (
    unravel_3D_pyart,
    unravel_3D_pyodim,
    dealias_long_range,
    dealiasing_process_2D,
    unravel_3D_pyart_multiproc,
)


def warmup() -> None:
    """Trigger numba JIT compilation for all compiled functions.

    Call this once in the main process before spawning workers (dask, multiprocessing).
    On fork-based systems (Linux HPC) workers inherit the compiled code from the
    parent and pay zero compilation cost.  On spawn-based systems the cache written
    here is reused by each worker, avoiding recompilation.
    """
    import numpy as np
    from . import continuity, filtering, initialisation
    from .cfg import log

    nrays, ngates = 64, 128
    vnyq = 13.0
    r = np.linspace(500.0, 4000.0, ngates)
    azi = np.linspace(0.0, 360.0, nrays, endpoint=False)
    vel = np.zeros((nrays, ngates), dtype=np.float64)
    flag = np.zeros((nrays, ngates), dtype=np.int32)
    flag[0, 0] = 1  # at least one processed gate so closest-ref has a reference
    fv = vel.copy()
    fv[flag <= 0] = np.nan

    # cfg.log
    log("warmup")

    # filtering
    filtering.unfold(0.0, 0.0, vnyq, 2 * vnyq)
    filtering.filter_data(vel.copy(), flag.copy(), vnyq, 2 * vnyq, 0.8)

    # initialisation — touch a representative subset
    azipos = np.arange(nrays, dtype=np.int64)
    initialisation.find_last_good_vel(1, 0, azipos, flag.copy(), 3)
    initialisation.flipud(vel)
    initialisation.first_pass(0, vel, fv, flag.copy(), vnyq, 2 * vnyq)
    initialisation.initialize_unfolding(0, 2, vel, flag.copy(), vnyq)

    # continuity — all JIT-compiled functions via jit_module
    quadrant = np.arange(nrays, dtype=np.int64)
    continuity.correct_clockwise(r, azi, vel, fv, flag.copy(), quadrant, vnyq)
    continuity.correct_counterclockwise(r, azi, vel, fv, flag.copy(), quadrant, vnyq)
    continuity.correct_range_onward(vel, fv, flag.copy(), vnyq)
    continuity.correct_range_backward(vel, fv, flag.copy(), vnyq)
    continuity.correct_linear_interp(vel, fv, flag.copy(), vnyq)
    continuity.radial_least_square_check(r, azi, vel, fv, flag.copy(), vnyq)
    continuity.least_square_radial_last_module(r, azi, fv, flag.copy(), vnyq)
    continuity.correct_box(azi, vel, fv, flag.copy(), vnyq)
    continuity.box_check(azi, fv, flag.copy(), vnyq)
    continuity.correct_closest_reference(azi, vel, fv, flag.copy(), vnyq)
    continuity.unfolding_3D(
        r, azi, 0.5, fv, flag.copy(),
        r, azi, 1.5, fv.copy(), flag.copy(), vel, vnyq,
    )
