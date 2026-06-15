# python3

from numba import jit

"""SHOW_PROGRESS: When True, log each stage, along with some of the crucial settings
(eg: alpha, window).

NB: If log() call site is within numba jit, f-string floats unsupported (nor is
.format nor %-format) "format spec in f-strings not supported yet".

NB: we can't use the `logging` module here as it's not supported in jit contexts.
"""
SHOW_PROGRESS = False

"""DO_ACT: when False, avoid most stages' expensive calcuations.
Use with `SHOW_PROGESS` to show which stages are run with which settings.
"""
DO_ACT = True

"""MAX_STAGE: When non-zero, skip stages after CUR_STAGE==MAX_STAGE in
stage_check()."""
MAX_STAGE = 0

"""SKIP_STAGE: When non-zero, skip given stage in stage_check()."""
SKIP_STAGE = 0

"""CUR_STAGE: Tracks current stage."""
CUR_STAGE = 0


"""USE_BOX_CHECK_V1: Choose between box_check V1 (cross window) and box_check V2 (box window).
"""
USE_BOX_CHECK_V1 = False


"""USE_BOX_CHECK_CONV: Use the fast separable (convolution-based) box check.

When True (default), box_check() uses box_check_conv(), which computes the masked
windowed-mean reference via cumulative sums (O(rays*gates)) instead of the per-gate
window gather of box_check_v2 (O(rays*gates*window)). It is provably identical to
box_check_v2 wherever a window is internally consistent (the 1-sigma trim is then a
no-op); it only differs in inconsistent windows (mixed aliased/dealiased velocities,
i.e. noise). Set to False to fall back to the exact, slower box_check_v2 (e.g. for
bit-exact regression on precipitation volumes).
"""
USE_BOX_CHECK_CONV = True


"""CONV_MIN_NYQUIST: Low-Nyquist guard for the convolution-based fast paths.

The separable convolution replaces a per-gate robust statistic (1-sigma-trimmed
mean / window median) with a windowed mean. This is identical to the exact path
wherever a window is internally consistent, which holds for coherent echo at high
Nyquist. On low-Nyquist sweeps the (2D-dealiased) velocity field can span several
folds within a single window, so the windowed mean diverges from the robust
statistic even on genuine signal. Sweeps whose Nyquist velocity is below this
threshold therefore use the exact path. Set to 0.0 to apply the fast path to all
sweeps regardless of Nyquist.
"""
CONV_MIN_NYQUIST = 10.0


"""USE_3D_CONV / USE_CLOSEST_CONV: fast separable paths for unfolding_3D and
correct_closest_reference, analogous to USE_BOX_CHECK_CONV and subject to the same
CONV_MIN_NYQUIST low-Nyquist guard. Set to False to use the exact, slower path.
"""
USE_3D_CONV = True
USE_CLOSEST_CONV = True


@jit(nopython=True)
def log(*args) -> None:
    if SHOW_PROGRESS:
        print(*args)


def stage_check(name, completed=None, stage=None):
    """Check whether to run or to skip current stage."""
    global CUR_STAGE
    if stage is not None:
        CUR_STAGE = stage
    else:
        CUR_STAGE += 1
    skip = None
    if completed:
        skip = f"completed in {completed}"
    elif MAX_STAGE and CUR_STAGE > MAX_STAGE:
        skip = "max reached"
    elif SKIP_STAGE and CUR_STAGE == SKIP_STAGE:
        skip = "skip current"
    if not skip:
        return True
    log(f"Skipping stage {CUR_STAGE} {name}: {skip}")
    return False


class Cfg:
    """Global configuration and flags."""

    def set_show_progress(self, val):
        global SHOW_PROGRESS
        SHOW_PROGRESS = val

    def set_do_act(self, val):
        global DO_ACT
        DO_ACT = val

    def set_max_stage(self, val):
        global MAX_STAGE
        MAX_STAGE = val

    def set_skip_stage(self, val):
        global SKIP_STAGE
        SKIP_STAGE = val

    def set_use_v1_box_check(self, val):
        global USE_BOX_CHECK_V1
        USE_BOX_CHECK_V1 = val

    def set_use_conv_box_check(self, val):
        global USE_BOX_CHECK_CONV
        USE_BOX_CHECK_CONV = val

    def set_conv_min_nyquist(self, val):
        global CONV_MIN_NYQUIST
        CONV_MIN_NYQUIST = val

    def set_use_3d_conv(self, val):
        global USE_3D_CONV
        USE_3D_CONV = val

    def set_use_closest_conv(self, val):
        global USE_CLOSEST_CONV
        USE_CLOSEST_CONV = val

    def show_progress(self):
        return SHOW_PROGRESS

    def do_act(self):
        return DO_ACT
