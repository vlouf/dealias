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

"""CUR_STAGE: Tracks current stage."""
CUR_STAGE = 0


"""USE_BOX_CHECK_V1: Choose between box_check V1 (cross window) and box_check V2 (box window).
"""
USE_BOX_CHECK_V1 = False


@jit(nopython=True)
def log(*args) -> None:
    if SHOW_PROGRESS:
        print(*args)


def stage_check(stage = None):
    """Check whether to run or to skip current stage."""
    global CUR_STAGE
    if stage is not None:
        CUR_STAGE = stage
    else:
        CUR_STAGE += 1
    if not MAX_STAGE or CUR_STAGE <= MAX_STAGE:
        return True
    log(f"Skipping stage {CUR_STAGE}")
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

    def set_use_v1_box_check(self, val):
        global USE_BOX_CHECK_V1
        USE_BOX_CHECK_V1 = val

    def show_progress(self):
        return SHOW_PROGRESS

    def do_act(self):
        return DO_ACT
