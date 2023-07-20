# python3

# NB: labelled stages/numbering only valid for strategy "default"
# cf: labelled stages in cent/config/unravel-test.toml
_STAGE_MAP = {
    1 : "unravel-0a-mad", # as 1
    #1 : "unravel-0b-find", # read-only
    2 : "unravel-0c-radial",
    3 : "unravel-0d-clock",
    4 : "unravel-1a-range", # first of 4,5,7
    6 : "unravel-1b-clock", # first of 6,8
    9 : "unravel-1c-box",   # first of 9-11
    12 : "unravel-2a-lsq",
    13 : "unravel-2b-extra",
    14 : "unravel-2c-closest",
    15 : "unravel-3a-lsq",
    16 : "unravel-3b-cbox",
    17 : "unravel-4a-3d",
    18 : "unravel-4b-cbox",
}

class cfg:
    """Global configuration and flags."""
    _instance = None

    def __new__(cls):
        """Singleton new."""
        if cls._instance:
            return cls._instance
        cls._instance = super(cfg, cls).__new__(cls)
        cls._instance._init_impl()
        return cls._instance

    # NB: __init__() is always called on return of __new__(), so leave it empty
    def _init_impl(self):
        """Singleton init.

        (test/debug settings)

        show_progress: when True, log each stage, along with some of the crucial
        settings (eg: alpha, window).

        do_act: when False, avoid most stages' expensive calcuations.  Use with
        `show_progess` to show which stages are run with which settings.

        short_circuit: when False, ignore check_completed() short-circuit -- run
        all stages.

        max_stage: when not 0, skip stages after max_stage is reached in
        stage_check()

        (behavioural settings)

        init_radial_no_zero: when True, don't zero untouched velocities in
        init_radial aka initialize_unfolding()

        init_radial_use_all: when True, use all updated velocities from
        initialize_unfolding() as reference, not just those from the primary
        radial.

        sign_compare_epsilon: ignore close-to-zero values in take_decision().

        correct_range_consistent: consistency in correct_range forwards and
        backwards.

        correct_clock_iter: consistency in correct_clock forwards and backwards.

        least_sq_no_nan: don't set unprocessed velocities to nans in
        unravel_least_square aka radial_least_square_check.

        least_sq_not_provisional: don't make values set in
        unravel_least_square(all) aka least_square_radial_last_module
        provisional (ie do set flag).

        post_box_check: when True, call box_check() (aka unravel_cross) after
        unfolding_3D() (as in unravel_3d_pyart_multiproc())

        """
        # cfg settings

        # - test/debug -
        self.do_act = True
        self.show_progress = False
        self.short_circuit = True
        self.max_stage = 0

        # - behaviour -
        self.init_radial_no_zero = False
        self.init_radial_use_all = False
        self.sign_compare_epsilon = 0.0
        self.correct_range_consistent = False
        self.correct_clock_iter = False
        self.least_sq_no_nan = False
        self.least_sq_not_provisional = False
        self.closest_no_delay = False
        self.post_box_check = False

        # progress
        self.cur_stage = 0

    # we require some global state for numba @jit(nopython=True) stages
    #
    # NB: SHOW_PROGRESS within numba jit: f-string floats unsupported,
    # "format spec in f-strings not supported yet" (nor in .format nor %-format)
    def update_globals(self):
        """Write globals from singleton state."""
        global SHOW_PROGRESS
        SHOW_PROGRESS = self.show_progress

        global DO_ACT
        DO_ACT = self.do_act

        global INIT_RADIAL_NO_ZERO
        INIT_RADIAL_NO_ZERO = self.init_radial_no_zero

        global SIGN_COMPARE_EPSILON
        SIGN_COMPARE_EPSILON = self.sign_compare_epsilon

        global CORRECT_RANGE_CONSISTENT
        CORRECT_RANGE_CONSISTENT = self.correct_range_consistent

        global CORRECT_CLOCK_ITER
        CORRECT_CLOCK_ITER = self.correct_clock_iter

        global LEAST_SQ_NO_NAN
        LEAST_SQ_NO_NAN = self.least_sq_no_nan

        global LEAST_SQ_NOT_PROVISIONAL
        LEAST_SQ_NOT_PROVISIONAL = self.least_sq_not_provisional

        global CLOSEST_NO_DELAY
        CLOSEST_NO_DELAY = self.closest_no_delay

    def set_max_stage(self, stage):
        """Set stage limit using number or string."""
        if isinstance(stage, int):
            self.max_stage = stage
            return
        for (key, val) in _STAGE_MAP.items():
            if stage == val:
                print(f"setting max_stage {key} {val}")
                self.max_stage = key
                return
        raise Exception(f"unrecognised stage {stage}")

    def mark_stage_done(self, stage):
        """Mark stage completion using string."""
        for (key, val) in _STAGE_MAP.items():
            if stage == val:
                self.cur_stage = key
                return
        raise Exception(f"unrecognised stage {stage}")

    def stage_check(self):
        """Check whether to run or skip current stage."""
        self.cur_stage += 1
        if not self.max_stage or self.cur_stage <= self.max_stage:
            return True
        stage_name = _STAGE_MAP[self.cur_stage] if self.cur_stage in _STAGE_MAP else ""
        if self.show_progress:
            print(f"Skipping stage {self.cur_stage} {stage_name}")
        return False

    def show(self):
        """Show settings."""
        print(" ".join([
            f"show_progress:{self.show_progress}",
            f"do_act:{self.do_act}",
            f"short_circuit:{self.short_circuit}",
            f"iradial_no_zero:{self.init_radial_no_zero}",
            f"iradial_use_all:{self.init_radial_use_all}",
            f"sign_compare_eps:{self.sign_compare_epsilon}",
            f"range_consistent:{self.correct_range_consistent}",
            f"clock_iter:{self.correct_clock_iter}",
            f"lsq_no_nan:{self.least_sq_no_nan}",
            f"lsq_no_prov:{self.least_sq_not_provisional}",
            f"post_box_check:{self.post_box_check}",
            f"max_stage:{self.max_stage}"]))
