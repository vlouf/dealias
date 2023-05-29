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

        post_box_check: when True, call box_check() after unfolding_3D() (like
        in unravel_3d_pyart_multiproc())


        """
        # cfg settings

        # - test/debug -
        self.do_act = True
        self.show_progress = False
        self.short_circuit = True
        self.max_stage = 0

        # - behaviour -
        self.post_box_check = False

        # progress
        self.cur_stage = 0

    # NB: we require some global state for numba @jit(nopython=True) stages
    #
    # NB: SHOW_PROGRESS within numba jit: f-string floats unsupported,
    # "format spec in f-strings not supported yet" (nor in .format nor %-format)
    def update_globals(self):
        """Write globals from singleton state."""
        global DO_ACT
        global SHOW_PROGRESS
        DO_ACT = self.do_act
        SHOW_PROGRESS = self.show_progress

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
            f"post_box_check:{self.post_box_check}",
            f"short_circuit:{self.short_circuit}",
            f"max_stage:{self.max_stage}"]))
