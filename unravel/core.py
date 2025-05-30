"""
The dealiasing class.

@title: core.py
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 25/03/2021

.. autosummary::
    :toctree: generated/

    Dealias
"""

import traceback

import numpy as np

from . import continuity
from . import filtering
from . import initialisation
from . import find_reference
from .cfg import stage_check


class Dealias:
    """
    A class to store the velocity field and its coordinates for dealiasing.
    """

    def __init__(self, r, azimuth, elevation, velocity, nyquist_velocity, alpha=0.6):
        self.r = r
        self.azimuth = azimuth
        self.elevation = elevation
        self.velocity = self._check_velocity(velocity)
        self.nyquist = nyquist_velocity
        self.alpha = alpha
        self.alpha_mad = 0.3  # Trusted velocity difference Nyquist multiplier (for filter_data)
        self.vshift = 2 * nyquist_velocity
        self.nrays = len(azimuth)
        self.ngates = len(r)
        self._check_inputs()
        self.flag = self._gen_flag_array()
        self.dealias_vel = self._gen_empty_velocity()

    def _gen_empty_velocity(self):
        """Initialiaze empty dealiased velocity field"""
        vel = np.zeros_like(self.velocity, dtype=self.velocity.dtype)
        vel[np.isnan(self.velocity)] = np.nan
        return vel

    def _gen_flag_array(self):
        """Initialiaze empty flag field"""
        flag = np.zeros(self.velocity.shape, dtype=np.int32)
        flag[np.isnan(self.velocity)] = -3
        return flag

    def _check_velocity(self, velocity):
        """FillValue should be NaN"""
        try:
            velocity = velocity.filled(np.nan)
        except AttributeError:
            pass
        return velocity

    def _check_inputs(self):
        """Check if coordinates correspond to the velocity field dimension"""
        if self.velocity.shape != (self.nrays, self.ngates):
            raise ValueError(f"Velocity, range and azimuth shape mismatch.")

    def check_completed(self):
        """Check if there are still gates to process"""
        return (self.flag == 0).sum() <= 10

    def initialize(self):
        """Initialize the dealiasing by filtering the data, finding the radials
        of reference and executer the first pass."""

        # stage 0 (MAD filter)
        # NB: filter_data() alters self.velocity, returns as dealias_vel
        stage_check("filter", stage=0)  # set stage, don't skip (we need this stage)
        dealias_vel, flag_vel = filtering.filter_data(
            self.velocity, self.flag, self.nyquist, self.vshift, self.alpha_mad
        )

        # stage 1 (find radials)
        stage_check("find")  # increment, don't skip (we need this stage)
        azi_start_pos, azi_end_pos = find_reference.find_reference_radials(self.velocity)

        # stage 2 (init radial)
        # NB: after initialize_unfolding() dealias_vel and self.velocity differ
        if stage_check("init-radial"):
            dealias_vel, flag_vel = initialisation.initialize_unfolding(
                azi_start_pos, azi_end_pos, self.velocity, flag_vel, vnyq=self.nyquist
            )

        # stage 3 (init clock)
        if stage_check("init-clock"):
            vel = self.velocity.copy()
            vel[azi_start_pos, :] = dealias_vel[azi_start_pos, :]
            dealias_vel, flag_vel = initialisation.first_pass(
                azi_start_pos, vel, dealias_vel, flag_vel, self.nyquist, 0.75 * self.nyquist
            )

        # keep final values
        self.dealias_vel = dealias_vel
        self.flag = flag_vel
        self.azi_start_pos = azi_start_pos
        self.azi_end_pos = azi_end_pos

    def correct_range(self, window_length=6, alpha=None):
        """
        Gate-by-gate velocity dealiasing through range continuity.

        Parameters:
        ===========
        window_length: int
            Size of window to look for a reference.
        """
        if alpha is None:
            alpha = self.alpha
        dealias_vel, flag_vel = continuity.correct_range_onward(
            self.velocity, self.dealias_vel, self.flag, self.nyquist, window_len=window_length, alpha=alpha
        )
        dealias_vel, flag_vel = continuity.correct_range_backward(
            self.velocity, dealias_vel, flag_vel, self.nyquist, window_len=window_length, alpha=alpha
        )
        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def correct_clock(self, window_length=3, alpha=None):
        """
        Radial-by-radial velocity dealiasing through azimuthal continuity.

        Parameters:
        ===========
        window_length: int
            Size of window to look for a reference.
        """
        if alpha is None:
            alpha = self.alpha
        azimuth_iteration = np.arange(self.azi_start_pos, self.azi_start_pos + self.nrays) % self.nrays
        dealias_vel, flag_vel = continuity.correct_clockwise(
            self.r,
            self.azimuth,
            self.velocity,
            self.dealias_vel,
            self.flag,
            azimuth_iteration,
            self.nyquist,
            window_len=window_length,
            alpha=alpha,
        )

        azimuth_iteration = np.arange(self.azi_start_pos, self.azi_start_pos - self.nrays, -1) % self.nrays
        dealias_vel, flag_vel = continuity.correct_counterclockwise(
            self.r,
            self.azimuth,
            self.velocity,
            dealias_vel,
            flag_vel,
            azimuth_iteration,
            self.nyquist,
            window_len=window_length,
            alpha=alpha,
        )

        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def correct_box(self, window_size=(20, 20), alpha=None):
        """
        Velocity dealiasing using a 2D plane continuity.

        Parameters:
        ===========
        window_length: (int, int)
            Size of plane to look for a reference.
        """
        if alpha is None:
            alpha = self.alpha
        if window_size is int:
            window_size = (window_size, window_size)

        dealias_vel, flag_vel = continuity.correct_box(
            self.azimuth,
            self.velocity,
            self.dealias_vel,
            self.flag,
            self.nyquist,
            window_size[0],
            window_size[1],
            alpha=alpha,
        )
        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def correct_leastsquare(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if self.elevation > 6:
            return None

        # Least squares error check in the radial direction
        dealias_vel, flag_vel = continuity.radial_least_square_check(
            self.r, self.azimuth, self.velocity, self.dealias_vel, self.flag, self.nyquist, alpha=alpha
        )
        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def correct_linregress(self, alpha=None):
        """
        Gate-by-gate velocity dealiasing through range continuity using a
        linear regression.
        """
        if alpha is None:
            alpha = self.alpha
        dealias_vel, flag_vel = continuity.correct_linear_interp(
            self.velocity, self.dealias_vel, self.flag, self.nyquist, alpha=alpha
        )
        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def correct_closest(self, alpha=None):
        """
        Velocity dealiasing using the closest available reference in a 2D
        plane.
        """
        if alpha is None:
            alpha = self.alpha
        dealias_vel, flag_vel = continuity.correct_closest_reference(
            self.azimuth, self.velocity, self.dealias_vel, self.flag, self.nyquist, alpha=alpha
        )
        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def check_leastsquare(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if self.elevation > 6:
            return None

        # Least squares error check in the radial direction
        dealias_vel = continuity.least_square_radial_last_module(
            self.r, self.azimuth, self.dealias_vel, self.flag, self.nyquist, alpha=alpha
        )
        self.dealias_vel = dealias_vel

    def check_box(self, window_size=(80, 20), alpha=None):
        """
        Checking function using a 2D plane of surrounding velocities. Faster
        than the check_box_median.
        """
        if alpha is None:
            alpha = self.alpha
        try:
            dealias_vel, flag_vel = continuity.box_check(
                self.azimuth,
                self.dealias_vel,
                self.flag,
                self.nyquist,
                window_range=window_size[0],
                window_azimuth=window_size[1],
                alpha=alpha,
            )
        except IndexError:
            traceback.print_exc()
            print("check_box not executed.")
            return None

        self.dealias_vel = dealias_vel
        self.flag = flag_vel
