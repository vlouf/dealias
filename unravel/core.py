"""
The dealiasing class.

@title: core.py
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 16/09/2026

.. autosummary::
    :toctree: generated/

    unmask_array
    Dealias
"""

import traceback
from typing import Optional, Tuple, Union

import numpy as np

from . import continuity
from . import filtering
from . import initialisation
from . import find_reference
from .cfg import stage_check


def unmask_array(x: Union[np.ndarray, np.ma.MaskedArray], fill_value=np.nan) -> np.ndarray:
    """
    Return a plain ndarray, replacing the mask (if any) by `fill_value`.

    Parameters:
    ===========
    x: ndarray or MaskedArray
        Array to unmask. Scalars are returned unchanged.
    fill_value:
        Value used to replace the masked elements. Default is NaN.

    Returns:
    ========
    x: ndarray
        Unmasked array.
    """
    try:
        x = x.filled(fill_value)
    except AttributeError:
        pass
    return x


class Dealias:
    """
    Dealiasing class to perform the dealiasing of a velocity field.

    Parameters:
    ===========
    r: np.ndarray
        Range coordinates of the radar.
    azimuth: np.ndarray
        Azimuth coordinates of the radar.
    elevation: float
        Elevation angle of the radar.
    velocity: np.ndarray
        Velocity field to dealias.
    nyquist_velocity: float
        Nyquist velocity of the radar.
    alpha: float
        Alpha parameter for the dealiasing. Default is 0.6.
    alpha_mad: float
        Trusted velocity difference Nyquist multiplier used by the MAD filter.
        Default is 0.3.
    """

    # Fraction of unprocessed gates below which a sweep counts as completed.
    COMPLETION_THRESHOLD: float = 0.01
    # Nyquist fraction used as the tolerance of the first (clockwise) pass.
    FIRST_PASS_NYQUIST_FRACTION: float = 0.75
    # Above this elevation angle (degrees) the radial least-square modules are
    # skipped: the velocity is no longer dominated by the horizontal wind.
    MAX_LEASTSQUARE_ELEVATION: float = 6.0

    def __init__(
        self,
        r: np.ndarray,
        azimuth: np.ndarray,
        elevation: float,
        velocity: np.ndarray,
        nyquist_velocity: float,
        alpha: float = 0.6,
        alpha_mad: float = 0.3,
    ):
        self._check_inputs(r, azimuth, velocity, alpha, alpha_mad)

        self.r = r
        self.azimuth = azimuth
        self.elevation = elevation
        self.velocity: np.ndarray = self._check_velocity(velocity)
        self.nyquist: float = nyquist_velocity
        self.alpha = alpha
        self.alpha_mad = alpha_mad
        self.vshift: float = 2 * nyquist_velocity
        self.nrays: int = len(azimuth)
        self.ngates: int = len(r)
        self.flag: np.ndarray = self._gen_flag_array()
        self.dealias_vel: np.ndarray = self._gen_empty_velocity()

        # Position of the reference radials, only known once initialize() has run.
        self.azi_start_pos: Optional[int] = None
        self.azi_end_pos: Optional[int] = None

    def _gen_empty_velocity(self) -> np.ndarray:
        """Initialiaze empty dealiased velocity field"""
        vel = np.zeros_like(self.velocity, dtype=self.velocity.dtype)
        vel[np.isnan(self.velocity)] = np.nan
        return vel

    def _gen_flag_array(self) -> np.ndarray:
        """Initialiaze empty flag field"""
        flag = np.zeros(self.velocity.shape, dtype=np.int32)
        flag[np.isnan(self.velocity)] = -3
        return flag

    @staticmethod
    def _check_velocity(velocity) -> np.ndarray:
        """
        Unmask the velocity field (FillValue should be NaN) and copy it: the
        MAD filter of initialize() unfolds the velocity in place, so we must
        never write into the array owned by the caller.
        """
        return unmask_array(velocity).copy()

    @staticmethod
    def _check_inputs(r, azimuth, velocity, alpha, alpha_mad) -> None:
        """Validate the parameters and the velocity field dimensions."""
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha parameter should be between 0 and 1, got {alpha}.")
        if not 0 <= alpha_mad <= 1:
            raise ValueError(f"Alpha MAD parameter should be between 0 and 1, got {alpha_mad}.")
        if velocity.ndim != 2:
            raise ValueError(f"Velocity field should be a 2D array, got {velocity.ndim} dimension(s).")
        expected = (len(azimuth), len(r))
        if velocity.shape != expected:
            raise ValueError(
                f"Velocity, range and azimuth shape mismatch: velocity is {velocity.shape}, "
                f"expected {expected} <azimuth, range>."
            )

    def _alpha(self, alpha: Optional[float]) -> float:
        """Fall back on the instance alpha when the caller does not override it."""
        return self.alpha if alpha is None else alpha

    def _apply(self, fn, *args, alpha: Optional[float] = None, **kwargs) -> None:
        """
        Run a continuity module and absorb its (dealiased velocity, flag) result.

        Every module reads the current velocity/flag state and returns the updated
        pair; storing it back happens here so that the write-back is expressed once
        rather than at the end of each module wrapper.
        """
        kwargs["alpha"] = self._alpha(alpha)
        self.dealias_vel, self.flag = fn(*args, **kwargs)

    def check_completed(self) -> bool:
        """Check if there are still gates to process"""
        valid = (self.flag != -3).sum()
        if valid == 0:
            return True
        return (self.flag == 0).sum() / valid <= self.COMPLETION_THRESHOLD

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
                azi_start_pos,
                vel,
                dealias_vel,
                flag_vel,
                self.nyquist,
                self.FIRST_PASS_NYQUIST_FRACTION * self.nyquist,
            )

        # keep final values
        self.dealias_vel = dealias_vel
        self.flag = flag_vel
        self.azi_start_pos = azi_start_pos
        self.azi_end_pos = azi_end_pos

    def correct_range(self, window_length: int = 6, alpha: Optional[float] = None):
        """
        Gate-by-gate velocity dealiasing through range continuity.

        Parameters:
        ===========
        window_length: int
            Size of window to look for a reference.
        """
        self._apply(
            continuity.correct_range_onward,
            self.velocity,
            self.dealias_vel,
            self.flag,
            self.nyquist,
            window_len=window_length,
            alpha=alpha,
        )
        self._apply(
            continuity.correct_range_backward,
            self.velocity,
            self.dealias_vel,
            self.flag,
            self.nyquist,
            window_len=window_length,
            alpha=alpha,
        )

    def correct_clock(self, window_length: int = 3, alpha: Optional[float] = None):
        """
        Radial-by-radial velocity dealiasing through azimuthal continuity.

        Parameters:
        ===========
        window_length: int
            Size of window to look for a reference.
        """
        if self.azi_start_pos is None:
            raise RuntimeError("Reference radials unknown: initialize() must be called before correct_clock().")

        azimuth_iteration = np.arange(self.azi_start_pos, self.azi_start_pos + self.nrays) % self.nrays
        self._apply(
            continuity.correct_clockwise,
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
        self._apply(
            continuity.correct_counterclockwise,
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

    def correct_box(self, window_size: Union[int, Tuple[int, int]] = (20, 20), alpha: Optional[float] = None):
        """
        Velocity dealiasing using a 2D plane continuity.

        Parameters:
        ===========
        window_length: (int, int)
            Size of plane to look for a reference.
        """
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        self._apply(
            continuity.correct_box,
            self.azimuth,
            self.velocity,
            self.dealias_vel,
            self.flag,
            self.nyquist,
            window_size[0],
            window_size[1],
            alpha=alpha,
        )

    def correct_leastsquare(self, alpha: Optional[float] = None):
        if self.elevation > self.MAX_LEASTSQUARE_ELEVATION:
            return None

        # Least squares error check in the radial direction
        self._apply(
            continuity.radial_least_square_check,
            self.r,
            self.azimuth,
            self.velocity,
            self.dealias_vel,
            self.flag,
            self.nyquist,
            alpha=alpha,
        )

    def correct_linregress(self, alpha: Optional[float] = None):
        """
        Gate-by-gate velocity dealiasing through range continuity using a
        linear regression.
        """
        self._apply(
            continuity.correct_linear_interp, self.velocity, self.dealias_vel, self.flag, self.nyquist, alpha=alpha
        )

    def correct_closest(self, alpha: Optional[float] = None):
        """
        Velocity dealiasing using the closest available reference in a 2D
        plane.
        """
        self._apply(
            continuity.correct_closest_reference,
            self.azimuth,
            self.velocity,
            self.dealias_vel,
            self.flag,
            self.nyquist,
            alpha=alpha,
        )

    def check_leastsquare(self, alpha: Optional[float] = None):
        if self.elevation > self.MAX_LEASTSQUARE_ELEVATION:
            return None

        # Least squares error check in the radial direction. This module returns
        # the velocity alone, so it cannot go through _apply().
        self.dealias_vel = continuity.least_square_radial_last_module(
            self.r, self.azimuth, self.dealias_vel, self.flag, self.nyquist, alpha=self._alpha(alpha)
        )

    def check_box(self, window_size: Tuple[int, int] = (80, 20), alpha: Optional[float] = None):
        """
        Checking function using a 2D plane of surrounding velocities. Faster
        than the check_box_median.
        """
        try:
            self._apply(
                continuity.box_check,
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
