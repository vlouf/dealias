
import numpy as np

from . import continuity
from . import filtering
from . import initialisation
from . import find_reference


class Dealias:
    def __init__(self, r, azimuth, elevation, velocity, nyquist_velocity, alpha=0.6):
        self.r = r
        self.azimuth = azimuth
        self.elevation = elevation
        self.velocity = velocity
        self.nyquist = nyquist_velocity
        self.alpha = alpha
        self.vshift = 2 * nyquist_velocity
        self.delta_vmax = 0.5 * nyquist_velocity
        self.nrays = len(azimuth)
        self.ngates = len(r)
        self._check_inputs()
        self.flag = self._gen_flag_array()

    def _gen_flag_array(self):
        flag = np.zeros(self.velocity.shape, dtype=np.int32)
        flag[np.isnan(self.velocity)] = -3
        return flag

    def _check_inputs(self):
        if self.velocity.shape != (self.nrays, self.ngates):
            raise ValueError(f'Velocity, range and azimuth shape mismatch.')

    def check_completed(self):
        return (self.flag == 0).sum() <= 10

    def initialize(self):
        dealias_vel, flag_vel = filtering.filter_data(self.velocity,
                                                      self.flag,
                                                      self.nyquist,
                                                      self.vshift,
                                                      self.delta_vmax)
        self.velocity[flag_vel == -3] = np.NaN
        start_beam, end_beam = find_reference.find_reference_radials(self.azimuth, self.velocity)
        azi_start_pos = np.argmin(np.abs(self.azimuth - start_beam))
        azi_end_pos = np.argmin(np.abs(self.azimuth - end_beam))
        dealias_vel, flag_vel = initialisation.initialize_unfolding(self.r,
                                                                    self.azimuth,
                                                                    azi_start_pos,
                                                                    azi_end_pos,
                                                                    self.velocity,
                                                                    flag_vel,
                                                                    vnyq=self.nyquist)
        vel = self.velocity.copy()
        vel[azi_start_pos, :] = dealias_vel[azi_start_pos, :]
        dealias_vel, flag_vel = initialisation.first_pass(azi_start_pos,
                                                          vel,
                                                          dealias_vel,
                                                          flag_vel,
                                                          self.nyquist,
                                                          self.vshift,
                                                          0.75 * self.nyquist)
        self.dealias_vel = dealias_vel
        self.flag = flag_vel
        self.azi_start_pos = azi_start_pos
        self.azi_end_pos = azi_end_pos

    def correct_range(self, window_length=6, alpha=None):
        if alpha is None:
            alpha = self.alpha
        dealias_vel, flag_vel = continuity.correct_range_onward(self.velocity,
                                                                self.dealias_vel,
                                                                self.flag,
                                                                self.nyquist,
                                                                window_len=window_length,
                                                                alpha=alpha)
        dealias_vel, flag_vel = continuity.correct_range_backward(self.velocity,
                                                                  dealias_vel,
                                                                  flag_vel,
                                                                  self.nyquist,
                                                                  window_len=window_length,
                                                                  alpha=alpha)
        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def correct_clock(self, window_length=3, alpha=None):
        if alpha is None:
            alpha = self.alpha
        azimuth_iteration = np.arange(self.azi_start_pos, self.azi_start_pos + self.nrays) % self.nrays
        dealias_vel, flag_vel = continuity.correct_clockwise(self.r,
                                                             self.azimuth,
                                                             self.velocity,
                                                             self.dealias_vel,
                                                             self.flag,
                                                             azimuth_iteration,
                                                             self.nyquist,
                                                             window_len=window_length,
                                                             alpha=alpha)
        dealias_vel, flag_vel = continuity.correct_counterclockwise(self.r,
                                                                    self.azimuth,
                                                                    self.velocity,
                                                                    dealias_vel,
                                                                    flag_vel,
                                                                    azimuth_iteration,
                                                                    self.nyquist,
                                                                    window_len=window_length,
                                                                    alpha=alpha)
        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def correct_box(self, window_size=(20, 20), alpha=None):
        if alpha is None:
            alpha = self.alpha
        if window_size is int:
            window_size = (window_size, window_size)
            
        dealias_vel, flag_vel = continuity.correct_box(self.azimuth,
                                                       self.velocity,
                                                       self.dealias_vel,
                                                       self.flag,
                                                       self.nyquist,
                                                       window_size[0],
                                                       window_size[1],
                                                       alpha=alpha)
        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def correct_linregress(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        dealias_vel, flag_vel = continuity.correct_linear_interp(self.velocity,
                                                                 self.dealias_vel,
                                                                 self.flag,
                                                                 self.nyquist,
                                                                 alpha=alpha)
        self.dealias_vel = dealias_vel
        self.flag = flag_vel

    def correct_closest(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        dealias_vel, flag_vel = continuity.correct_closest_reference(self.azimuth,
                                                                     self.velocity,
                                                                     self.dealias_vel,
                                                                     self.flag,
                                                                     self.nyquist,
                                                                     alpha=alpha)
        self.dealias_vel = dealias_vel
        self.flag = flag_vel