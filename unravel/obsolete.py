from .continuity import *


@jit(nopython=True, cache=True)
def _convolve_check(azi, velref, final_vel, flag_vel, vnyq, alpha):
    """
    JIT part of the convolution_check function.
    """
    maxazi, maxrange = final_vel.shape
    for nbeam in range(maxazi):
        for ngate in range(maxrange):
            if flag_vel[nbeam, ngate] <= 0:
                continue

            if not is_good_velocity(velref[nbeam, ngate], final_vel[nbeam, ngate], vnyq, alpha=alpha):
                final_vel[nbeam, ngate] = velref[nbeam, ngate]
                flag_vel[nbeam, ngate] = 3

    return final_vel, flag_vel


def convolution_check(
    azi, final_vel, flag_vel, vnyq, window_range=80, window_azimuth=20, strategy="surround", alpha=0.8
):
    """
    Faster version of the box_check this time using a convolution product.

    Parameters:
    ===========
    azi: ndarray
        Radar scan azimuth.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vnyq: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array NEW value: 3->had to be corrected.
    """
    from astropy.convolution import convolve
    
    # Odd number only
    if window_range % 2 == 0:
        window_range += 1
    if window_azimuth % 2 == 0:
        window_azimuth += 1

    kernel = np.zeros((window_azimuth, window_range)) + 1
    kernel = kernel / kernel.sum()
    velref = convolve(np.ma.masked_where(flag_vel < 1, final_vel), kernel, nan_treatment="interpolate")

    final_vel, flag_vel = _convolve_check(azi, velref, final_vel, flag_vel, vnyq, alpha)
    return final_vel, flag_vel


@jit(nopython=True, cache=True)
def box_check_v0(azi, final_vel, flag_vel, vnyq, window_range=80, window_azimuth=20, strategy="surround", alpha=0.8):
    """
    Check if all individual points are consistent with their surrounding
    velocities based on the median of an area of corrected velocities preceding
    the gate being processed. This module is similar to the dealiasing technique
    from Bergen et al. (1988). This function will look at ALL points.

    Parameters:
    ===========
    azi: ndarray
        Radar scan azimuth.
    final_vel: ndarray <azimuth, r>
        Dealiased Doppler velocity field.
    flag_vel: ndarray int <azimuth, range>
        Flag array -3: No data, 0: Unprocessed, 1: good as is, 2: dealiased.
    vnyq: float
        Nyquist velocity.

    Returns:
    ========
    dealias_vel: ndarray <azimuth, range>
        Dealiased velocity slice.
    flag_vel: ndarray int <azimuth, range>
        Flag array NEW value: 3->had to be corrected.
    """
    if strategy == "vertex":
        azi_window_offset = window_azimuth
    else:
        azi_window_offset = window_azimuth // 2

    maxazi, maxrange = final_vel.shape
    for nbeam in range(maxazi):
        for ngate in np.arange(maxrange - 1, -1, -1):
            if flag_vel[nbeam, ngate] <= 0:
                continue

            myvel = final_vel[nbeam, ngate]

            npos_range = iter_range(ngate, window_range, maxrange)

            flag_ref_vec = np.zeros((len(npos_range) * window_azimuth)) + np.NaN
            vel_ref_vec = np.zeros((len(npos_range) * window_azimuth)) + np.NaN

            cnt = -1
            for na in iter_azimuth(azi, nbeam - azi_window_offset, window_azimuth):
                for nr in npos_range:
                    cnt += 1
                    if (na, nr) == (nbeam, ngate):
                        continue
                    vel_ref_vec[cnt] = final_vel[na, nr]
                    flag_ref_vec[cnt] = flag_vel[na, nr]

            if np.sum(flag_ref_vec >= 1) == 0:
                continue

            true_vel = vel_ref_vec[flag_ref_vec >= 1]
            mvel = np.nanmean(true_vel)
            svel = np.nanstd(true_vel)
            myvelref = np.nanmedian(true_vel[(true_vel >= mvel - svel) & (true_vel <= mvel + svel)])

            if not is_good_velocity(myvelref, myvel, vnyq, alpha=alpha):
                final_vel[nbeam, ngate] = myvelref
                flag_vel[nbeam, ngate] = 3

    return final_vel, flag_vel