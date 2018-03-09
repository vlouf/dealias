# UNRAVEL

UNRAVEL (UNfold RAdar VELocit) is a dealiasing technique for unfolding Doppler radar velocity. It is based on the continuous consistency of the velocity field in the 3 directions (range, azimuth, elevation).

## Dependencies

Mandatory dependencies:
- [Numpy][1]
- [Numba][2]

Even thought it does not directly requires [Py-ART][3] to run, UNRAVEL uses [Py-ART][3]
data class as input.

[1]: http://www.scipy.org/
[2]: http://www.scipy.org/
[3]: https://github.com/ARM-DOE/pyart

<!-- # References: -->

<!-- Based upon the work of:
J. Zhang and S. Wang, "An automated 2D multipass Doppler radar velocity dealiasing scheme," J. Atmos. Ocean. Technol., vol. 23, no. 9, pp. 1239–1248, 2006.
G. He, G. Li, X. Zou, and P. S. Ray, "A velocity dealiasing scheme for synthetic C-band data from China’s new generation weather radar system (CINRAD)," J. Atmos. Ocean. Technol., vol. 29, no. 9, pp. 1263–1274, 2012.
G. Li, G. He, X. Zou, and P. S. Ray, "A velocity dealiasing scheme for C-band weather radar systems," Adv. Atmos. Sci., vol. 31, no. 1, pp. 17–26, 2014. -->
