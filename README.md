[![DOI](https://zenodo.org/badge/119326382.svg)](https://zenodo.org/badge/latestdoi/119326382)

# UNRAVEL

UNRAVEL (UNfold RAdar VELocity) is an open-source modular Doppler velocity dealiasing algorithm for weather radars. UNRAVEL is an algorithm that does not need external reference velocity data, making it easily applicable. The proposed algorithm includes eleven core modules and two dealiasing strategies. UNRAVEL is an iterative algorithm. The goal is to build the dealiasing results starting with the strictest possible continuity tests in azimuth and range and, after each step, relaxing the parameters to include more results from a progressively growing number of reference points. UNRAVEL also has modules that perform 3D continuity checks. Thanks to this modular design, the number of dealiasing strategies can be expanded in order to optimise the dealiasing results.

## Changelog:

Version 1.2.5:
- New box check using a fast and robust striding window algorithm, up to 10x faster for that function call.
Version 1.2.0:
- Multiprocessing using `unravel_3D_pyart_multiproc`
- Various optimization and speed up.

## Installation

The easiest method for installing UNRAVEL is to use pip:

```pip install unravel```

## Dependencies

Mandatory dependencies:
- [numpy][1]
- [numba][2]
- [Py-ART][3] 

[1]: http://www.scipy.org/
[2]: http://numba.pydata.org
[3]: https://github.com/ARM-DOE/pyart

# References:

Louf, V., Protat, A., Jackson, R. C., Collis, S. M., & Helmus, J. (2020). UNRAVEL: A Robust Modular Velocity Dealiasing Technique For Doppler Radar. Journal of Atmospheric and Oceanic Technology, 4(1), 741–758. [https://doi.org/10.1175/jtech-d-19-0020.1]
