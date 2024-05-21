[![DOI](https://zenodo.org/badge/119326382.svg)](https://zenodo.org/badge/latestdoi/119326382)

# UNRAVEL

UNRAVEL (UNfold RAdar VELocity) is an open-source modular Doppler velocity dealiasing algorithm for weather radars. This algorithm does not require external reference velocity data, making it highly versatile and easily applicable across various contexts. UNRAVEL consists of eleven core modules and two dealiasing strategies, enabling iterative processing. It starts with the strictest continuity tests in azimuth and range, progressively relaxing the parameters to include more results from a growing number of reference points. Additionally, UNRAVEL includes modules for performing 3D continuity checks. This modular design allows for the expansion of dealiasing strategies to optimize results further.

## Changelog:

Version 1.2.5:
- New box check using a fast and robust striding window algorithm, up to 10x faster for that function call.
Version 1.2.0:
- Multiprocessing using `unravel_3D_pyart_multiproc`
- Various optimization and speed up.

## Installation

UNRAVEL is available on [PyPI](https://pypi.org/project/unravel/). The easiest method for installing UNRAVEL is to use pip:

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
