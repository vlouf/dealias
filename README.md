# UNRAVEL

UNRAVEL (UNfold RAdar VELocity) is an open-source modular Doppler velocity dealiasing algorithm for weather radars. UNRAVEL is an algorithm that does not need external reference velocity data, making it easily applicable. The proposed algorithm includes eleven core modules and two dealiasing strategies. UNRAVEL is an iterative algorithm. The goal is to build the dealiasing results starting with the strictest possible continuity tests in azimuth and range and, after each step, relaxing the parameters to include more results from a progressively growing number of reference points. UNRAVEL also has modules that perform 3D continuity checks. Thanks to this modular design, the number of dealiasing strategies can be expanded in order to optimise the dealiasing results.

## Installation

The easiest method for installing UNRAVEL is to use pip:

```pip install unravel```

## Dependencies

Mandatory dependencies:
- [numpy][1]
- [numba][2]

Even thought it does not directly requires [Py-ART][3] to run, UNRAVEL can use [Py-ART][3] data class as input.

[1]: http://www.scipy.org/
[2]: http://numba.pydata.org
[3]: https://github.com/ARM-DOE/pyart

# References:

A paper describing UNRAVEL and its performances is currently under review in the Journal of Atmospheric and Oceanic Technology.