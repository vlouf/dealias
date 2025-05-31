[![DOI](https://zenodo.org/badge/119326382.svg)](https://zenodo.org/badge/latestdoi/119326382)

# UNRAVEL

**UNfold RAdar VELocity (UNRAVEL)** is an open-source modular Doppler velocity dealiasing algorithm for weather radars. Designed for flexibility, UNRAVEL does not require external reference velocity data, making it highly adaptable across various contexts.

## Features
- **Modular Design:** Consists of **eleven core modules** and **two dealiasing strategies** for iterative processing.
- **Adaptive Dealiasing:** Starts with strict continuity tests in azimuth and range, then progressively relaxes parameters to include more reference points.
- **3D Continuity Checks:** Modules for multi-dimensional dealiasing enhance accuracy.
- **Expandable Framework:** Allows for additional strategies to optimize results further.

## Requirements
To use UNRAVEL, install the following dependencies:
```sh
pip install h5py numba numpy xarray pyodim dask pyart
```

## Installation

UNRAVEL is available on [PyPI](https://pypi.org/project/unravel/). The easiest method for installing UNRAVEL is to use pip:

```pip install unravel```

## Dependencies

## References

If you use `UNRAVEL` in your research, please cite the following paper:

**Louf, V., Protat, A., Jackson, R. C., Collis, S. M., & Helmus, J.** (2020). *UNRAVEL: A Robust Modular Velocity Dealiasing Technique For Doppler Radar*. Journal of Atmospheric and Oceanic Technology, 4(1), 741–758. (10.1175/jtech-d-19-0020.1)[https://doi.org/10.1175/jtech-d-19-0020.1]

```bibtex
@article {Louf2020,
      author = "Valentin Louf and Alain Protat and Robert C. Jackson and Scott M. Collis and Jonathan Helmus",
      title = "UNRAVEL: A Robust Modular Velocity Dealiasing Technique for Doppler Radar",
      journal = "Journal of Atmospheric and Oceanic Technology",
      year = "2020",
      publisher = "American Meteorological Society",      
      volume = "37",
      number = "5",
      doi = "10.1175/JTECH-D-19-0020.1",
      pages= "741 - 758",
      url = "https://journals.ametsoc.org/view/journals/atot/37/5/jtech-d-19-0020.1.xml"
}
```