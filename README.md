[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20711799.svg)](https://doi.org/10.5281/zenodo.20711799)

# UNRAVEL

**UNfold RAdar VELocity (UNRAVEL)** is an open-source modular Doppler velocity dealiasing algorithm for weather radars. Designed for flexibility, UNRAVEL does not require external reference velocity data, making it highly adaptable across various contexts.

## Features
- **Modular Design:** Consists of **eleven core modules** and **two dealiasing strategies** for iterative processing.
- **Adaptive Dealiasing:** Starts with strict continuity tests in azimuth and range, then progressively relaxes parameters to include more reference points.
- **3D Continuity Checks:** Modules for multi-dimensional dealiasing enhance accuracy.
- **Expandable Framework:** Allows for additional strategies to optimize results further.

## Installation

UNRAVEL requires Python 3.9 or newer and is available on [PyPI](https://pypi.org/project/unravel/):

```sh
pip install unravel
```

pip pulls in the dependencies automatically: `numba`, `numpy`, `xarray`, `dask`,
[`pyodim`](https://github.com/vlouf/pyodim) (>= 0.7) and
[`arm_pyart`](https://github.com/ARM-DOE/pyart).

## Usage

There is one entry point per radar data model. Both dealias a full volume and take
a `strategy` (`"default"` or `"long_range"`, for long-range scans with few gates
per beam) and an `alpha` threshold (0.6 by default; lower is stricter).

With [Py-ART](https://github.com/ARM-DOE/pyart), which returns the dealiased field
as an array:

```python
import pyart
import unravel

radar = pyart.io.read("radar_volume.nc")
velocity = unravel.unravel_3D_pyart(radar, velname="VEL", dbzname="DBZ")
radar.add_field_like("VEL", "dealiased_velocity", velocity)
```

With [pyodim](https://github.com/vlouf/pyodim), for ODIM H5 files, which returns one
xarray dataset per sweep:

```python
import unravel

sweeps = unravel.unravel_3D_pyodim(
    "radar_volume.pvol.h5",
    vel_name="VRADH",
    output_vel_name="unraveled_velocity",
    strategy="long_range",
)
```

`unravel_3D_pyodim` also accepts a list of pre-loaded pyodim datasets in place of a
file path, so corrections such as dual-PRF unfolding can be applied first. The
datasets passed in are left untouched; the dealiased sweeps come back as a new list.

Before spawning workers (dask, multiprocessing), call `unravel.warmup()` once in the
main process. It triggers numba's JIT compilation so that workers inherit the
compiled code (fork) or reuse its on-disk cache (spawn), instead of each paying the
compilation cost:

```python
unravel.warmup()
```

To drive the modules yourself rather than running a whole strategy, use the
`Dealias` class on a single sweep:

```python
from unravel import Dealias

dealias = Dealias(r, azimuth, elevation, velocity, nyquist_velocity, alpha=0.6)
dealias.initialize()
dealias.correct_range()
dealias.correct_clock()
dealiased_velocity, flag = dealias.dealias_vel, dealias.flag
```

`flag` marks each gate: `-3` no data, `0` unprocessed, `1` processed and unchanged,
`2` dealiased.

## References

If you use `UNRAVEL` in your research, please cite the following paper:

**Louf, V., Protat, A., Jackson, R. C., Collis, S. M., & Helmus, J.** (2020). *UNRAVEL: A Robust Modular Velocity Dealiasing Technique For Doppler Radar*. Journal of Atmospheric and Oceanic Technology, 37(5), 741–758. [10.1175/JTECH-D-19-0020.1](https://doi.org/10.1175/JTECH-D-19-0020.1)

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

## Star History

<a href="https://www.star-history.com/?repos=vlouf%2Fdealias&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=vlouf/dealias&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=vlouf/dealias&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=vlouf/dealias&type=date&legend=top-left" />
 </picture>
</a>