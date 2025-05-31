# Changelog

## [1.3.4](https://github.com/vlouf/dealias/releases/tag/v1.3.4) - 2025-02-25
### Added
- Support for gatefilter with PyODIM.

### Changed
- Updated CI-testing URLs.

## [1.3.3](https://github.com/vlouf/dealias/releases/tag/v1.3.3) - 2024-08-23
### Added
- Compatibility with NumPy v2.0.

### Fixed
- Continuous integration environment by @vlouf in #18.
- Stop at stage by @vlouf in #19.

## [1.3.2](https://github.com/vlouf/dealias/releases/tag/v1.3.2) - 2024-01-11
### Fixed
- Fixed bug related to masked arrays and Numba.

## [1.3.0](https://github.com/vlouf/dealias/releases/tag/v0.0.1-beta) - 2024-01-11
### Added
- Implemented `write_odim_slice()`.
- Introduced `unravel_3D_pyodim_slice()` extraction.
- Added `rename_old_data()` to handle data renaming instead of bailing.

### Changed
- Moved ODIM helper routines to `pyodim`.
- Updated PyODIM interface.
- Renamed `readwrite` to `read_write`.

## [1.2.5] - 2021-03-01
### Added
- New box check using a fast and robust striding window algorithm, up to 10x faster for that function call.

## [1.2.0](https://github.com/vlouf/dealias/releases/tag/v1.2.0) - 2020-12-15
### Added
- Implemented multiprocessing support via `unravel_3D_pyart_multiproc`.

### Fixed
- Various optimizations and performance improvements.
