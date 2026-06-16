# Changelog

## [1.5.0](https://github.com/vlouf/dealias/releases/tag/v1.5.0) - 2026-06-16
### Added
- Convolution-based (separable cumulative-sum) fast paths for all three expensive stages: box check, inter-sweep 3-D unfolding, and closest-reference correction. These replace the previous per-gate window gather with O(rays × gates) operations, giving 5–20× speedups on typical precipitation volumes.
- `unravel.warmup()` public function that triggers numba JIT compilation of every compiled function in the main process. On fork-based HPC systems (Linux) worker processes inherit the already-compiled code at zero cost; on spawn-based systems the disk cache written by `warmup()` is reused by each worker, avoiding per-worker recompilation.
- Early-exit fractional threshold in `check_completed()`: a sweep is considered done when fewer than 1% of valid gates remain unprocessed (previously the threshold was an absolute count of 10 gates, which never fired on sparse or clear-air scans).
- Progress-based early break in the window loops of `dealiasing_process_2D` and `dealias_long_range`: if a window iteration processes zero new gates, larger windows are skipped immediately.

### Fixed
- `cache=True` added to the `@jit` decorators on `cfg.log`, `filtering.unfold`, and `filtering.filter_data`, which were the only JIT-compiled functions not persisting their compiled code to disk.
- `box_check` dispatcher: `if not window_azimuth:` replaced with `if window_azimuth is None:` to avoid treating an explicit `window_azimuth=0` as absent.

## [1.4.1](https://github.com/vlouf/dealias/releases/tag/v1.4.1) - 2025-12-11
### Added
- Users can now pass either a file path or pre-loaded pyodim datasets to `unravel_3D_pyodim()`. This enables preprocessing workflows (e.g., dual-PRF correction) before dealiasing.

### Fixed
- Original field modification bug: When using the `condition` parameter, the function now creates an independent masked copy instead of modifying the original velocity field (e.g., VRADH).

### Changed
- Updated unit tests to cover pyodim dealiasing functions with comprehensive test cases.

## [1.4.0](https://github.com/vlouf/dealias/releases/tag/v1.4.0) - 2025-11-07
### Changed
- Migrated from `setup.py` to modern `pyproject.toml` build system
- Improved type hints throughout the codebase (#21)
- Updated CI/CD dependencies.
  
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
