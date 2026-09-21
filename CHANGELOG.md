# Changelog

## [Unreleased]
### Fixed
- `unravel_3D_pyart` never passed its `alpha` to the 2-D chain, so the chain always ran at its own default of 0.6 while the driver advertised 0.8. Since `alpha` is a named parameter of the driver it could not reach the chain through `**kwargs` either, making the 2-D tolerance unsettable from this entry point. `alpha` is now used consistently by the 2-D chain, the inter-sweep continuity and the box check in both Py-ART drivers.
- Both Py-ART drivers handed the *pre*-3-D flag array to the next tilt while handing on the post-3-D velocity, so gates that only became trustworthy through vertical continuity were excluded from the reference set of the tilt above.
- Both Py-ART drivers seeded the 3-D continuity from sweep 0 and marched strictly upward. They now seed from the highest-Nyquist sweep among the lower half of the tilts and propagate downward then upward, matching `unravel_3D_pyodim`. This matters wherever the PRF changes with elevation, since the lowest tilt is then not the one that folds least.
- The two Py-ART drivers applied the box check at different points relative to the reference handoff (serial before, multiproc deferred to a separate dask bag), and used different window sizes for both the inter-sweep continuity and the box check. They now share one code path and one window set, taken from `unravel_3D_pyodim`. The box check is no longer parallelised; `box_check_conv` has been the cumulative-sum implementation since 1.5.0, so the cost is small.
- `velocity[gatefilter.gate_excluded] = np.nan` wrote into the caller's radar object whenever `radar.fields[...]["data"]` was a plain ndarray rather than a MaskedArray, because `unmask_array` returns its argument unchanged in that case. Both drivers now copy first. Same class of bug as the one fixed in `Dealias` in 1.5.1.
- `unravel_3D_pyart` omitted `debug=debug` on the `long_range` branch, so `debug=True` with that strategy silently returned a list of `None`. `pointbreak` is also now indexed by sweep rather than by processing order.
- `find_reference_radials` compared beam *indices* on a circle of 360, so the search for the second reference radial was only correct on a sweep with exactly 360 rays. At 720 rays the diametrically opposite beam scored zero instead of the maximum and the second radial landed a quarter turn from the first; at 120 rays the scoring degenerated into "largest index difference", picking a nearly adjacent beam. The circle is now the number of beams.
- `unravel_3D_pyodim` raised `ValueError: attempt to get argmax of an empty sequence` on a single-sweep volume, because `nyquists[: len(elev_angles) // 2]` sliced to nothing.
- `unravel_3D_pyodim_slice` gated the 3-D step on `and ds_ref:`, relying on `xr.Dataset.__bool__`; an empty dataset would silently skip inter-sweep continuity. Now an explicit `is not None`, evaluated before `stage_check` so the stage counter is not spent on the seed sweep.
- Removed the stale `# TODO: pass alpha` in `unravel_3D_pyodim_slice`: alpha is passed on the following line.

### Added
- `tests/test_pyart_drivers.py`, including an equivalence test asserting that `unravel_3D_pyart` and `unravel_3D_pyart_multiproc` return the same field.
- `_sweep_processing_order()` and `_unfold_3d_pyart_slice()`, shared by both Py-ART drivers so they cannot drift apart again. `unravel_3D_pyodim` now uses `_sweep_processing_order()` too, over the elevation-sorted sweeps, which replaces its three separate loops.
- `tests/test_find_reference.py` and `tests/test_pyodim_driver.py`.

### Changed
- Documented, rather than removed, the tighter agreement test in `correct_clockwise` / `correct_counterclockwise` (`window_len <= 3`). It is unreachable from both shipped strategies, which iterate from a window of 6 upwards, but it is live for anyone driving `Dealias.correct_clock()` directly, whose default window is 3. Whether the strategies should use it is a question for measurement, not a cleanup.

### Notes
- These change the output of both Py-ART drivers. On a synthetic 6-sweep volume with a varying PRF the two drivers disagreed on 12.66% of gates before (1.31% with `do_3d=False`) and are now bit-identical in both configurations. `unravel_3D_pyodim` is untouched and verified bit-identical on a 13-sweep ODIM volume.
- The reference-radial fix changes which second radial is chosen on any sweep that is not 360 rays. No end-to-end change was observed on either synthetic volume used here (the second radial only seeds a handful of gates, and those cases are easy), but the selection itself demonstrably moves: on a 120-ray sweep the old scoring ranked a beam 90 degrees away top, the new one picks the one 180 degrees away.
- `unravel_3D_pyodim` output is unchanged on a 13-sweep ODIM volume, which is 360-ray, so none of these fixes bite on it.

## [1.6.0](https://github.com/vlouf/dealias/releases/tag/v1.6.0) - 2026-09-16
### Added
- `Dealias` is now exported from the package root (`from unravel import Dealias`), for callers driving the dealiasing modules themselves.
- `Dealias` takes `alpha_mad` as a constructor argument (previously a hard-coded attribute), and exposes `COMPLETION_THRESHOLD`, `FIRST_PASS_NYQUIST_FRACTION` and `MAX_LEASTSQUARE_ELEVATION` as class constants instead of literals buried in the methods.

### Changed
- **Requires `pyodim >= 0.7.0`.** pyodim 0.7 replaced `read_odim(lazy_load=...)` with `lazy`, and reads eagerly by default; the `.compute()` that followed `read_odim()` was a full copy of every sweep and has been removed.
- **Requires Python >= 3.9** (was 3.8).
- `unravel_3D_pyodim()` no longer mutates the list passed as `odim_input`. It returns a new list of new datasets and leaves the caller's list and sweeps untouched, matching the file-path branch. Code that relied on the in-place update and discarded the return value must now use the returned list.
- `unmask_array()` moved to `unravel.core` (it duplicated `Dealias._check_velocity`); it remains importable from `unravel.dealias`.
- Type hints: `Union[None, X]` replaced by `Optional[X]` throughout, and `Dealias.correct_box` is typed `Union[int, Tuple[int, int]]`, matching the bare int it already accepted.
- The eight `correct_*`/`check_*` methods of `Dealias` share a single `_apply()` helper rather than each repeating the alpha fallback and the velocity/flag write-back. Output is numerically identical.
- Test suite: live logging reports each phase with its duration plus a per-volume summary of coverage and unfolded fraction (`--log-cli-level=DEBUG` adds a per-sweep breakdown, `WARNING` silences it). Test bodies are no longer wrapped in `try/except` + `pytest.fail`, which reduced every failure to a single line with no traceback.

### Fixed
- **`Dealias` modified the caller's velocity array in place.** `filtering.filter_data()` unfolds the velocity in place and returns the same object, so `dealiasing_process_2D()` rewrote gates of the array it was given. The field is now copied on construction.
- **An all-NaN beam could be chosen as the reference radial.** `find_reference_radials()` averaged every beam before selecting, which emitted `RuntimeWarning: Mean of empty slice` for empty beams; when no beam held 10 valid gates the selection threshold became vacuous and `argmin` returned an empty beam. Beams are now selected before averaging, and the fallback picks the quietest beam that holds data.
- **`dealias_long_range()` reported the wrong `completed` stage.** An inverted condition set `completed = "closest"` when the sweep was *not* complete and left it empty when it was. Affects the `debug=True` return value and log line only; no velocity was changed.
- `Dealias` validated `alpha` and the velocity dimensions with `assert`, after the arrays had already been used, and `python -O` stripped the checks entirely. Validation now runs first and raises `ValueError` naming the offending value and shapes.
- `Dealias.correct_clock()` raised a bare `AttributeError` when called before `initialize()`; it now raises `RuntimeError` naming the missing step.

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
