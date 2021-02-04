# ChangeLog for Deep learning ORChestrator (DORC)

## [2021-02-02 Tue 15:42]
- Fixed return type annotation of `dorc.device.gpu_ranking`
- Fixed bug in `spec.exec_and_return` where modules weren't loaded while doing
  `exec` (added `globals()` to modules)
- Removed `self` reference from hooks in `dorc.task.LoopTaskWithHooks`
- Added `init_hooks` and `init_hooks_with_args` to `dorc.trainer`
- Added `hooks` and `hooks_with_args` to `dorc/trainer` and `dorc.epoch.Epoch`
- Added `_init_data_helper` to `dorc.trainer`. Fixes data initialization.
- Added `TrainingType` and how it's handled in `dorc.trainer`
- Changed how the step functions are determined in `dorc.trainer`. They're now
  in `_training_steps`.
- Hooks can now be added and removed from `dorc.epoch.Epoch`. For `dorc.trainer`
  they're loaded from `dorc/trainer/hooks.py`
- Added `trainer` to `tests/fixtures.py`.
- Fixed tests up to `test_trainer_metrics.py`.

## [2021-02-03 Wed 14:49]
- Fixed gpu tests.

## [2021-02-03 Wed 22:54]
- Added more docs to interfaces.
- Consolidated methods and extras in the `dorc.interfaces.FlaskInterface`
- Added `spec` to pytest markers and separated spec tests.
- Fixed some tests in `test_spec_schema`.
- Added `test_spec_docstring.py`

## [2021-02-04 Thu 12:23]
- Added spec generation and validation and fixes for same.
- Github workflow for spec generation and validation.
- Fixed return values of functions.
  - Removed Tuple and added where there was no return annotation.
