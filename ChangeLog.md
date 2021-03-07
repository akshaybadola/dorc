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

## [2021-02-04 Thu 12:23]
- Added templated operationId generation in spec generation.
- There's only a single version now.
- Created `setup.py` and checked install.
- Added `dorc/__main__.py` and removed `dorc/spec/__main__.py`
- Added Readme
- Added custom parsers to `docs/conf.py`. Now we can parse `Requests,Responses`
  etc. sections.
- Fixed cycling dependencies in `dorc/__init__.py`
- Separated port forwarding stuff from `dorc/daemon/__init__.py` and some fixes
  to the file.
- Minor changes to `if_run.py` and it's now part of the package
- `config` is now entirely json and can be updated with multiple `overrides` in
  `dorc.interfaces.FlaskInterface`
- Added `_write_overrides`, `_get_overrides` and fixed some functions in
  `dorc.interfaces.FlaskInterface`
- fixed yaml references in spec generation. See `dorc.spec.fix_yaml_references`
- Added `state_var` to `dorc.trainer.Trainer`. State will be handled in a more
  streamlined manner from now.
- Dumping of `dorc.trainer.model.Model` as json
- Added various tests and `pytest` mark `todo`.

## [2021-02-16 Tue 16:31]
- Added [Flask-Pydantic](https://github.com/bauerji/flask_pydantic) to requirements.
- Added `@validate` to a couple of daemon routes.

## [2021-02-18 Thu 08:57]
- Fixed some `spec` issues.
- Separated `parser.py` from `spec.__init__.py`
- Fixed a lot of daemon and trainer bugs.
- Interface runner `if_run.py` accepts `gmods_dir` and `gdata_dir` as args now.
- Enhancements to `dorc.mods.Module`
- Defined `TrainerState` schema and integrated with daemon.
- Fixed trainer load, save and resume functions.
- Changed the tests to run all of them

## [2021-02-20 Sat 03:16]
- Fixed `Trainer.set_model` and `Trainer.active_models` and added tests.
- New docs and some adjustments.
- Added parametrized fixtures to tests.

## [2021-02-22 Mon 05:14]
- Changed docs theme to (customized)
  [sphinx_rtd_theme](https://github.com/readthedocs/sphinx_rtd_theme)
- Added new docs
- Deleted `dorc/trainer/_checks.py`.
- Renamed `dorc/trainer/hooks.py` -> `dorc/trainer/funcs.py`
- Added `run_quick.sh` and `run_threaded.sh` for running tests.
- Fixed issues related to `trainer.funcs` and `Trainer.funcs`
- Added some tests for the above also.

## [2021-02-24 Wed 16:20]
- Fixed model loading and unloading.
- Fixed some device allocation though conflicts are still not handled.
- Fixed `Trainer._init_update_funcs`
- Parametrized a lot of tests and some new marks.

## [2021-02-27 Sat 18:38]
- Added and fixed python file config.
- config writing and checks are in `daemon.Daemon` now. Interface only reads
  it. Functions `Daemon._write_python_config` and `Daemon._write_json_config`
  handle that.
- classmethod `FlaskInterface.read_config` and instance method
  `FlaskInterface._read_config` now read configs.
- Fixed a bug in `TranslationLayer.patch_config`.
- `dorc.mods.Modules.add_config` is now a classmethod.
- Changes to tests accordingly and a new `_setup_py.py` file for testing python
  config.

## [2021-02-28 Sun 05:34]
- Fixed `Daemon._create_trainer` and `Daemon.create_session` when trainer state
  is missing.
- Randomized test_dir generation for trainer tests.
- Timeout for `test_trainer_metrics.py`.
- Changed workflows to use bash scripts instead. Fixed `test_and_validate_spec.yml`.

## [2021-03-01 Mon 02:35]
- Added `Daemon.url`.
- Daemon starts up before scanning for sessions is finished now.
- There's an intermediate state for each session `scan_state` which keeps track
  of the session scanning during startup.
- Added debug route `_get_param` to `Daemon`.
- Fixed issues with gpu reservation in daemon and interface.
- Fixed start trainer without loading models.

## [2021-03-07 Sun 17:11]
- Separated `spec` as a separate package.
  See [flask-docspec](https://github.com/akshaybadola/flask-docspec)
- Some minor refactorings of `Daemon` responses.
