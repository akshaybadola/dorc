import pytest
import os
import shutil
import time
from datetime import datetime
import re
import sys
from _setup_local import config
from dorc.device import all_devices, useable_devices
from dorc.trainer import Trainer
from dorc.util import diff_as_sets
from util import SubTrainer


@pytest.mark.todo
def test_props_all_tests_present(params_and_trainer):
    params, trainer = params_and_trainer
    props = trainer.props
    reg = re.compile(r'def ([a-z_]+).*')
    with open(__file__) as f:
        lines = f.read().split("\n")
        lines = set([reg.sub(r'\1', x).replace("test_trainer_property_", "")
                     for x in lines
                     if x.startswith("def") and "all_test_present" not in x])
    diff_a = diff_as_sets(props, lines)
    diff_b = diff_as_sets(lines, props)
    assert not diff_a and not diff_b


@pytest.mark.todo
def test_trainer_property_version(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_current_state(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_loop_type(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_logger(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_logfile(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_saves(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_gpus(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_system_info(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_devices(params_and_trainer):
    pass



@pytest.mark.todo
def test_trainer_property_models(params_and_trainer):
    params, trainer = params_and_trainer
    assert isinstance(trainer.trainer.models, list)
    assert all([isinstance(x, list) for x in trainer.models])



@pytest.mark.todo
def test_trainer_property_train_step_func(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_val_step_func(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_test_step_func(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_active_model(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_epoch(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_max_epochs(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_iterations(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_max_iterations(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_controls(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_methods(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_all_props(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_extras(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_running(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_current_run(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_paused(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_adhoc_paused(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_adhoc_aborted(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_userfunc_paused(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_userfunc_aborted(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_best_save(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_trainer_params(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_model_params(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_dataloader_params(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_data(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_updatable_params(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_all_params(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_train_losses(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_progress(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_user_funcs(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_metrics(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_val_samples(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_all_post_epoch_hooks(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_post_epoch_hooks_to_run(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_items_to_log_dict(params_and_trainer):
    pass


@pytest.mark.todo
def test_trainer_property_log_levels(params_and_trainer):
    pass
