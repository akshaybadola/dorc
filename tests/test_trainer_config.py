import pytest
from pydantic import ValidationError
import sys
sys.path.append("..")
from dorc.autoloads import ClassificationStep
from dorc.trainer import Trainer, config


@pytest.mark.todo
@pytest.mark.quick
def test_config_should_not_coerce_types(setup_and_net, basic_config):
    pass


@pytest.mark.quick
def test_config_should_have_same_params_as_config(setup_and_net, basic_config):
    _config, _ = setup_and_net
    assert set(basic_config.__dict__.keys()) == set({*_config.keys(), "data_dir",
                                                     "global_modules_dir",
                                                     "global_datasets_dir"})


@pytest.mark.quick
def test_config_should_have_same_params_as_trainer(basic_config):
    import inspect
    sig = inspect.signature(Trainer)
    assert set(basic_config.__dict__.keys()) == set(sig.parameters.keys())


@pytest.mark.quick
def test_config_empty_model_params_should_raise_error(basic_config):
    default_config = basic_config
    test_config = config.Config(**default_config.__dict__)
    with pytest.raises(ValidationError):
        bleh = {**default_config.__dict__}
        bleh["model_params"] = {}
        test_config = config.Config(**bleh)


@pytest.mark.quick
def test_config_model_step_params(basic_config):
    model_step_params = {"function": "ClassificationStep",
                         "params": {"models": ["net"],
                                    "criteria_map": {"net": "criterion_ce_loss"},
                                    "checks": {"net": lambda x: True},
                                    "logs": ["loss"]}}
    update_func_params = config.UpdateFunctionsParams(**model_step_params["params"])
    update_functions = config.UpdateFunctions(function=ClassificationStep,
                                              params=update_func_params)
    config_dict = basic_config.__dict__.copy()
    config_dict["update_functions"] = update_functions
    test_config = config.Config(**config_dict)
    with pytest.raises(ValidationError):
        test_config.update_functions = "bleh"
    with pytest.raises(ValidationError):
        config_dict["update_functions"].train = "bleh"
    with pytest.raises(ValidationError):
        config_dict["update_functions"].train = lambda x: x


@pytest.mark.quick
def test_trainer_trainer_params_resume(basic_config):
    from pathlib import Path
    basic_config.trainer_params.resume = True
    basic_config.trainer_params.resume = False
    with pytest.raises(ValidationError):
        basic_config.trainer_params.resume_dict = False
    with pytest.raises(ValidationError):
        basic_config.trainer_params.resume_best = False
    basic_config.trainer_params.init_weights = Path.home()
    with pytest.raises(ValidationError):
        basic_config.trainer_params.resume = True
        basic_config.trainer_params.init_weights = Path.home()
    basic_config.trainer_params.init_weights = None
    basic_config.trainer_params.resume = True
    basic_config.trainer_params.resume_dict = Path.home()
