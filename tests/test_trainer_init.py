import pytest
import sys
sys.path.append("..")
from trainer.trainer import Trainer, config
from trainer.trainer.model import Model
from trainer.autoloads import ClassificationStep


# TODO
@pytest.mark.quick
def test_trainer_init_basic_trainer_data_params(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()


# TODO
@pytest.mark.quick
def test_trainer_init_bad_data_params(params_and_trainer):
    # inject bad data
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()


# TODO
@pytest.mark.quick
def test_trainer_init_custom_loader_good_params(params_and_trainer):
    # inject custom loader  data
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()


# TODO
@pytest.mark.quick
def test_trainer_init_custom_loader_bad_params(params_and_trainer):
    # inject custom loader with bad params
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()


# TODO
@pytest.mark.quick
def test_trainer_init_data_check_raw(params_and_trainer):
    pass


@pytest.mark.quick
def test_trainer_init_update_functions(params_and_trainer, basic_config):
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert trainer._model_step_func is None
    assert (isinstance(x, Model) for x in trainer.update_functions.train.models.values())
    assert (callable(x) for x in trainer.update_functions.train.criteria.values())
    assert (isinstance(x, Model) for x in trainer.update_functions.test.models.values())
    assert (callable(x) for x in trainer.update_functions.test.criteria.values())
    params["update_functions"] = {"function": ClassificationStep,
                                  "params": {"models": ["net"],
                                             "criteria_map": {"net": "criterion_ce_loss"},
                                             "checks": {"net": lambda x: True},
                                             "logs": ["loss"]}}
    trainer = Trainer(**params)
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert trainer._model_step_func is not None
    assert (isinstance(x, Model) for x in trainer._model_step_func.models.values())
    assert (callable(x) for x in trainer._model_step_func.criteria.values())


@pytest.mark.quick
def test_trainer_init_metrics(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    trainer._init_metrics()
    assert all(x in trainer.metrics for x in trainer._training_steps)
    assert all(y in trainer.metrics[x] for x in trainer._training_steps
               for y in trainer._training_steps[x][-1])
    params["update_functions"] = {"function": ClassificationStep,
                                  "params": {"models": ["net"],
                                             "criteria_map": {"net": "criterion_ce_loss"},
                                             "checks": {"net": lambda x: True},
                                             "logs": ["loss"]}}
    trainer = Trainer(**params)
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    trainer._init_metrics()
    assert all(x in trainer.metrics for x in trainer._training_steps)
    assert all(y in trainer.metrics[x] for x in trainer._training_steps
               for y in trainer._training_steps[x][-1])


@pytest.mark.quick
def test_trainer_init_metrics_extra_metrics(params_and_trainer):
    params, trainer = params_and_trainer

    def dummy_func(loss, epoch):
        return (epoch, loss ** 2)

    params["extra_metrics"] = {'awesome_metric':
                               {"steps": ["train", "test"],
                                "function": dummy_func,
                                "inputs": ["epoch", "loss"],
                                "when": "EPOCH"}}
    trainer = Trainer(**params)
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    trainer._init_metrics()
    assert "awesome_metric" in trainer.metrics["train"]
    params["update_functions"] = {"function": ClassificationStep,
                                  "params": {"models": ["net"],
                                             "criteria_map": {"net": "criterion_ce_loss"},
                                             "checks": {"net": lambda x: True},
                                             "logs": ["loss"]}}
    params["extra_metrics"] = {'awesome_metric':
                               {"steps": ["train", "test"],
                                "function": dummy_func,
                                "inputs": ["epoch", "loss"],
                                "when": "EPOCH"}}
    trainer = Trainer(**params)
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    trainer._init_metrics()
    assert "awesome_metric" in trainer.metrics["train"]


# @pytest.mark.quick
# def test_trainer_init_task_runners(params_and_trainer):
#     params, trainer = params_and_trainer
#     trainer._init_device()
#     trainer._init_models()
#     trainer._init_data_and_dataloaders()
#     trainer._init_update_funcs()
#     trainer._init_training_steps()
#     trainer._init_metrics()
#     trainer._init_task_runners()
