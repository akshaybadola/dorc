import pytest
import os
import json
import torch
import torchvision
import pydantic
from dorc.trainer import Trainer, config
from dorc.trainer.model import Model
from dorc.autoloads import ClassificationStep


@pytest.mark.quick
def test_trainer_init_data_bad_params(params_and_trainer):
    # inject custom loader  data
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    with pytest.raises(pydantic.ValidationError):
        trainer.data_params.train = []
    with pytest.raises(AttributeError):
        trainer.data_params.train = {"function": "meow"}
        trainer._init_data_and_dataloaders()


@pytest.mark.quick
def test_trainer_init_data_good_params(params_and_trainer):
    # inject custom loader  data
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    assert isinstance(iter(trainer.train_loader).__next__()[0], torch.Tensor)
    assert isinstance(trainer.data_params.train, torchvision.datasets.mnist.MNIST)


@pytest.mark.quick
def test_trainer_init_data_good_params_dict(trainer_json_config):
    # inject custom loader  data
    _, trainer = trainer_json_config
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    assert isinstance(trainer.data_params.train, dict)
    assert isinstance(iter(trainer.train_loader).__next__()[0], torch.Tensor)


@pytest.mark.quick
@pytest.mark.todo
def test_trainer_init_custom_loader_good_params(params_and_trainer):
    # inject custom loader with bad params
    params, trainer = params_and_trainer
    pass

@pytest.mark.quick
@pytest.mark.todo
def test_trainer_init_custom_loader_bad_params(params_and_trainer):
    # inject custom loader with bad params
    params, trainer = params_and_trainer
    pass


@pytest.mark.todo
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
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_init_update_functions_dict(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert "net" in trainer._training_steps["train"][0].models
    if trainer._model_step_func:
        assert trainer._training_steps["train"][0].models['net'] is None
        assert trainer.active_models == {}
    else:
        assert not trainer._training_steps["train"][0].models['net'].loaded
        assert trainer.active_models == {}


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


@pytest.mark.quick
def test_trainer_init_and_dump_state(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    trainer._init_metrics()
    trainer._init_state_vars()
    trainer._init_task_runners()
    trainer._init_modules()
    trainer._init_funcs()
    trainer._init_hooks()
    retval = trainer._dump_state()
    assert retval.status
    state_file = os.path.join(trainer._data_dir, "session_state")
    assert os.path.exists(state_file)
    with open(state_file) as f:
        state = json.load(f)
    from dorc.trainer import TrainerState
    TrainerState.parse_obj(state)


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_init_state_vars(params_and_trainer):
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
    trainer._init_state_vars()
    # TODO: assert stuff about the state


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_init_resume_or_init(params_and_trainer):
    params, trainer = params_and_trainer
    # import pytest; pytest.set_trace()
