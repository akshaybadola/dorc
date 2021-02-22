import pytest
import torch
from dorc.device import all_devices, useable_devices
from dorc.trainer import Trainer
from util import Net, Net2, SubTrainer, ClassificationStepTwoModels, identity


@pytest.mark.quick
def test_trainer_init_models_one_model_no_gpus_and_no_gpus_in_params_inits_correctly(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net": {"model": Net,
                                      "optimizer": "Adam",
                                      "params": {},
                                      "gpus": []}}
    trainer = SubTrainer(False, **params)
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert "net" in trainer._training_steps["train"][0].models
    if trainer._model_step_func:
        assert trainer.active_models == {}
        trainer.set_model({"net", "net"})
        assert trainer._models["net"].gpus == []
        assert trainer._devices == {"net": []}
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())
    else:
        trainer._load_models(["net"])
        assert trainer._training_steps["train"][0].models['net'].loaded
        assert trainer.active_models == {"net": "net"}
        assert trainer._models["net"].gpus == []
        assert trainer._devices == {"net": []}
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())

@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_init_models_one_model_no_gpus_but_gpus_in_params_init_correctly(params_and_trainer):
    params, trainer = params_and_trainer
    params["trainer_params"]["gpus"] = [0]
    trainer = SubTrainer(False, **params)
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert "net" in trainer._training_steps["train"][0].models
    if trainer._model_step_func:
        assert trainer.active_models == {}
        trainer.set_model({"net": "net"})
        assert trainer.active_models == {"net": "net"}
        assert trainer._models["net"].gpus == []
        assert trainer._devices == {"net": []}
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())
    else:
        trainer._load_models(["net"])
        assert trainer._training_steps["train"][0].models['net'].loaded
        assert trainer.active_models == {"net": "net"}
        assert trainer._models["net"].gpus == []
        assert trainer._devices == {"net": []}
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.quick
@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_init_models_one_model_one_gpu(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net": {"model": Net,
                                      "optimizer": "Adam",
                                      "params": {},
                                      "gpus": [0]}}
    params["trainer_params"]["gpus"] = [0]
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.cuda = True
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert "net" in trainer._training_steps["train"][0].models
    if trainer._model_step_func:
        assert trainer.active_models == {}
        assert trainer.set_model({"net": "net"}).status
        trainer._training_steps["train"][0].models["net"].gpus == [0]
        assert trainer._devices["net"] == [0]
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())
    else:
        trainer._load_models(["net"])
        assert trainer._training_steps["train"][0].models['net'].loaded
        assert trainer.active_models == {"net": "net"}
        assert trainer._models["net"].gpus == [0]
        assert trainer._devices["net"] == [0]
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True], indirect=True)
def test_trainer_init_models_many_models_no_auto_no_parallel_gpus_sufficient(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": True,
                                        "gpus": [0]},
                              "net_2": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": True,
                                        "gpus": [1]}}
    params["trainer_params"]["gpus"] = [0, 1]
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.cuda = True
    trainer.config.update_functions.function = ClassificationStepTwoModels
    trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]}
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert trainer.models_available == ["net_1", "net_2"]
    trainer._training_steps["train"][0].models["net_1"].gpus == [0]
    trainer._training_steps["train"][0].models["net_2"].gpus == [1]
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True], indirect=True)
def test_trainer_init_models_many_models_no_auto_no_parallel_gpus_deficient(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": True,
                                        "gpus": [0, 1]},
                              "net_2": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": True,
                                        "gpus": [2, 3]}}
    params["trainer_params"]["gpus"] = [0, 1]
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.cuda = True
    trainer.config.update_functions.function = ClassificationStepTwoModels
    trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]}
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert trainer.models_available == ["net_1", "net_2"]
    assert "net_1" in trainer._models
    assert "net_2" in trainer._models
    assert trainer._models["net_1"].gpus == [0, 1]
    assert trainer._models["net_2"].gpus == []
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True], indirect=True)
def test_trainer_init_models_many_models_no_auto_no_parallel_gpus_conflict(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": True,
                                        "gpus": [0, 1]},
                              "net_2": {"model": Net,
                                        "optimizer": "Adam",
                                        "loaded": True,
                                        "params": {},
                                        "gpus": [0]}}
    params["trainer_params"]["gpus"] = [0, 1]
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.cuda = True
    trainer.config.update_functions.function = ClassificationStepTwoModels
    trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]}
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert "net_1" in trainer._models
    assert "net_2" in trainer._models
    assert trainer._models["net_1"].gpus == [0, 1]
    assert trainer._models["net_2"].gpus == []
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without at least 2 GPUs.")
@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True], indirect=True)
def test_trainer_init_models_many_models_two_gpus_auto_no_parallel(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": True,
                                        "gpus": "auto"},
                              "net_2": {"model": Net2,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": True,
                                        "gpus": "auto"}}
    params["trainer_params"]["gpus"] = [0, 1]
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.cuda = True
    trainer.config.update_functions.function = ClassificationStepTwoModels
    trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]}
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert "net_1" in trainer._models
    assert "net_2" in trainer._models
    assert trainer._models["net_1"].gpus == [1]
    assert trainer._models["net_2"].gpus == [0]
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_set_model_active_no_gpus(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": True,
                                        "gpus": "auto"},
                              "net_2": {"model": Net2,
                                        "optimizer": "Adam",
                                        "loaded": True,
                                        "params": {},
                                        "gpus": "auto"}}
    trainer = SubTrainer(False, **params)
    trainer.config.update_functions.function = ClassificationStepTwoModels
    trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]}
    trainer._init_all()
    if trainer._model_step_func:
        assert trainer.models_available == ["net_1", "net_2"]
        assert trainer.active_models == {}
    else:
        assert trainer.models_available == ["net_1", "net_2", "net"]
        assert trainer.active_models == {"net": "net"}
    trainer.set_model({"net": "net_1"})
    assert trainer.active_models == {"net": "net_1"}
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_set_model_active_load_unload_correctly(params_and_trainer):
    pass


# @pytest.mark.quick
# def test_trainer_set_model_active_alt(trainer):
#     trainer._init_all()
#     trainer._training_steps["test"][0](trainer.train_loader.dataset[0])
#     pytest.set_trace()


# TODO
@pytest.mark.quick
def test_trainer_add_model(params_and_trainer):
    params, trainer = params_and_trainer
    pass


# TODO
@pytest.mark.quick
def test_trainer_load_and_unload_model(params_and_trainer):
    params, trainer = params_and_trainer
    pass


# TODO
@pytest.mark.quick
def test_trainer_load_models_state(params_and_trainer):
    params, trainer = params_and_trainer
    pass

# TODO
@pytest.mark.quick
def test_trainer_get_new_models(params_and_trainer):
    params, trainer = params_and_trainer
    pass
