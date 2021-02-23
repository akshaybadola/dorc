import pytest
import torch
from dorc.device import all_devices, useable_devices
from dorc.trainer import Trainer
from util import Net, Net2, SubTrainer, ClassificationStepTwoModels, identity


@pytest.mark.quick
def test_trainer_models_init_one_model_no_gpus_and_no_gpus_in_params_inits_correctly(params_and_trainer):
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
        trainer.set_model({"net": "net"})
        assert trainer._models["net"].gpus == []
        assert trainer._devices == {"net": []}
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())
    else:
        assert trainer.active_models == {}
        trainer.set_model({"net": "net"})
        assert trainer._training_steps["train"][0].models['net'].loaded
        assert trainer.active_models == {"net": "net"}
        assert trainer._models["net"].gpus == []
        assert trainer._devices == {"net": []}
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())

@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_models_init_one_model_no_gpus_but_gpus_in_params_init_correctly(params_and_trainer):
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
        trainer.model_params["net"].loaded = True
        trainer._init_models()
        trainer._init_update_funcs()
        trainer._init_training_steps()
        assert trainer._training_steps["train"][0].models['net'].loaded
        assert trainer.active_models == {"net": "net"}
        assert trainer._models["net"].gpus == []
        assert trainer._devices == {"net": []}
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.quick
@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_models_init_one_model_one_gpu(params_and_trainer):
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
        trainer.model_params["net"].loaded = True
        trainer._init_models()
        trainer._init_update_funcs()
        trainer._init_training_steps()
        assert trainer._training_steps["train"][0].models['net'].loaded
        assert trainer.active_models == {"net": "net"}
        assert trainer._models["net"].gpus == [0]
        assert trainer._devices["net"] == [0]
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True], indirect=True)
def test_trainer_models_init_many_models_no_auto_no_parallel_gpus_sufficient(params_and_trainer):
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
def test_trainer_models_init_many_models_no_auto_no_parallel_gpus_deficient(params_and_trainer):
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
def test_trainer_models_init_many_models_no_auto_no_parallel_gpus_conflict(params_and_trainer):
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
def test_trainer_models_init_many_models_two_gpus_auto_no_parallel(params_and_trainer):
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


@pytest.mark.todo
@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
@pytest.mark.parametrize("loaded", [True, False])
def test_trainer_models_set_model_active_no_gpus(params_and_trainer, loaded):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": loaded,
                                        "gpus": "auto"},
                              "net_2": {"model": Net2,
                                        "optimizer": "Adam",
                                        "loaded": loaded,
                                        "params": {},
                                        "gpus": "auto"}}
    trainer = SubTrainer(False, **params)
    if trainer.config.update_functions.function is not None:
        trainer.config.update_functions.function = ClassificationStepTwoModels
        trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                                  "criteria_map": {"net_1": "criterion_ce_loss",
                                                                   "net_2": "criterion_ce_loss"},
                                                  "checks": {"net_1": identity, "net_2": identity},
                                                  "logs": ["loss"]}
    trainer._init_all()
    if trainer._model_step_func:
        assert trainer.models_available == ["net_1", "net_2"]
        if loaded:
            assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        else:
            assert trainer.active_models == {}
    else:
        assert trainer.models_available == ["net_1", "net_2", "net"]
        if loaded:
            assert trainer.active_models == {"net": "net"}
        else:
            assert trainer.active_models == {}
    trainer.set_model({"net_1": "net_1"})
    assert trainer.active_models == {"net_1": "net_1"}


@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_models_no_gpus_load_correctly(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "gpus": "auto"},
                              "net_2": {"model": Net2,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "gpus": "auto"}}
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.gpus = [0, 1]
    trainer.trainer_params.cuda = True
    if trainer.config.update_functions.function is not None:
        trainer.config.update_functions.function = ClassificationStepTwoModels
        trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                                  "criteria_map": {"net_1": "criterion_ce_loss",
                                                                   "net_2": "criterion_ce_loss"},
                                                  "checks": {"net_1": identity, "net_2": identity},
                                                  "logs": ["loss"]}
    else:
        step = ClassificationStepTwoModels(**{"models": {"net_1": None, "net_2": None},
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]})
        step._returns = {"losses", "outputs", "labels", "total"}
        step._logs = ["loss"]
        trainer.config.update_functions.train = step
        trainer.config.update_functions.val = step
        trainer.config.update_functions.test = step
    trainer._init_all()
    if trainer._model_step_func:
        assert trainer.active_models == {}
        assert trainer.models_available == ["net_1", "net_2"]
        trainer.set_model({"net_1": "net_1", "net_2": "net_2"})
        assert trainer.models_available == ["net_1", "net_2"]
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
    else:
        assert trainer.active_models == {}
        assert trainer.models_available == ["net_1", "net_2"]
        trainer.set_model({"net_1": "net_1", "net_2": "net_2"})
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_models_no_gpus_unload_correctly(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "loaded": True,
                                        "params": {},
                                        "gpus": "auto"},
                              "net_2": {"model": Net2,
                                        "loaded": True,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "gpus": "auto"}}
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.gpus = [0, 1]
    trainer.trainer_params.cuda = True
    if trainer.config.update_functions.function is not None:
        trainer.config.update_functions.function = ClassificationStepTwoModels
        trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                                  "criteria_map": {"net_1": "criterion_ce_loss",
                                                                   "net_2": "criterion_ce_loss"},
                                                  "checks": {"net_1": identity, "net_2": identity},
                                                  "logs": ["loss"]}
    else:
        step = ClassificationStepTwoModels(**{"models": {"net_1": None, "net_2": None},
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]})
        step._returns = {"losses", "outputs", "labels", "total"}
        step._logs = ["loss"]
        trainer.config.update_functions.train = step
        trainer.config.update_functions.val = step
        trainer.config.update_functions.test = step
    trainer._init_all()
    if trainer._model_step_func:
        assert trainer.models_available == ["net_1", "net_2"]
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        assert trainer._models["net_1"].unload("RAM")
        assert trainer._models["net_2"].unload("RAM")
    else:
        assert trainer.models_available == ["net_1", "net_2"]
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        assert trainer._models["net_1"].unload("RAM")
        assert trainer._models["net_2"].unload("RAM")
    assert not trainer._training_steps["train"][0].models["net_1"].loaded
    assert not trainer._training_steps["train"][0].models["net_2"].loaded
    assert not trainer._training_steps["test"][0].models["net_1"].loaded
    assert not trainer._training_steps["test"][0].models["net_2"].loaded


@pytest.mark.todo
@pytest.mark.quick
@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without at least 2 GPUs.")
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
@pytest.mark.parametrize("loaded", [True, False])
def test_trainer_models_given_gpus_load_unload_correctly(params_and_trainer, loaded):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "loaded": loaded,
                                        "params": {},
                                        "gpus": [0]},
                              "net_2": {"model": Net2,
                                        "loaded": loaded,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "gpus": [1]}}
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.gpus = [0, 1]
    trainer.trainer_params.cuda = True
    if trainer.config.update_functions.function is not None:
        trainer.config.update_functions.function = ClassificationStepTwoModels
        trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                                  "criteria_map": {"net_1": "criterion_ce_loss",
                                                                   "net_2": "criterion_ce_loss"},
                                                  "checks": {"net_1": identity, "net_2": identity},
                                                  "logs": ["loss"]}
    else:
        step = ClassificationStepTwoModels(**{"models": {"net_1": None, "net_2": None},
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]})
        step._returns = {"losses", "outputs", "labels", "total"}
        step._logs = ["loss"]
        trainer.config.update_functions.train = step
        trainer.config.update_functions.val = step
        trainer.config.update_functions.test = step
    trainer._init_all()
    if trainer._model_step_func:
        assert trainer.models_available == ["net_1", "net_2"]
        if loaded:
            assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        else:
            assert trainer.active_models == {}
            trainer.set_model({"net_1": "net_1", "net_2": "net_2"})
            assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        assert trainer._models["net_1"].device == torch.device("cuda:0")
        assert trainer._models["net_2"].device == torch.device("cuda:1")
        assert trainer._models["net_1"].unload("RAM")
        assert trainer._models["net_2"].unload("RAM")
        assert trainer._models["net_1"].device == torch.device("cpu")
        assert trainer._models["net_2"].device == torch.device("cpu")
    else:
        assert trainer.models_available == ["net_1", "net_2"]
        if loaded:
            assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        else:
            assert trainer.active_models == {}
            trainer.set_model({"net_1": "net_1", "net_2": "net_2"})
            assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        assert trainer._models["net_1"].loaded
        assert trainer._models["net_2"].loaded
        assert trainer._models["net_1"].device == torch.device("cuda:0")
        assert trainer._models["net_2"].device == torch.device("cuda:1")
        assert trainer._models["net_1"].unload("RAM")
        assert trainer._models["net_2"].unload("RAM")
        assert trainer._models["net_1"].device == torch.device("cpu")
        assert trainer._models["net_2"].device == torch.device("cpu")


@pytest.mark.todo
@pytest.mark.quick
@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without at least 2 GPUs.")
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_models_auto_gpus_load_unload_correctly(params_and_trainer):
    params, trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "loaded": True,
                                        "params": {},
                                        "gpus": "auto"},
                              "net_2": {"model": Net2,
                                        "loaded": True,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "gpus": "auto"}}
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.gpus = [0, 1]
    trainer.trainer_params.cuda = True
    if trainer.config.update_functions.function is not None:
        trainer.config.update_functions.function = ClassificationStepTwoModels
        trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                                  "criteria_map": {"net_1": "criterion_ce_loss",
                                                                   "net_2": "criterion_ce_loss"},
                                                  "checks": {"net_1": identity, "net_2": identity},
                                                  "logs": ["loss"]}
    else:
        step = ClassificationStepTwoModels(**{"models": {"net_1": None, "net_2": None},
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]})
        step._returns = {"losses", "outputs", "labels", "total"}
        step._logs = ["loss"]
        trainer.config.update_functions.train = step
        trainer.config.update_functions.val = step
        trainer.config.update_functions.test = step
    trainer._init_all()
    if trainer._model_step_func:
        assert trainer.models_available == ["net_1", "net_2"]
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        # assert trainer._models["net_1"].device == torch.device("cuda:0")
        # assert trainer._models["net_2"].device == torch.device("cuda:1")
        assert trainer._models["net_1"].unload("RAM")
        assert trainer._models["net_2"].unload("RAM")
        pytest.set_trace()
    else:
        assert trainer.models_available == ["net_1", "net_2"]
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        # assert trainer._models["net_1"].device == torch.device("cuda:0")
        # assert trainer._models["net_2"].device == torch.device("cuda:1")
        assert trainer._models["net_1"].unload("RAM")
        assert trainer._models["net_2"].unload("RAM")
        pytest.set_trace()
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_models_parallel_load_unload_correctly(params_and_trainer):
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_models_dataparallel_load_unload_correctly(params_and_trainer):
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_models_add_model(params_and_trainer):
    params, trainer = params_and_trainer
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_models_load_models_state(params_and_trainer):
    params, trainer = params_and_trainer
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_models_get_new_models(params_and_trainer):
    params, trainer = params_and_trainer
    pass
