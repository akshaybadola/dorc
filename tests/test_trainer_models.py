import pytest
import torch
from dorc.device import all_devices, useable_devices
from dorc.trainer import Trainer
from util import Net, Net2, SubTrainer, ClassificationStepTwoModels, identity


@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
@pytest.mark.parametrize("loaded", [True, False])
@pytest.mark.parametrize("cuda_given", [True, False])
@pytest.mark.parametrize("model_gpus", ["auto", [], [0], [1], [0, 1]])
def test_trainer_models_init_one_model_no_gpus(params_and_trainer,
                                               cuda_given,
                                               loaded,
                                               model_gpus):
    params, _trainer = params_and_trainer
    params["model_params"] = {"net": {"model": Net,
                                      "optimizer": "Adam",
                                      "loaded": loaded,
                                      "params": {},
                                      "gpus": model_gpus}}
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.cuda = cuda_given
    trainer.trainer_params.gpus = []  # no_gpus
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert trainer.gpus == [-1]
    assert "net" in trainer._training_steps["train"][0].models
    if loaded:
        assert trainer.active_models == {"net": "net"}
        assert trainer._models["net"].gpus == []
        assert trainer.devices == {"net": []}
    else:
        assert trainer.devices == {}
        assert trainer._models == {}
        result = trainer.set_model({"net": "net"})
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


@pytest.mark.gpus
@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without two GPUs.")
@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True], indirect=True)
@pytest.mark.parametrize("loaded", [True, False])
@pytest.mark.parametrize("cuda_given", [True, False])
@pytest.mark.parametrize("model_gpus", ["auto", [], [0], [1], [0, 1]])
@pytest.mark.parametrize("gpus", [[], None, [0], [0, 1]])
def test_trainer_models_init_one_model_with_gpus(params_and_trainer,
                                                 cuda_given,
                                                 loaded,
                                                 model_gpus,
                                                 gpus):
    params, _trainer = params_and_trainer
    params["model_params"] = {"net": {"model": Net,
                                      "optimizer": "Adam",
                                      "loaded": loaded,
                                      "params": {},
                                      "gpus": model_gpus}}
    trainer = SubTrainer(False, **params)
    trainer.trainer_params.cuda = cuda_given
    trainer.trainer_params.gpus = gpus
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    if cuda_given:
        tgpus = gpus or [-1]
    else:
        tgpus = [-1]
    assert trainer.gpus == tgpus
    if tgpus == [-1]:
        mgpus = []
    else:
        if model_gpus == "auto":
            mgpus = tgpus
        else:
            mgpus = list(set(model_gpus).intersection(tgpus))
    tdev = f"cuda:{mgpus[0]}" if mgpus else "cpu"
    assert "net" in trainer._training_steps["train"][0].models
    if loaded:
        assert trainer.active_models == {"net": "net"}
        assert trainer.devices == {"net": mgpus}
    else:
        assert trainer.devices == {}
        assert trainer._models == {}
    result = trainer.set_model({"net": "net"})
    assert trainer.active_models == {"net": "net"}
    assert trainer.devices == {"net": mgpus}
    assert trainer._models["net"].gpus == mgpus
    assert trainer._models["net"].device == torch.device(tdev)
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)



@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
@pytest.mark.parametrize("loaded", [True, False])
@pytest.mark.parametrize("cuda_given", [True, False])
@pytest.mark.parametrize("model1_gpus", ["auto", [], [0], [1], [0, 1]])
@pytest.mark.parametrize("model2_gpus", ["auto", [], [0], [1], [0, 1]])
def test_trainer_models_init_two_models_no_gpus(params_and_trainer,
                                                cuda_given,
                                                loaded,
                                                model1_gpus,
                                                model2_gpus):
    params, _trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": loaded,
                                        "gpus": model1_gpus},
                              "net_2": {"model": Net2,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": loaded,
                                        "gpus": model2_gpus}}
    trainer = SubTrainer(False, **params)
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
    trainer._init_device()
    trainer._init_models()
    trainer._init_data_and_dataloaders()
    trainer._init_update_funcs()
    trainer._init_training_steps()
    assert trainer.gpus == [-1]
    assert "net_1" in trainer._training_steps["train"][0].models
    assert "net_2" in trainer._training_steps["train"][0].models
    if loaded:
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        assert trainer._models["net_1"].gpus == []
        assert trainer._models["net_2"].gpus == []
        assert trainer.devices == {"net_1": [], "net_2": []}
    else:
        assert trainer.devices == {}
        assert trainer._models == {}
        result = trainer.set_model({"net_1": "net_1", "net_2": "net_2"})
    assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


@pytest.mark.quick
@pytest.mark.gpus
@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without two GPUs.")
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
@pytest.mark.parametrize("loaded", [True, False])
@pytest.mark.parametrize("cuda_given", [True, False])
@pytest.mark.parametrize("model1_gpus", ["auto", [], [0], [1], [0, 1]])
@pytest.mark.parametrize("model2_gpus", ["auto", [], [0], [1], [0, 1]])
def test_trainer_models_init_two_models_with_gpus(params_and_trainer,
                                                  cuda_given,
                                                  loaded,
                                                  model1_gpus,
                                                  model2_gpus):
    params, _trainer = params_and_trainer
    params["model_params"] = {"net_1": {"model": Net,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": loaded,
                                        "gpus": model1_gpus},
                              "net_2": {"model": Net2,
                                        "optimizer": "Adam",
                                        "params": {},
                                        "loaded": loaded,
                                        "gpus": model2_gpus}}
    trainer = SubTrainer(False, **params)
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
    gpus = [[], [0], [1], [0, 1]]
    for _gpus in gpus:
        trainer.config.trainer_params.cuda = cuda_given
        trainer.config.trainer_params.gpus = _gpus
        trainer._init_device()
        trainer._init_models()
        trainer._init_data_and_dataloaders()
        trainer._init_update_funcs()
        # pytest.set_trace()
        trainer._init_training_steps()
        if cuda_given:
            tgpus = _gpus or [-1]
        else:
            tgpus = [-1]
        assert trainer.gpus == tgpus
        if tgpus == [-1]:
            mgpus1 = []
            mgpus2 = []
        else:
            if model1_gpus == "auto" and model2_gpus == "auto":
                print("skipping both auto")
                continue
            elif model1_gpus == [] and model2_gpus == []:
                mgpus1 = []
                mgpus2 = []
            elif model1_gpus == "auto":
                mgpus2 = list(set(model2_gpus).intersection(tgpus))
                mgpus1 = list(set(tgpus) - set(mgpus2))
            elif model2_gpus == "auto":
                mgpus1 = list(set(model1_gpus).intersection(tgpus))
                mgpus2 = list(set(tgpus) - set(mgpus1))
            elif not set(model1_gpus).intersection(model2_gpus):
                mgpus1 = list(set(model1_gpus).intersection(tgpus))
                mgpus2 = list(set(model2_gpus).intersection(tgpus))
            else:
                print(f"Skipping possible conflict\n{model1_gpus, model2_gpus}")
                continue
        tdev1 = f"cuda:{mgpus1[0]}" if mgpus1 else "cpu"
        tdev2 = f"cuda:{mgpus2[0]}" if mgpus2 else "cpu"
        assert trainer.gpus == tgpus
        assert "net_1" in trainer._training_steps["train"][0].models
        assert "net_2" in trainer._training_steps["train"][0].models
        _args = {"model1_gpus": model1_gpus, "model2_gpus": model2_gpus,
                 "tgpus": tgpus, "mgpus1": mgpus1, "mgpus2": mgpus2,
                 "msf": trainer._model_step_func}
        if loaded:
            assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
            assert trainer._models["net_1"].gpus == mgpus1
            assert trainer._models["net_2"].gpus == mgpus2
            assert trainer.devices == {"net_1": mgpus1, "net_2": mgpus2}
        else:
            assert trainer.devices == {}
            assert trainer._models == {}
            result = trainer.set_model({"net_1": "net_1", "net_2": "net_2"})
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


@pytest.mark.gpus
@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True], indirect=True)
def test_trainer_models_init_many_models_no_auto_no_parallel_gpus_conflict(params_and_trainer):
    params, _trainer = params_and_trainer
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
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


@pytest.mark.gpus
@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without at least 2 GPUs.")
@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True], indirect=True)
def test_trainer_models_init_many_models_two_gpus_auto_no_parallel(params_and_trainer):
    params, _trainer = params_and_trainer
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
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
@pytest.mark.parametrize("loaded", [True, False])
def test_trainer_models_set_model_active_no_gpus(params_and_trainer, loaded):
    params, _trainer = params_and_trainer
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
    else:
        assert trainer.models_available == ["net_1", "net_2"]
        if loaded:
            assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        else:
            assert trainer.active_models == {}
    trainer.set_model({"net_2": "net_1"})
    if loaded:
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_1"}
    else:
        assert trainer.active_models == {"net_2": "net_1"}
        trainer.set_model({"net_1": "net_1"})
    assert trainer.active_models == {"net_1": "net_1", "net_2": "net_1"}
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_models_no_gpus_load_correctly(params_and_trainer):
    params, _trainer = params_and_trainer
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
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


@pytest.mark.quick
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_models_no_gpus_unload_correctly(params_and_trainer):
    params, _trainer = params_and_trainer
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
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


@pytest.mark.quick
@pytest.mark.gpus
@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without at least 2 GPUs.")
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
@pytest.mark.parametrize("loaded", [True, False])
def test_trainer_models_given_gpus_load_unload_correctly(params_and_trainer, loaded):
    params, _trainer = params_and_trainer
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
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


@pytest.mark.quick
@pytest.mark.gpus
@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without at least 2 GPUs.")
@pytest.mark.parametrize("params_and_trainer", [True, False], indirect=True)
def test_trainer_models_auto_gpus_load_unload_correctly(params_and_trainer):
    params, _trainer = params_and_trainer
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
    else:
        assert trainer.models_available == ["net_1", "net_2"]
        assert trainer.active_models == {"net_1": "net_1", "net_2": "net_2"}
        # assert trainer._models["net_1"].device == torch.device("cuda:0")
        # assert trainer._models["net_2"].device == torch.device("cuda:1")
        assert trainer._models["net_1"].unload("RAM")
        assert trainer._models["net_2"].unload("RAM")
    trainer._training_steps["test"][0](trainer.train_loader.__iter__().__next__())
    for handler in _trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


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
    params, _trainer = params_and_trainer
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_models_load_models_state(params_and_trainer):
    params, _trainer = params_and_trainer
    pass


@pytest.mark.todo
@pytest.mark.quick
def test_trainer_models_get_new_models(params_and_trainer):
    params, _trainer = params_and_trainer
    pass
