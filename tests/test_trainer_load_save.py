import pytest
import os
import shutil
from datetime import datetime
import unittest
import sys
import json
import torch
from dorc.device import all_devices, useable_devices
from dorc.trainer import Trainer


class FakeRequest:
    def __init__(self):
        self.form = {}
        self.json = None
        self.files = {}


class SubTrainer(Trainer):
    def __init__(self, _cuda, *args, **kwargs):
        self._cuda = _cuda
        super().__init__(*args, **kwargs)

    @property
    def have_cuda(self):
        return self._cuda


@pytest.mark.quick
def test_trainer_load_saves_bad_params(params_and_trainer):
    params, _ = params_and_trainer
    trainer = SubTrainer(False, **params)
    trainer.reserved_gpus = []
    trainer.reserve_gpus = lambda x: [True, None]
    trainer.trainer_params.cuda = True
    data = {}
    with pytest.raises(TypeError):
        assert trainer.load_saves(data)
    data = {"weights": "meh", "method": "woof"}
    result = trainer.load_saves(**data)
    assert not result.status
    assert result.message == "[load_saves()] Invalid or no such method"
    data = {"weights": "meh", "method": "load"}
    assert trainer.load_saves(**data).message == "[load_saves()] No such file"


@pytest.mark.quick
def test_trainer_load_saves_good_params(params_and_trainer):
    params, _ = params_and_trainer
    trainer = SubTrainer(False, **params)
    trainer.reserved_gpus = []
    trainer.reserve_gpus = lambda x: [True, None]
    # trainer.trainer_params.cuda = True
    trainer._init_all()
    trainer._save("_save.pth")
    trainer._save_path_with_epoch
    trainer._save_path_without_epoch
    os.remove("_save.pth")


@pytest.mark.quick
def test_trainer_load_weights_no_gpu(params_and_trainer, setup_and_net):
    params, _ = params_and_trainer
    _, Net = setup_and_net
    params["model_params"] = {"net_1": {"model": Net, "optimizer": "Adam",
                                        "params": {}, "gpus": "auto"},
                              "net_2": {"model": Net, "optimizer": "Adam",
                                        "params": {}, "gpus": "auto"}}
    trainer = SubTrainer(False, **params)
    trainer.reserved_gpus = []
    trainer.reserve_gpus = lambda x: [True, None]
    trainer.trainer_params.gpus = []
    trainer.trainer_params.cuda = True
    trainer._init_device()
    trainer._init_models()
    with pytest.raises(TypeError):
        trainer.load_weights(["net_1", "net_2"])
    with open("_test_weights.pth", "rb") as f:
        tmp = f.read()
    ret = trainer.load_weights(["net_1", "net_2"], tmp)
    assert not ret.status, "weights_for_only_one_model_given"
    assert "given weights" in ret.message.lower()
    net = torch.load("_test_weights.pth")
    weights = net["net"]
    torch.save({"net_1": weights, "net_2": weights}, "_temp_weights.pth")
    with open("_temp_weights.pth", "rb") as f:
        tmp = f.read()
    ret = trainer.load_weights(["net_1", "net_2"], tmp)
    assert ret.status


@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without at least 2 GPUs.")
def test_trainer_load_weights_two_models_two_gpus(params_and_trainer, setup_and_net):
    params, _ = params_and_trainer
    _, Net = setup_and_net
    params["model_params"] = {"net_1": {"model": Net, "optimizer": "Adam",
                                        "params": {}, "gpus": "auto"},
                              "net_2": {"model": Net, "optimizer": "Adam",
                                        "params": {}, "gpus": "auto"}}
    trainer = SubTrainer(False, **params)
    trainer.reserved_gpus = []
    trainer.reserve_gpus = lambda x: [True, None]
    trainer.trainer_params.cuda = True
    trainer.trainer_params.gpus = [0, 1]
    trainer._init_device()
    trainer._init_models()
    net = torch.load("_test_weights.pth")
    weights = net["net"]
    torch.save({"net_1": weights, "net_2": weights}, "_temp_weights.pth")
    with open("_temp_weights.pth", "rb") as f:
        tmp = f.read()
    ret = trainer.load_weights(["net_1", "net_2"], tmp)
    assert ret.status
    assert all(x.device == torch.device("cuda:1")
               for x in trainer._models["net_1"].weights.values())
    assert all(x.device == torch.device("cuda:0")
               for x in trainer._models["net_0"].weights.values())


@pytest.mark.quick
def test_trainer_get_state(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_all()
    trainer._get_state(True)
    state = trainer._get_state()
    import torch
    torch.save(state, "_state.pth")
    os.remove("_state.pth")


@pytest.mark.quick
def test_trainer_save_and_resume(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_all()
    state = trainer._get_state()
    result = trainer._resume_from_state(state)
    assert result.status


@pytest.mark.quick
def test_trainer_should_not_resume_from_missing_param_keys(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_all()
    state = trainer._get_state()
    bleh = state.dict()
    bleh.pop("epoch")
    result = trainer._resume_from_state(bleh)
    assert not result.status
    bleh["epoch"] = 20
    bleh["data"] = "not mnist"
    result = trainer._resume_from_state(bleh)
    assert not result.status


@pytest.mark.quick
def test_trainer_should_not_resume_from_missing_model_names(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_all()
    state = trainer._get_state()
    bleh = state.dict()
    models = bleh["models"]
    bleh["models"] = {"meh": [*models.values()][0]}
    result = trainer._resume_from_state(bleh)
    assert not result.status


@pytest.mark.quick
def test_trainer_should_not_resume_from_extra_model_names(params_and_trainer):
    params, trainer = params_and_trainer
    trainer._init_all()
    state = trainer._get_state()
    bleh = state.dict()
    models = bleh["models"]
    bleh["models"]["meh"] = [*models.values()][0]
    result = trainer._resume_from_state(bleh)
    assert result.status
    assert "meh" not in trainer.models
    bleh["models"]["meh"]["model_def"] = trainer._models["net"]._model_def
    result = trainer._resume_from_state(bleh)
    assert result.status
    assert "meh" not in trainer.models
