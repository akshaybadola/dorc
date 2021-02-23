import os
import shutil
import pytest
import torch
from datetime import datetime
from pydantic import ValidationError
from _setup_local import config
from dorc.device import all_devices, useable_devices
from util import Net, Net2, SubTrainer, ClassificationStepTwoModels, identity



@pytest.mark.quick
def test_trainer_should_have_no_gpus_on_bad_params(params_and_trainer):
    params, _ = params_and_trainer
    trainer = SubTrainer(_cuda=False, **params)
    cases = [None, "bleh", -2, False, {"bleh"}, ["test"]]
    for i, case in enumerate(cases):
        if case == "bleh" or case == {"bleh"} or case == ["test"]:
            with pytest.raises(ValidationError):
                trainer.trainer_params.gpus = case
        assert trainer.gpus == []


@pytest.mark.quick
def test_trainer_should_have_correct_gpus_on_good_params(params_and_trainer):
    params, _ = params_and_trainer
    trainer = SubTrainer(_cuda=False, **params)
    cases = [0, 4, [0, 1], [-1]]
    for i, case in enumerate(cases):
        trainer.trainer_params.gpus = case
        assert trainer.gpus == case if isinstance(case, list) else [case]


@pytest.mark.todo
@pytest.mark.quick
def test_allocate_devices(params_and_trainer):
    # 1. explicitly mentioned gpus are given preference
    # 2. after that auto
    # 3. after that parallel
    # 4. What about distributed?
    pass


@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
@pytest.mark.quick
def test_trainer_should_set_valid_gpus(params_and_trainer):
    params, _ = params_and_trainer
    trainer = SubTrainer(_cuda=False, **params)
    have_gpus = all_devices()
    useable_gpus = useable_devices()
    if have_gpus != useable_gpus:
        trainer.gpus = have_gpus
        trainer._maybe_init_gpus()
        assert trainer.gpus == useable_gpus
    trainer.reserved_gpus = []
    trainer.gpus = have_gpus + [*range(max(have_gpus) + 1, max(have_gpus) + 3)]
    trainer._maybe_init_gpus()
    assert trainer.gpus == useable_gpus


@pytest.mark.quick
@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without two GPUs.")
@pytest.mark.parametrize("loaded", [True, False])
@pytest.mark.parametrize("cuda_given", [True, False])
@pytest.mark.parametrize("model_gpus", ["auto", [], [0], [1], [0, 1]])
@pytest.mark.parametrize("gpus", [([], "cpu"), (None, "cpu"),
                                  ([0], "cuda:0"), ([0, 1], "cuda:0")])
def test_trainer_should_set_correct_device_one_model(params_and_trainer,
                                                     cuda_given,
                                                     model_gpus,
                                                     loaded,
                                                     gpus):
    params, _ = params_and_trainer
    trainer = SubTrainer(_cuda=False, **params)
    trainer.trainer_params.cuda = cuda_given
    trainer.model_params["net"].gpus = model_gpus
    trainer.model_params["net"].loaded = loaded
    cases = [[], [0], [1], [0, 1]]
    for reserved in cases:
        trainer.trainer_params.gpus = gpus[0]
        trainer.reserved_gpus = reserved
        trainer._maybe_init_gpus()
        trainer._set_device()
        available_gpus = list(set(gpus[0] or []) - set(reserved))
        if cuda_given:
            tgpus = available_gpus if bool(available_gpus) else [-1]
        else:
            tgpus = [-1]
        assert trainer.gpus == tgpus
        trainer._init_models()
        trainer._init_update_funcs()
        if tgpus == [-1]:
            mgpus = []
        else:
            if model_gpus == "auto":
                mgpus = tgpus
            else:
                mgpus = list(set(model_gpus).intersection(tgpus))
        tdev = f"cuda:{mgpus[0]}" if mgpus else "cpu"
        if loaded:
            assert trainer._models["net"].loaded
            assert trainer.devices == {"net": mgpus}
            assert trainer._models["net"].device == torch.device(tdev)
        else:
            assert trainer.devices == {}
            assert trainer._models == {}


@pytest.mark.quick
@pytest.mark.skipif(len(all_devices()) < 2, reason=f"Cannot run without two GPUs.")
@pytest.mark.parametrize("model1_gpus", ["auto", [], [0], [1], [0, 1]])
@pytest.mark.parametrize("model2_gpus", ["auto", [], [0], [1], [0, 1]])
@pytest.mark.parametrize("gpus", [([], "cpu"), (None, "cpu"),
                                  ([0], "cuda:0"), ([0, 1], "cuda:0")])
@pytest.mark.parametrize("params_and_trainer", [True], indirect=True)
def test_trainer_should_set_correct_device_two_models(params_and_trainer,
                                                      model1_gpus,
                                                      model2_gpus,
                                                      gpus):
    loaded = True
    cuda_given = True
    params, _ = params_and_trainer
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
    params["log_levels"] = {"file": "debug", "stream": "error"}
    trainer = SubTrainer(_cuda=False, **params)
    trainer.trainer_params.cuda = cuda_given
    trainer.config.update_functions.function = ClassificationStepTwoModels
    trainer.config.update_functions.params = {"models": ["net_1", "net_2"],
                                              "criteria_map": {"net_1": "criterion_ce_loss",
                                                               "net_2": "criterion_ce_loss"},
                                              "checks": {"net_1": identity, "net_2": identity},
                                              "logs": ["loss"]}
    cases = [[], [0], [1], [0, 1]]
    for reserved in cases:
        trainer.trainer_params.gpus = gpus[0]
        trainer.reserved_gpus = reserved
        trainer._maybe_init_gpus()
        trainer._set_device()
        available_gpus = list(set(gpus[0] or []) - set(reserved))
        if cuda_given:
            tgpus = available_gpus if bool(available_gpus) else [-1]
        else:
            tgpus = [-1]
        assert trainer.gpus == tgpus
        trainer._init_models()
        trainer._init_data_and_dataloaders()
        trainer._init_update_funcs()
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
                mgpus2 = model2_gpus
                mgpus1 = list(set(model2_gpus).intersection(tgpus))
            elif model2_gpus == "auto":
                mgpus1 = model1_gpus
                mgpus2 = list(set(model1_gpus).intersection(tgpus))
            elif not set(model1_gpus).intersection(model2_gpus):
                mgpus1 = list(set(model1_gpus).intersection(tgpus))
                mgpus2 = list(set(model2_gpus).intersection(tgpus))
            else:
                print(f"Skipping possible conflict\n{model1_gpus, model2_gpus}")
                continue
        tdev1 = f"cuda:{mgpus1[0]}" if mgpus1 else "cpu"
        tdev2 = f"cuda:{mgpus2[0]}" if mgpus2 else "cpu"
        if loaded:
            assert trainer._models["net_1"].loaded
            assert trainer._models["net_2"].loaded
            assert trainer.devices == {"net_1": mgpus1, "net_2": mgpus2}
            assert trainer._models["net_1"].device == torch.device(tdev1)
            assert trainer._models["net_2"].device == torch.device(tdev2)
        else:
            assert trainer.devices == {}
            assert trainer._models == {}


# NOTE: This should test correct allocation of devices across many gpus and
#       perhaps DDP
@pytest.mark.todo
@pytest.mark.quick
def test_check_trainer_set_device_load_unload_auto_adjust(params_and_trainer):
    pass
