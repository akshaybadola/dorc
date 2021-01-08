import pytest
import sys
import torch
sys.path.append("../")
from trainer.model import Model
from trainer.autoloads import ClassificationStep


@pytest.fixture
def get_local_config():
    from _setup_local import config
    return config


@pytest.fixture
def get_remote_config():
    from _setup import config
    return config


def get_batch():
    return [torch.rand(8, 1, 28, 28), torch.Tensor([0]*8).long()]


def get_model(name, config, gpus):
    _name = "net"
    model_def = config["model_params"][_name]["model"]
    params = config["model_params"][_name]["params"]
    optimizer = {"name": "Adam",
                 **config["optimizers"]["Adam"]}
    model = Model(name, model_def, params, optimizer, gpus)
    model.load_into_memory()
    return model


def get_model_batch(name, config, gpus):
    _name = "net"
    model_def = config["model_params"][_name]["model"]
    params = config["model_params"][_name]["params"]
    optimizer = {"name": "Adam",
                 **config["optimizers"]["Adam"]}
    model = Model(name, model_def, params, optimizer, gpus)
    model.load_into_memory()
    return model, get_batch()


def get_step(models, config, train_or_test):
    checks = {x: lambda _: True for x in config['model_params']}
    cname = [*config["criteria"].keys()][0]
    criteria = {x: cname for x in config['model_params']}
    model_names = [*config['model_params'].keys()]
    step = ClassificationStep(model_names, criteria, checks)
    step.models = models
    step.returns = {"loss", "outputs", "labels", "total"}
    step.criteria = {cname: config["criteria"][cname]["function"](
        **config["criteria"][cname]["params"])}
    if train_or_test == "train":
        step.train = True
    else:
        step.train = False
    return step
