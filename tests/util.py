import pytest
import sys
import shutil
import datetime
import os
import torch
sys.path.append("../")
from trainer.trainer import Trainer
from trainer.trainer.model import Model
from trainer.autoloads import ClassificationStep



def make_daemon():
    import os
    import time
    import shutil
    import requests
    from trainer.daemon import _start_daemon

    data_dir = ".test_dir"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    port = 23232
    hostname = "127.0.0.1"
    daemon = _start_daemon(hostname, port, ".test_dir")
    host = "http://" + ":".join([hostname, str(port) + "/"])
    time.sleep(.5)
    cookies = requests.request("POST", host + "login",
                               data={"username": "admin",
                                     "password": "AdminAdmin_33"}).cookies
    return daemon, cookies


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
