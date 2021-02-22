import pytest
import sys
import copy
import torch
sys.path.append("../")
from dorc.trainer import Trainer
from dorc.trainer.model import Model, ModelStep
from dorc.autoloads import ClassificationStep
from _setup_local import Net


def identity(x):
    return True


class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Net()
        self.b = Net()

    def forward(self, x):
        x_a = self.a(x)
        x_b = self.b(x)
        return x_a + x_b


class ClassificationStepTwoModels(ModelStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._returns = {"losses", "outputs", "labels", "total"}
        self._logs = ["loss"]

    def __call__(self, batch):
        model_1 = self.models["net_1"]
        model_2 = self.models["net_2"]
        criterion_1 = self.criteria["net_1"]
        criterion_2 = self.criteria["net_2"]
        inputs, labels = batch
        inputs = model_1.to_(inputs)
        labels = model_1.to_(labels)
        if self.train:
            model_1.train()
            model_2.train()
            model_1.optimizer.zero_grad()
            model_2.optimizer.zero_grad()
        outputs_1 = model_1(inputs)
        outputs_2 = model_2(model_2.to_(inputs))
        loss_1 = criterion_1(outputs_1, labels)
        loss_2 = criterion_2(outputs_2, model_2.to_(labels))
        if self.train:
            loss_1.backward()
            loss_2.backward()
        return {"losses": {"loss_1": loss_1.detach().item(), "loss_2": loss_2.detach().item()},
                "outputs": {"outputs_1": outputs_1.detach(), "outputs_2": outputs_2.detach()},
                "labels": labels.detach(), "total": len(labels)}


class SubTrainer(Trainer):
    def __init__(self, _cuda, *args, **kwargs):
        self._cuda = _cuda
        super().__init__(*args, **kwargs)
        self.reserved_gpus = []
        self.reserve_gpus = lambda x: [True, None]

    @property
    def have_cuda(self):
        return self._cuda


class FakeRequest:
    def __init__(self):
        self.form = {}
        self.json = None
        self.files = {}


def assertIn(a, b):
    assert a in b


def terminate_live_sessions(daemon):
    for s in daemon._sessions.values():
        for s_name in s["sessions"]:
            if "process" in s["sessions"][s_name]:
                s["sessions"][s_name]["process"].terminate()
                print(f'Terminated {s["sessions"][s_name]["process"]}')


def _create_session(daemon, config, load=False):
    data = {"name": "test_session", "config": copy.deepcopy(config), "load": load}
    daemon.create_session(0, data)
    result = daemon._check_result(0)
    return result


def make_daemon():
    import os
    import time
    import shutil
    import requests
    from dorc.daemon import _start_daemon

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
