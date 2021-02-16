import sys
import shutil
import datetime
import os
import time
import requests
from threading import Thread
from pathlib import Path
import pytest
import importlib
sys.path.append("..")
from dorc.autoloads import ClassificationStep
from dorc.trainer import config as trainer_config
from dorc.interfaces import FlaskInterface
from dorc.interfaces.translation import TranslationLayer
from dorc.trainer import Trainer
import _setup


def create_module(module_dir, module_files=[]):
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    if not os.path.exists(os.path.join(module_dir, "__init__.py")):
        with open(os.path.join(module_dir, "__init__.py"), "w") as f:
            f.write("")
    for f in module_files:
        shutil.copy(f, module_dir)


@pytest.fixture
def json_config():
    importlib.reload(_setup)
    return _setup.config


@pytest.fixture
def trainer(json_config):
    config = json_config
    if os.path.exists(".test_dir"):
        shutil.rmtree(".test_dir")
    os.mkdir(".test_dir")
    os.mkdir(".test_dir/test_session")
    time_str = datetime.datetime.now().isoformat()
    data_dir = f".test_dir/test_session/{time_str}"
    os.mkdir(data_dir)
    tlayer = TranslationLayer(config, data_dir)
    test_config = tlayer.from_json()
    test_config.pop("model_step_params")
    return Trainer(**test_config)


@pytest.fixture
def setup_and_net():
    from _setup_local import config, Net
    return config, Net


@pytest.fixture
def get_step(setup_and_net):
    config, _ = setup_and_net
    checks = {x: lambda _: True for x in config['model_params']}
    cname = [*config["criteria"].keys()][0]
    criteria_map = {x: cname for x in config['model_params']}
    model_names = [*config['model_params'].keys()]
    step = ClassificationStep(models=model_names, criteria_map=criteria_map,
                              checks=checks, logs=["loss"])
    step.returns = {"loss", "outputs", "labels", "total"}
    step.set_criteria({"net": config["criteria"][cname]["function"](
        **config["criteria"][cname]["params"])})
    return step


@pytest.fixture
def params_and_trainer(setup_and_net):
    config, _ = setup_and_net
    if os.path.exists(".test_dir"):
        shutil.rmtree(".test_dir")
    os.mkdir(".test_dir")
    os.mkdir(".test_dir/test_session")
    time_str = datetime.datetime.now().isoformat()
    os.mkdir(f".test_dir/test_session/{time_str}")
    data_dir = f".test_dir/test_session/{time_str}"
    params = {"data_dir": data_dir, **config}
    yield (params, Trainer(**params))
    shutil.rmtree(".test_dir")


@pytest.fixture(scope="module")
def params_and_iface():
    importlib.reload(_setup)
    config = _setup.config.copy()
    hostname = "127.0.0.1"
    port = 12321
    root_dir = ".test_dir"
    data_dir = os.path.join(".test_dir", "test_iface", datetime.datetime.now().isoformat())
    host = "http://" + ":".join([hostname, str(port)]) + "/"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    create_module(os.path.abspath(os.path.join(root_dir, "global_modules")),
                  [os.path.abspath("../dorc/autoloads.py")])
    sys.path.append(os.path.abspath(root_dir))
    iface = FlaskInterface(hostname, port, data_dir, no_start=True)
    # from global_modules import autoloads
    status, message = iface.create_trainer(config)
    iface_thread = Thread(target=iface.start)
    iface_thread.start()
    time.sleep(1)
    yield ({"config": config, "host": host, "data_dir": data_dir}, iface)
    requests.get(host + "_shutdown")
    shutil.rmtree(".test_dir")


@pytest.fixture(scope="package")
def daemon_and_cookies():
    import time
    import requests
    from dorc.daemon import Daemon
    data_dir = ".test_dir"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    port = 23232
    hostname = "127.0.0.1"
    daemon = Daemon(hostname, port, ".test_dir", "test_name")
    thread = Thread(target=daemon.start)
    thread.start()
    host = "http://" + ":".join([hostname, str(port) + "/"])
    time.sleep(.5)
    cookies = requests.request("POST", host + "login",
                               data={"username": "admin",
                                     "password": "AdminAdmin_33"}).cookies
    yield (daemon, cookies)
    requests.get(host + "_shutdown", cookies=cookies)


@pytest.fixture
def basic_config(setup_and_net):
    config, _ = setup_and_net
    trainer_params = trainer_config.TrainerParams(**config["trainer_params"])
    model_params = {k: trainer_config.ModelParams(**v)
                    for k, v in config["model_params"].items()}
    optimizers = {k: trainer_config.Optimizer(**v)
                  for k, v in config["optimizers"].items()}
    criteria = {k: trainer_config.Criterion(**v)
                for k, v in config["criteria"].items()}
    update_functions = trainer_config.UpdateFunctions(**config["update_functions"])
    extra_metrics = {k: trainer_config.Metric(**v)
                     for k, v in config["extra_metrics"].items()}
    data_params = trainer_config.DataParams(name=config["data_params"]["name"],
                                            train=config["data_params"]["train"],
                                            val=config["data_params"]["val"],
                                            test=config["data_params"]["test"])
    dataloader_params = trainer_config.DataLoaderParams(**config["dataloader_params"])
    log_levels = trainer_config.LogLevelParams(**config.get("log_levels"))
    return trainer_config.Config(model_params=model_params,
                                 trainer_params=trainer_params,
                                 optimizers=optimizers,
                                 criteria=criteria,
                                 update_functions=update_functions,
                                 log_levels=log_levels,
                                 data_params=data_params,
                                 extra_metrics=extra_metrics,
                                 dataloader_params=dataloader_params,
                                 data_dir=Path("."))
