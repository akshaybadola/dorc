import sys
import base64
import random
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
import util
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


    # config = json_config
    # if os.path.exists(".test_dir"):
    #     shutil.rmtree(".test_dir")
    # os.mkdir(".test_dir")
    # os.mkdir(".test_dir/test_session")
    # time_str = datetime.datetime.now().isoformat()
    # data_dir = f".test_dir/test_session/{time_str}"
    # gmods_dir = os.path.abspath(".test_dir/global_modules")
    # gdata_dir = os.path.abspath(".test_dir/global_datasets")
    # os.mkdir(gmods_dir)
    # os.mkdir(gdata_dir)
    # os.mkdir(data_dir)
    # extra_opts = {"data_dir": data_dir,
    #               "global_modules_dir": gmods_dir,
    #               "global_datasets_dir": gdata_dir}
    # tlayer = TranslationLayer(config, extra_opts)
    # test_config = tlayer.from_json()
    # test_config.pop("model_step_params")
    # return Trainer(**test_config)


@pytest.fixture(scope="module")
def indirect_config(request):
    gm_dir = os.path.abspath("_some_modules_dir/global_modules")
    tc_dir = "test_config_dir"
    if os.path.exists(gm_dir):
        shutil.rmtree(os.path.dirname(gm_dir))
    if os.path.exists(tc_dir):
        shutil.rmtree(tc_dir)
    os.makedirs(gm_dir)
    create_module(gm_dir, [os.path.join("../dorc/", x)
                           for x in ["autoloads.py"]])
    os.mkdir(tc_dir)
    if hasattr(request, "param") and request.param:
        param = request.param
    else:
        param = "json"
    if param == "pybytes":
        with open("_setup_py.py", "rb") as f:
            conf_bytes = f.read()
        yield conf_bytes
    elif param == "pystr":
        with open("_setup_py.py", "rb") as f:
            conf_bytes = f.read()
        yield base64.b64encode(conf_bytes)
    else:
        importlib.reload(_setup)
        yield _setup.config
    if os.path.exists(gm_dir):
        shutil.rmtree(os.path.dirname(gm_dir))
    if os.path.exists(tc_dir):
        shutil.rmtree(tc_dir)


@pytest.fixture
def trainer_json_config(json_config):
    config = json_config
    gmods_dir = os.path.abspath(".test_dir/global_modules")
    gdata_dir = os.path.abspath(".test_dir/global_datasets")
    if not os.path.exists(".test_dir"):
        os.mkdir(".test_dir")
    if not os.path.exists(".test_dir/test_session"):
        os.mkdir(".test_dir/test_session")
    if not os.path.exists(gmods_dir):
        os.mkdir(gmods_dir)
    if not os.path.exists(gdata_dir):
        os.mkdir(gdata_dir)
    time_str = datetime.datetime.now().isoformat()
    data_dir = f".test_dir/test_session/{time_str}"
    os.mkdir(data_dir)
    extra_opts = {"data_dir": data_dir,
                  "global_modules_dir": gmods_dir,
                  "global_datasets_dir": gdata_dir}
    tlayer = TranslationLayer(config, extra_opts)
    test_config = tlayer.from_json()
    test_config.pop("model_step_params")
    trainer = Trainer(**test_config)
    yield (test_config, trainer)


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
def params_old_config(setup_and_net):
    config, _ = setup_and_net
    gmods_dir = os.path.abspath(".test_dir/global_modules")
    gdata_dir = os.path.abspath(".test_dir/global_datasets")
    if not os.path.exists(".test_dir"):
        os.mkdir(".test_dir")
        os.mkdir(".test_dir/test_session")
        os.mkdir(gmods_dir)
        os.mkdir(gdata_dir)
    time_str = datetime.datetime.now().isoformat()
    os.mkdir(f".test_dir/test_session/{time_str}")
    data_dir = os.path.abspath(f".test_dir/test_session/{time_str}")
    params = {"data_dir": data_dir, "global_modules_dir": gmods_dir,
              "global_datasets_dir": gdata_dir, **config}
    trainer = Trainer(**params)
    yield (params, trainer)
    # for handler in trainer._logger.handlers:
    #     handler.close()
    #     trainer._logger.removeHandler(handler)


@pytest.fixture
def params_and_trainer(request, trainer_json_config, params_old_config):
    if hasattr(request, "param") and request.param:
        params, trainer = trainer_json_config
    else:
        params, trainer = params_old_config
    yield (params, trainer)
    # for handler in trainer._logger.handlers:
    #     handler.close()
    #     trainer._logger.removeHandler(handler)
    shutil.rmtree(".test_dir")


@pytest.fixture(scope="module")
def params_and_iface():
    importlib.reload(_setup)
    config = _setup.config.copy()
    hostname = "127.0.0.1"
    port = random.randint(12321, 19321)
    root_dir = os.path.abspath(".test_dir")
    data_dir = os.path.join(".test_dir", "test_iface", datetime.datetime.now().isoformat())
    host = "http://" + ":".join([hostname, str(port)]) + "/"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    gmods_dir = os.path.join(root_dir, "global_modules")
    gdata_dir = os.path.join(root_dir, "global_datasets")
    create_module(gmods_dir, [os.path.abspath("../dorc/autoloads.py")])
    create_module(gdata_dir)
    sys.path.append(os.path.abspath(root_dir))
    with open("_setup_py.py", "rb") as f:
        conf_bytes = f.read()
    util.write_py_config(conf_bytes, data_dir, gmods_dir, gdata_dir)
    iface = FlaskInterface(hostname, port, data_dir, gmods_dir, gdata_dir, no_start=True)
    # from global_modules import autoloads
    status, message = iface.create_trainer()
    iface_thread = Thread(target=iface.start)
    iface_thread.start()
    time.sleep(1)
    yield ({"config": config, "host": host,
            "gmods_dir": gmods_dir,
            "gdata_dir": gdata_dir,
            "data_dir": data_dir}, iface)
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
                                 data_dir=Path("."),
                                 global_modules_dir=Path("."),
                                 global_datasets_dir=Path("."))
