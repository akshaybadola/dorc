import pytest
import os
import time
import json
import shutil
import requests
import zipfile

from util import assertIn
from dorc.daemon.models import SessionMethodResponseModel as smod


mod_string = """
import os
from subprocess import Popen
from threading import Thread
import requests
import json
import random


url = "http://localhost:20202/_helpers/"


def add_and_set_active(model_name):
    response = requests.request("POST", url + "add_model",
                                files={"file": open("modules/test_model.py")})
    print(response.content)
    response = requests.post(url + "set_model", json=model_name)
    print(response.content)


# def add_and_set_active(model_name):
#     response = requests.request("POST", url + "add_model",
#                                 files={"file": open("modules/att_model.py")})
#     print(response.content)
#     response = requests.post(url + "set_model", json=model_name)
#     print(response.content)


def load_weights(model_name):
    data = {}
    data["model_names"] = [model_name]
    with open(f"modules/{model_name}.pth", "rb") as f:
        response = requests.request("POST", url + "load_weights",
                                    files={"file": f},
                                    data={"model_names": json.dumps([model_name])})
    print(response.content)


def call_adhoc_run():
    response = requests.request("POST", "http://localhost:20202/_extras/call_adhoc_run",
                                json={"val": {"data": "val",
                                              "metrics": ["loss"],
                                              "epoch": "current",
                                              "num_or_fraction": 50}})
    print(response.content)


def add_func():
    response = requests.request("POST", "http://localhost:20202/_helpers/add_user_funcs",
                                files={"file": open("modules/test_funcs.py")})
    print(response.content)


def report_adhoc_run():
    response = requests.request("POST", "http://localhost:20202/_extras/report_adhoc_run",
                                json={"report_function": "ruotianlou_report_func_2"})
    print(response.content)


def call_sample():
    response = requests.request("POST", "http://localhost:20202/_extras/call_user_func",
                                json=["model_sample"])
    print(response.content)


def check_image():
    img_dir = "/home/joe/lib/docprocess/images_pdfs/"
    img_dir = "/datacache/coco/test2014/"
    f = os.path.join(img_dir,
                     random.choice([x for x in os.listdir(img_dir)
                                    if x.endswith("png") or x.endswith("jpg")]))
    print(f)
    Thread(target=Popen, args=[["qiv", f]]).run()
    requests.request("POST", "http://localhost:20202/_helpers/load_image",
                     files={"file": open(os.path.join(img_dir, f), "rb")},
                     data={"callbacks": json.dumps(["model_sample_image",
                                                    "get_features_from_image_data"])})
    # for f in os.listdir(img_dir):
    #     if f.endswith("png"):
    #         requests.request("POST", "http://localhost:20202/_helpers/load_image",
    #                          files={"file": open(os.path.join(img_dir, f), "rb")},
    #                          data={"callbacks": json.dumps(["model_sample_image",
    #                                                         "get_features_from_image_data"])})


def hack_param():
    # requests.request("POST", "http://localhost:20202/_helpers/hack_param",
    #                  json={"_unique_id": {"type": "str", "value": "new_training"}})
    print(requests.request("POST", "http://localhost:20202/_helpers/hack_param",
                           json={"_hooks_run_iter_frequency": {"type": "int", "value": "1000"}}).content)


# add_func()
# hack_param()
# add_and_set_active("att2in2")
# load_weights("att2in2")
# check_image()
# call_sample()
# call_adhoc_run()
# report_adhoc_run()
"""


@pytest.mark.http
def test_list_modules(daemon_and_cookies, json_config):
    daemon, cookies = daemon_and_cookies
    time.sleep(1)
    host = f"http://{daemon.hostname}:{daemon.port}/"
    response = requests.request("GET", host + "list_global_modules", cookies=cookies)
    assert isinstance(json.loads(response.content), list)
    assert isinstance(json.loads(response.content)[1], dict)
    mods = json.loads(response.content)
    expected = {"autoloads": ["torch", "ModelStep", "sys",
                              "ClassificationStep",
                              "accuracy", "CheckFunc",
                              "CheckGreater",
                              "CheckGreaterName",
                              "CheckLesserName",
                              "CheckAccuracy"]}
    for k in mods[1]:
        assert k in expected
        assert set(mods[1][k]) == set(expected[k])


@pytest.mark.http
def test_add_delete_global_module(daemon_and_cookies, json_config):
    daemon, cookies = daemon_and_cookies
    host = f"http://{daemon.hostname}:{daemon.port}/"
    with open("./._test_py_file.py", "w") as f:
        f.write(mod_string)
    with open("./._test_py_file.py", "rb") as f:
        response = requests.request("POST", host + "add_global_module",
                                    files={"file": f},
                                    data={"name": json.dumps("test_module_a")},
                                    cookies=cookies)
    time.sleep(.5)
    result = smod.parse_obj(json.loads(response.content))
    assert result.status
    task_id = result.task_id
    time.sleep(1)
    response = requests.request("GET", host + f"check_task?task_id={task_id}",
                                cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    assert result.status
    assertIn("_module_test_module_a", daemon._modules)
    with open("./._test_py_file.py", "rb") as f:
        response = requests.request("POST", host + "add_global_module",
                                    files={"file": f},
                                    data={"name": json.dumps("test_module_b")},
                                    cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    assert result.status
    task_id = result.task_id
    time.sleep(1)
    response = requests.request("GET", host + f"check_task?task_id={task_id}",
                                cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    assert result.status
    assertIn("_module_test_module_b", daemon._modules)
    response = requests.request("POST", host + "delete_global_module",
                                data={"name": "test_module_a"},
                                cookies=cookies)
    time.sleep(.3)
    result = response.content
    print(result)
    assert "_module_test_module_a" not in daemon._modules


@pytest.mark.http
@pytest.mark.skipif("IN_GITHUB_WORKFLOW" in os.environ, reason="Don't run in Github workflow")
def test_add_delete_dataset(daemon_and_cookies, json_config):
    py_string = """
class MnistDataset:
    def __init__(self, pt_file):
        import torch
        data = torch.load(pt_file)

    def __len__(daemon_and_cookies, json_config):
        return len(data[0])

    def __getitem__(self, idx):
        return data[0][idx], data[1][idx]

import os
file_dir = os.path.abspath(os.path.dirname(__file__))
dataset = MnistDataset(os.path.join(file_dir, "training.pt"))
"""
    daemon, cookies = daemon_and_cookies
    host = f"http://{daemon.hostname}:{daemon.port}/"
    if not os.path.exists("./._test_data_dir"):
        os.mkdir("./._test_data_dir")
    t_dir = "./._test_data_dir"
    with open(os.path.join(t_dir, "__init__.py"), "w") as f:
        f.write(py_string)
    shutil.copy(".data/MNIST/processed/training.pt", t_dir)
    zf = zipfile.ZipFile(".test.zip", "w", zipfile.ZIP_DEFLATED)
    for f in os.listdir(t_dir):
        zf.write(os.path.join(t_dir, f), arcname=f)
    zf.close()
    with open("./.test.zip", "rb") as f:
        response = requests.request("POST", host + "upload_dataset",
                                    files={"file": f},
                                    data={"name": "MNIST",
                                          "description": "MNIST Image dataset",
                                          "type": "image"},
                                    cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    assert result.status
    task_id = result.task_id
    time.sleep(1)
    response = requests.request("GET", host + f"check_task?task_id={task_id}",
                                cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    assert result.status
    assertIn("_dataset_MNIST", daemon._datasets)
    py_string_2 = """
class MnistDataset:
    def __init__(self, pt_file):
        import torch
        data = torch.load(pt_file)

    def __len__(daemon_and_cookies, json_config):
        return len(data[0])

    def __getitem__(self, idx):
        return data[0][idx], data[1][idx]

dataset = MnistDataset("/home/joe/projects/trainer/.data/MNIST/processed/training.pt")
"""
    with open("./.data_file.py", "w") as f:
        f.write(py_string_2)
    with open("./.data_file.py", "rb") as f:
        response = requests.request("POST", host + "upload_dataset",
                                    files={"file": f},
                                    data={"name": "MNIST_2",
                                          "description": "MNIST Image dataset",
                                          "type": "image"},
                                    cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    assert result.status
    task_id = result.task_id
    time.sleep(1)
    response = requests.request("GET", host + f"check_task?task_id={task_id}",
                                cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    assert result.status
    assertIn("_dataset_MNIST_2", daemon._datasets)
    response = requests.request("POST", host + "delete_dataset",
                                data={"name": "MNIST_2"},
                                cookies=cookies)
    time.sleep(.3)
    print(response.content)
    assert "_dataset_MNIST_2" not in daemon._datasets


@pytest.mark.http
def test_daemon_global_modules_available_to_trainer(daemon_and_cookies,
                                                    params_and_iface, json_config):
    class FakeProc:
        def poll(self):
            return None
    daemon, cookies = daemon_and_cookies
    params, iface = params_and_iface
    host = f"http://{daemon.hostname}:{daemon.port}/"
    with open("./._test_py_file.py", "w") as f:
        f.write(mod_string)
    with open("./._test_py_file.py", "rb") as f:
        response = requests.request("POST", host + "add_global_module",
                                    files={"file": f},
                                    data={"name": json.dumps("test_module_a")},
                                    cookies=cookies)
    name, time_str = iface.data_dir.split("/")[-2:]
    daemon._sessions[name] = {}
    daemon._sessions[name]["path"] = os.path.dirname(iface.data_dir)
    daemon._sessions[name]["modules"] = {}
    daemon._sessions[name]["sessions"] = {}
    daemon._sessions[name]["sessions"][time_str] = {}
    daemon._sessions[name]["sessions"][time_str]["config"] = json_config.copy()
    daemon._sessions[name]["sessions"][time_str]["port"] = iface.api_port
    daemon._sessions[name]["sessions"][time_str]["data_dir"] = iface.data_dir
    daemon._sessions[name]["sessions"][time_str]["process"] = FakeProc()
    daemon._refresh_state(name + "/" + time_str)
    iface.trainer.exec_some_string("import _module_test_module_a")
