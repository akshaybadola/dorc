import random
import pytest
import os
import time
import json
import requests
from threading import Thread
import zipfile
from subprocess import Popen, PIPE
from dorc.interfaces import FlaskInterface


# def create_module(module_dir, module_files=[]):
#     if not os.path.exists(module_dir):
#         os.mkdir(module_dir)
#     if not os.path.exists(os.path.join(module_dir, "__init__.py")):
#         with open(os.path.join(module_dir, "__init__.py"), "w") as f:
#             f.write("")
#     for f in module_files:
#         shutil.copy(f, module_dir)



# class TrainerMethodsTest(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.hostname = "127.0.0.1"
#         cls.port = 12321
#         cls.data_dir = ".test_dir"
#         cls.host = "http://" + ":".join([cls.hostname, str(cls.port)]) + "/"
#         if os.path.exists(cls.data_dir):
#             shutil.rmtree(cls.data_dir)
#         if not os.path.exists(cls.data_dir):
#             os.mkdir(cls.data_dir)
#         cls.iface = FlaskInterface(cls.hostname, cls.port, cls.data_dir,
#                                    no_start=True)
#         with open("_setup.py", "rb") as f:
#             f_bytes = f.read()
#             status, message = cls.iface.create_trainer(f_bytes)
#         cls.iface_thread = Thread(target=cls.iface.start)
#         create_module(os.path.abspath(os.path.join(cls.data_dir, "global_modules")),
#                       [os.path.abspath("../trainer/autoloads.py")])
#         sys.path.append(os.path.abspath(cls.data_dir))
#         from global_modules import autoloads
#         status, message = cls.iface.create_trainer()
#         cls.iface_thread = Thread(target=cls.iface.start)
#         cls.iface_thread.start()
#         time.sleep(1)


@pytest.mark.todo
@pytest.mark.http
def test_add_and_set_model(params_and_iface):
    params, iface = params_and_iface
    host = params["host"]
    model_name = "att2in"
    f = open(f"_modules/test_model.py")
    response = requests.post(host + "methods/add_model",
                             files={"file": f})
    f.close()
    result = json.loads(response)
    assert result[0]
    assert "att2in" in result[1]
    assert "att2in2" in result[1]
    response = requests.post(host + "methods/set_model", json=model_name)
    assert json.loads(response.content[0])
    response = requests.get(host + "props/active_model")
    assert json.loads(response.content) == model_name


@pytest.mark.todo
def test_fetch_preds(params_and_iface):
    pass


@pytest.mark.todo
def test_fetch_image(params_and_iface):
    pass


@pytest.mark.todo
def test_load_image(params_and_iface):
    pass


@pytest.mark.http
@pytest.mark.todo
def test_load_weights(params_and_iface):
    params, iface = params_and_iface
    host = params["host"]
    model_name = "att2in"
    model_name = "att2in"
    f = open(f"_modules/test_model.py")
    response = requests.post(host + "methods/add_model",
                             files={"file": f})
    f.close()
    data = {}
    data["model_names"] = [model_name]
    f = open(f"_modules/test_model.pth", "rb")
    response = requests.request("POST", host + "methods/load_weights",
                                files={"file": f},
                                data={"model_names": json.dumps([model_name])})
    f.close()


@pytest.mark.todo
def test_add_user_func(params_and_iface):
    params, iface = params_and_iface
    host = params["host"]
    response = requests.post(host + "methods/add_user_funcs",
                             files={"file": open("modules/test_funcs.py")})
    print(response.content)


@pytest.mark.todo
def test_add_module(params_and_iface):
    # This is actually the same as add_model as that calls add_module internally
    pass


@pytest.mark.todo
def test_hack_param(params_and_iface):
    params, iface = params_and_iface
    host = params["host"]
    response = requests.post(host + "methods/hack_param",
                             json={"_hooks_run_iter_frequency":
                                   {"type": "int", "value": "1000"}})


@pytest.mark.todo
def test_adhoc_eval(params_and_iface):
    params, iface = params_and_iface
    host = params["host"]
    response = requests.post(host + "extras/adhoc_eval",
                             json={"val": {"data": "val",
                                           "metrics": ["loss"],
                                           "epoch": "current",
                                           "num_or_fraction": 50}})
    print(response.content)


@pytest.mark.todo
def test_call_user_func(params_and_iface):
    params, iface = params_and_iface
    host = params["host"]
    response = requests.post(host + "extras/call_user_func",
                             json=["model_sample"])
    print(response.content)


# FIXME: This is a custom callback infact
@pytest.mark.todo
def test_report_adhoc_run(params_and_iface):
    params, iface = params_and_iface
    host = params["host"]
    response = requests.request("POST", host + "methods/report_adhoc_run",
                                json={"report_function": "ruotianlou_report_func_2"})


@pytest.mark.todo
def check_image(params_and_iface):
    params, iface = params_and_iface
    host = params["host"]
    img_dir = "/home/joe/lib/docprocess/images_pdfs/"
    img_dir = "/datacache/coco/test2014/"
    f = os.path.join(img_dir,
                     random.choice([x for x in os.listdir(img_dir)
                                    if x.endswith("png") or x.endswith("jpg")]))
    print(f)
    Thread(target=Popen, args=[["qiv", f]]).run()
    requests.post(host + "helpers/load_image",
                  files={"file": open(os.path.join(img_dir, f), "rb")},
                  data={"callbacks": json.dumps(["model_sample_image",
                                                 "get_features_from_image_data"])})
    # for f in os.listdir(img_dir):
    #     if f.endswith("png"):
    #         requests.request("POST", "http://localhost:20202/_helpers/load_image",
    #                          files={"file": open(os.path.join(img_dir, f), "rb")},
    #                          data={"callbacks": json.dumps(["model_sample_image",
    #                                                         "get_features_from_image_data"])})
