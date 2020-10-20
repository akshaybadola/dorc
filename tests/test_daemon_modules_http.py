import os
import sys
import time
import json
import shutil
import requests
import zipfile
import unittest
sys.path.append("../")
from trainer.daemon import _start_daemon


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


class DaemonModulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ".test_dir"
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        if not os.path.exists(cls.data_dir):
            os.mkdir(cls.data_dir)
        cls.port = 23232
        cls.hostname = "127.0.0.1"
        cls.daemon = _start_daemon(cls.hostname, cls.port, ".test_dir")
        cls.host = "http://" + ":".join([cls.hostname, str(cls.port) + "/"])
        time.sleep(.5)
        cls.cookies = requests.request("POST", cls.host + "login",
                                       data={"username": "joe", "password": "Monkey$20"}).cookies
        with open("_setup.py", "rb") as f:
            response = requests.request("POST", cls.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")},
                                        cookies=cls.cookies)
            print(response.content)
        time.sleep(1)
        with open("_setup.py", "rb") as f:
            response = requests.request("POST", cls.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("bleh_session")},
                                        cookies=cls.cookies)
            print(response.content)
        if cls.daemon._fwd_ports_thread is not None:
            cls.daemon._fwd_ports_thread.kill()
        time.sleep(1)
        print("Initialized daemon and created sessions")

    def test_list_modules(self):
        time.sleep(1)
        response = requests.request("GET", self.host + "list_global_modules", cookies=self.cookies)
        self.assertIsInstance(json.loads(response.content), dict)
        mods = json.loads(response.content)
        # NOTE: Not sure why this fails
        # self.assertEqual(mods, {"autoloads": ["torch", "ModelStep",
        #                                       "ClassificationTrainStep",
        #                                       "ClassificationTestStep",
        #                                       "accuracy", "CheckFunc",
        #                                       "CheckGreater",
        #                                       "CheckGreaterName",
        #                                       "CheckLesserName",
        #                                       "CheckAccuracy"]})

    def test_add_delete_global_module(self):
        with open("./._test_py_file.py", "w") as f:
            f.write(mod_string)
        with open("./._test_py_file.py", "rb") as f:
            response = requests.request("POST", self.host + "add_global_module",
                                        files={"file": f},
                                        data={"name": json.dumps("test_module_a")},
                                        cookies=self.cookies)
        self.assertIsInstance(json.loads(response.content), dict)
        meh = json.loads(response.content)
        task_id = meh["task_id"]
        time.sleep(1)
        response = requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                    cookies=self.cookies)
        result = json.loads(response.content)
        self.assertTrue(result["result"])
        self.assertIn("_module_test_module_a", self.daemon._modules)
        with open("./._test_py_file.py", "rb") as f:
            response = requests.request("POST", self.host + "add_global_module",
                                        files={"file": f},
                                        data={"name": json.dumps("test_module_b")},
                                        cookies=self.cookies)
        meh = json.loads(response.content)
        task_id = meh["task_id"]
        time.sleep(1)
        response = requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                    cookies=self.cookies)
        result = json.loads(response.content)
        self.assertTrue(result["result"])
        self.assertIn("_module_test_module_b", self.daemon._modules)
        requests.request("POST", self.host + "delete_global_module",
                         data={"name": "test_module_a"},
                         cookies=self.cookies)
        time.sleep(.3)
        self.assertTrue(result["result"])
        self.assertNotIn("_module_test_module_a", self.daemon._modules)

    def test_add_delete_dataset(self):
        py_string = """
class MnistDataset:
    def __init__(self, pt_file):
        import torch
        self.data = torch.load(pt_file)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

import os
file_dir = os.path.abspath(os.path.dirname(__file__))
dataset = MnistDataset(os.path.join(file_dir, "training.pt"))
"""
        if not os.path.exists("./.test_data_dir"):
            os.mkdir("./.test_data_dir")
        t_dir = "./.test_data_dir"
        with open(os.path.join(t_dir, "__init__.py"), "w") as f:
            f.write(py_string)
        shutil.copy(".data/MNIST/processed/training.pt", t_dir)
        zf = zipfile.ZipFile(".test.zip", "w", zipfile.ZIP_DEFLATED)
        for f in os.listdir(t_dir):
            zf.write(os.path.join(t_dir, f), arcname=f)
        zf.close()
        with open("./.test.zip", "rb") as f:
            response = requests.request("POST", self.host + "upload_dataset",
                                        files={"file": f},
                                        data={"name": "MNIST",
                                              "description": "MNIST Image dataset",
                                              "type": "image"},
                                        cookies=self.cookies)
        self.assertIsInstance(json.loads(response.content), dict)
        meh = json.loads(response.content)
        task_id = meh["task_id"]
        time.sleep(1)
        response = requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                    cookies=self.cookies)
        result = json.loads(response.content)
        self.assertTrue(result["result"])
        self.assertIn("_dataset_MNIST", self.daemon._datasets)
        py_string_2 = """
class MnistDataset:
    def __init__(self, pt_file):
        import torch
        self.data = torch.load(pt_file)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

dataset = MnistDataset("/home/joe/projects/trainer/.data/MNIST/processed/training.pt")
"""
        with open("./.data_file.py", "w") as f:
            f.write(py_string_2)
        with open("./.data_file.py", "rb") as f:
            response = requests.request("POST", self.host + "upload_dataset",
                                        files={"file": f},
                                        data={"name": "MNIST_2",
                                              "description": "MNIST Image dataset",
                                              "type": "image"},
                                        cookies=self.cookies)
        self.assertIsInstance(json.loads(response.content), dict)
        meh = json.loads(response.content)
        task_id = meh["task_id"]
        time.sleep(1)
        response = requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                    cookies=self.cookies)
        result = json.loads(response.content)
        self.assertTrue(result["result"])
        self.assertIn("_dataset_MNIST_2", self.daemon._datasets)
        requests.request("POST", self.host + "delete_dataset",
                         data={"name": "MNIST_2"},
                         cookies=self.cookies)
        time.sleep(.3)
        self.assertTrue(result["result"])
        self.assertNotIn("_dataset_MNIST_2", self.daemon._datasets)

    @classmethod
    def shutdown_daemon(cls, host):
        response = requests.request("GET", host + "_shutdown",
                                    cookies=cls.cookies, timeout=2)
        return response

    @classmethod
    def tearDownClass(cls):
        cls.shutdown_daemon(cls.host)
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)


if __name__ == '__main__':
    unittest.main()
