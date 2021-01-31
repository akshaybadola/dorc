import os
import sys
import time
import json
import shutil
import requests
from threading import Thread
import zipfile
import unittest
sys.path.append("../")
from dorc.interfaces import FlaskInterface


def create_module(module_dir, module_files=[]):
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    if not os.path.exists(os.path.join(module_dir, "__init__.py")):
        with open(os.path.join(module_dir, "__init__.py"), "w") as f:
            f.write("")
    for f in module_files:
        shutil.copy(f, module_dir)



class TrainerMethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hostname = "127.0.0.1"
        cls.port = 12321
        cls.data_dir = ".test_dir"
        cls.host = "http://" + ":".join([cls.hostname, str(cls.port)]) + "/"
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        if not os.path.exists(cls.data_dir):
            os.mkdir(cls.data_dir)
        cls.iface = FlaskInterface(cls.hostname, cls.port, cls.data_dir,
                                   no_start=True)
        with open("_setup.py", "rb") as f:
            f_bytes = f.read()
            status, message = cls.iface.create_trainer(f_bytes)
        cls.iface_thread = Thread(target=cls.iface.start)
        create_module(os.path.abspath(os.path.join(cls.data_dir, "global_modules")),
                      [os.path.abspath("../trainer/autoloads.py")])
        sys.path.append(os.path.abspath(cls.data_dir))
        from global_modules import autoloads
        status, message = cls.iface.create_trainer()
        cls.iface_thread = Thread(target=cls.iface.start)
        cls.iface_thread.start()
        time.sleep(1)

    # START: methods
    def test_add_and_set_model(self):
        model_name = "att2in"
        f = open(f"_modules/test_model.py")
        response = requests.post(self.host + "methods/add_model",
                                 files={"file": f})
        f.close()
        result = json.loads(response)
        self.assertTrue(result[0])
        self.assertIn("att2in", result[1])
        self.assertIn("att2in2", result[1])
        response = requests.post(self.host + "methods/set_model", json=model_name)
        self.assertTrue(json.loads(response.content)[0])
        response = requests.get(self.host + "props/active_model")
        self.assertEqual(json.loads(response.content), model_name)

    def test_fetch_preds(self):
        pass

    def test_fetch_image(self):
        pass

    def test_load_image(self):
        pass

    # FIXME: to_ has to be added to a model which doesn't contain it
    def test_load_weights(self):
        model_name = "att2in"
        model_name = "att2in"
        f = open(f"_modules/test_model.py")
        response = requests.post(self.host + "methods/add_model",
                                 files={"file": f})
        f.close()
        data = {}
        data["model_names"] = [model_name]
        f = open(f"_modules/test_model.pth", "rb")
        response = requests.request("POST", self.host + "methods/load_weights",
                                    files={"file": f},
                                    data={"model_names": json.dumps([model_name])})
        f.close()

    def test_add_user_func(self):
        response = requests.post(f"{self.host}" + "methods/add_user_funcs",
                                 files={"file": open("modules/test_funcs.py")})
        print(response.content)

    def test_add_module(self):
        # This is actually the same as add_model as that calls add_module internally
        pass

    def test_hack_param(self):
        response = requests.post("http://localhost:20202/methods/hack_param",
                                 json={"_hooks_run_iter_frequency":
                                       {"type": "int", "value": "1000"}})
        import ipdb; ipdb.set_trace()

    # START: extras
    def test_adhoc_eval(self):
        response = requests.post(f"{self.host}" + "extras/adhoc_eval",
                                 json={"val": {"data": "val",
                                               "metrics": ["loss"],
                                               "epoch": "current",
                                               "num_or_fraction": 50}})
        print(response.content)

    def test_call_user_func(self):
        response = requests.request("POST", "http://localhost:20202/extras/call_user_func",
                                    json=["model_sample"])
        print(response.content)

    # FIXME: This is a custom callback infact
    def test_report_adhoc_run():
        response = requests.request("POST", "http://localhost:20202/extras/report_adhoc_run",
                                    json={"report_function": "ruotianlou_report_func_2"})
        import ipdb; ipdb.set_trace()


    # def check_image(self):
    #     img_dir = "/home/joe/lib/docprocess/images_pdfs/"
    #     img_dir = "/datacache/coco/test2014/"
    #     f = os.path.join(img_dir,
    #                      random.choice([x for x in os.listdir(img_dir)
    #                                     if x.endswith("png") or x.endswith("jpg")]))
    #     print(f)
    #     Thread(target=Popen, args=[["qiv", f]]).run()
    #     requests.request("POST", "http://localhost:20202/_helpers/load_image",
    #                      files={"file": open(os.path.join(img_dir, f), "rb")},
    #                      data={"callbacks": json.dumps(["model_sample_image",
    #                                                     "get_features_from_image_data"])})
    #     # for f in os.listdir(img_dir):
    #     #     if f.endswith("png"):
    #     #         requests.request("POST", "http://localhost:20202/_helpers/load_image",
    #     #                          files={"file": open(os.path.join(img_dir, f), "rb")},
    #     #                          data={"callbacks": json.dumps(["model_sample_image",
    #     #                                                         "get_features_from_image_data"])})


