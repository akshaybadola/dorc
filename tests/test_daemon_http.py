import os
import sys
import time
import json
import shutil
import requests
import unittest
sys.path.append("../")
from trainer.daemon import _start_daemon


class DaemonHTTPTest(unittest.TestCase):
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

    def test_daemon_started(self):
        response = requests.request("GET", self.host)
        self.assertEqual(response.status_code, 404)
        response = requests.request("POST", self.host + "view_session")
        self.assertEqual(json.loads(response.content), "Doesn't do anything")

    def test_create_session_http(self):
        data = {}
        data["name"] = "meh_session"
        with open("_setup.py", "rb") as f:
            response = requests.request("POST", self.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")})
            print(response.content)
        time.sleep(1)
        with open("_setup.py", "rb") as f:
            response = requests.request("POST", self.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")})
            print(response.content)
        time.sleep(1)
        response = requests.request("GET", self.host + "sessions")
        self.assertIsInstance(json.loads(response.content), dict)
        meh = [*json.loads(response.content).keys()][0]
        self.assertTrue("meh_session" in meh)

    def test_unload_session(self):
        response = requests.request("GET", self.host + "sessions")
        self.assertIsInstance(json.loads(response.content), dict)
        meh = [*json.loads(response.content).keys()][0]
        response = requests.request("POST", self.host + "unload_session",
                                    json=json.dumps({"session_key": meh}))
        task_id = json.loads(response.content)["task_id"]
        response = requests.request("GET", self.host + f"check_task?task_id={task_id}")
        self.assertTrue("task_id" in json.loads(response.content))

    def test_load_session(self):
        "First unload then load"
        data = {}
        data["name"] = "meh_session"
        with open("_setup.py", "rb") as f:
            response = requests.request("POST", self.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")})
            print(response.content)
        time.sleep(1)
        with open("_setup.py", "rb") as f:
            response = requests.request("POST", self.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")})
            print(response.content)
        time.sleep(1)
        response = requests.request("GET", self.host + "sessions")

        # NOTE: DEBUG
        # def test():
        #     import sys
        #     sys.path.append("..")
        #     from trainer.trainer import Trainer
        #     if os.path.exists(".meh"):
        #         import shutil
        #         shutil.rmtree(".meh")
        #     os.mkdir(".meh")
        #     from _setup import config
        #     config["dataloader_params"]["train"]["pin_memory"] = False
        #     config["dataloader_params"]["test"]["pin_memory"] = False
        #     trainer = Trainer(**{"data_dir": ".meh", **config})
        #     trainer._init_all()
        #     return trainer
        # trainer = test()

        self.assertIsInstance(json.loads(response.content), dict)
        meh = [*json.loads(response.content).keys()][0]
        response = requests.request("POST", self.host + "unload_session",
                                    json=json.dumps({"session_key": meh}))
        task_id = json.loads(response.content)["task_id"]
        response = requests.request("GET", self.host + f"check_task?task_id={task_id}")
        self.assertTrue("task_id" in json.loads(response.content))
        response = requests.request("POST", self.host + "load_session",
                                    json=json.dumps({"session_key": meh}))
        task_id = json.loads(response.content)["task_id"]
        time.sleep(1)
        response = requests.request("GET", self.host + f"check_task?task_id={task_id}")
        self.assertTrue("task_id" in json.loads(response.content))

    @classmethod
    def shutdown_daemon(cls, host):
        response = requests.request("GET", host + "_shutdown", timeout=2)
        return response

    @classmethod
    def tearDownClass(cls):
        cls.shutdown_daemon(cls.host)
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)


if __name__ == '__main__':
    unittest.main()
