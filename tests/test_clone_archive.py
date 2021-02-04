import pprint
import unittest
import sys
import os
import shutil
import time
import json
import requests
import zipfile
sys.path.append("../")
from dorc.daemon import _start_daemon


class CloneHTTPTest(unittest.TestCase):
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

    # NOTE: Now tests for cloning zip file also
    def test_clone(self):
        data = {}
        data["name"] = "meh_session"
        responses = []
        with open("_setup_local.py", "rb") as f:
            response = requests.request("POST", self.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")},
                                        cookies=self.cookies)
            responses.append(response)
        time.sleep(1)
        zf = zipfile.ZipFile("setup_local.zip", "w")
        zf.write("_setup_local.py", arcname="__init__.py")
        zf.close()
        with open("setup_local.zip", "rb") as f:
            response = requests.request("POST", self.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")},
                                        cookies=self.cookies)
            responses.append(response)
        time.sleep(1)
        response = requests.request("GET", self.host + "sessions",
                                    cookies=self.cookies)
        status, sessions = json.loads(response.content)
        key_py = [*sessions.keys()][0]
        key_zip = [*sessions.keys()][1]
        config = {":".join(["optimizer", "Adam", "params", "lr"]): 0.05,
                  "uid": "test_monkey_trainer",
                  ":".join(["trainer_params", "seed"]): 2222}
        response = requests.request("POST", self.host + "clone_session",
                                    json={"session_key": key_py, "config": config},
                                    cookies=self.cookies)
        time.sleep(.5)
        status, resp = json.loads(response.content)
        task_id = resp["task_id"]
        pprint.pprint(requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                       cookies=self.cookies).content)
        keys = [*self.daemon._sessions["meh_session"]["sessions"].keys()]
        keys.sort()
        response = requests.request("POST", self.host + "load_session",
                                    json={"session_key": "meh_session/" + keys[-1]},
                                    cookies=self.cookies)
        time.sleep(1)
        status, resp = json.loads(response.content)
        task_id = resp["task_id"]
        pprint.pprint(requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                       cookies=self.cookies).content)
        port = self.daemon._sessions["meh_session"]["sessions"][keys[-1]]["port"]
        response = requests.request("GET", f"http://{self.hostname}:{port}/props/all_params")
        bleh = json.loads(response.content)
        self.assertEqual(bleh["trainer_params"]["seed"], 2222)
        config = {":".join(["optimizer", "Adam", "params", "lr"]): 0.05,
                  "uid": "test_monkey_trainer",
                  ":".join(["trainer_params", "seed"]): 3333}
        # NOTE: test_zip
        response = requests.request("POST", self.host + "clone_session",
                                    json={"session_key": key_zip, "config": config},
                                    cookies=self.cookies)
        time.sleep(.5)
        status, resp = json.loads(response.content)
        task_id = resp["task_id"]
        pprint.pprint(requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                       cookies=self.cookies).content)
        keys = [*self.daemon._sessions["meh_session"]["sessions"].keys()]
        keys.sort()
        response = requests.request("POST", self.host + "load_session",
                                    json={"session_key": "meh_session/" + keys[-1]},
                                    cookies=self.cookies)
        time.sleep(1)
        status, resp = json.loads(response.content)
        task_id = resp["task_id"]
        pprint.pprint(requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                       cookies=self.cookies).content)
        port = self.daemon._sessions["meh_session"]["sessions"][keys[-1]]["port"]
        response = requests.request("GET", f"http://{self.hostname}:{port}/props/all_params")
        bleh = json.loads(response.content)
        self.assertEqual(bleh["trainer_params"]["seed"], 3333)

    def test_clone_to(self):
        new_port = 23234
        new_hostname = "127.0.44.1"
        new_daemon = _start_daemon(new_hostname, new_port, ".new_test_dir")
        new_host = "http://" + ":".join([new_hostname, str(new_port) + "/"])
        time.sleep(.5)
        new_cookies = requests.request("POST", new_host + "login",
                                       data={"username": "joe",
                                             "password": "Monkey$20"}).cookies
        # NOTE: Init in new daemon
        data = {}
        data["name"] = "meh_session"
        responses = []
        with open("_setup_local.py", "rb") as f:
            response = requests.request("POST", new_host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")},
                                        cookies=new_cookies)
            responses.append(response)
        time.sleep(1)
        zf = zipfile.ZipFile("setup_local.zip", "w")
        zf.write("_setup_local.py", arcname="__init__.py")
        zf.close()
        with open("setup_local.zip", "rb") as f:
            response = requests.request("POST", new_host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")},
                                        cookies=new_cookies)
            responses.append(response)
        time.sleep(1)
        response = requests.request("GET", new_host + "sessions",
                                    cookies=new_cookies)
        status, sessions = json.loads(response.content)
        key = [*sessions.keys()][1]
        config = {":".join(["optimizer", "Adam", "params", "lr"]): 0.05,
                  "uid": "test_monkey_trainer",
                  ":".join(["trainer_params", "seed"]): 2222}
        shutil.copy("_forward.py", os.path.join(new_daemon.data_dir, key, "savedir"))
        response = requests.request("POST", new_host + "clone_to",
                                    json={"session_key": key,
                                          "saves": True,
                                          "server": f"{self.hostname}:{self.port}",
                                          "config": config},
                                    cookies=new_cookies)
        time.sleep(.5)
        requests.request("GET", new_host + "_shutdown", timeout=2,
                         cookies=new_cookies)
        shutil.rmtree(".new_test_dir")

    @classmethod
    def shutdown_daemon(cls, host):
        response = requests.request("GET", host + "_shutdown", timeout=2,
                                    cookies=cls.cookies)
        return response

    @classmethod
    def tearDownClass(cls):
        cls.shutdown_daemon(cls.host)
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        if os.path.exists("setup_local.zip"):
            os.remove("setup_local.zip")


if __name__ == "__main__":
    unittest.main()
