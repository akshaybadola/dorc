import pprint
import unittest
import sys
import os
import shutil
import time
import json
import requests
sys.path.append("../")
from trainer.daemon import _start_daemon


class CloneArchiveHTTPTest(unittest.TestCase):
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
        with open("_setup_local.py", "rb") as f:
            response = requests.request("POST", self.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")},
                                        cookies=self.cookies)
            responses.append(response)
        time.sleep(1)
        response = requests.request("GET", self.host + "sessions",
                                    cookies=self.cookies)
        sessions = json.loads(response.content)
        key = [*sessions.keys()][0]
        config = [["optimizer", "Adam", "params", "lr", 0.05],
                  ["uid", "test_monkey_trainer"],
                  ["trainer_params", "seed", 2222]]
        response = requests.request("POST", self.host + "clone_session",
                                    json={"session_key": key, "config": config},
                                    cookies=self.cookies)
        time.sleep(.5)
        task_id = json.loads(response.content)["task_id"]
        pprint.pprint(requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                       cookies=self.cookies).content)
        import ipdb; ipdb.set_trace()
        keys = [*self.daemon._sessions["meh_session"]["sessions"].keys()]
        keys.sort()
        response = requests.request("POST", self.host + "load_session",
                                    json={"session_key": "meh_session/" + keys[-1]},
                                    cookies=self.cookies)
        time.sleep(1)
        task_id = json.loads(response.content)["task_id"]
        pprint.pprint(requests.request("GET", self.host + f"check_task?task_id={task_id}",
                                       cookies=self.cookies).content)
        import ipdb; ipdb.set_trace()

    @classmethod
    def shutdown_daemon(cls, host):
        response = requests.request("GET", host + "_shutdown", timeout=2,
                                    cookies=cls.cookies)
        return response

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)


if __name__ == "__main__":
    unittest.main()
