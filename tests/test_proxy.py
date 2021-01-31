import unittest
import sys
import os
import shutil
import time
import json
import requests
sys.path.append("../")
from dorc.daemon import _start_daemon


class ProxyTest(unittest.TestCase):
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

    def test_get_methods(self):
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


if __name__ == "__main__":
    unittest.main()
