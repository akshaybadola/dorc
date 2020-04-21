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

    def test_sessions_view(self):
        response = requests.request("GET", self.host + "sessions", allow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertTrue("/login" in response.content.decode())
        response = requests.request("POST", self.host + "login",
                                    data={"username": "admin", "password": "admin"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(json.loads(response.content)[0])
        response = requests.request("POST", self.host + "login",
                                    data={"username": "admin", "password": "AdminAdmin_33"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(json.loads(response.content)[0])
        cookies = response.cookies
        response = requests.request("GET", self.host + "sessions", allow_redirects=False,
                                    cookies=cookies)
        self.assertEqual(response.status_code, 200)

    @classmethod
    def shutdown_daemon(cls, host):
        cookies = requests.request("POST", host + "login",
                                   data={"username": "admin",
                                         "password": "AdminAdmin_33"}).cookies
        response = requests.request("GET", host + "_shutdown", cookies=cookies, timeout=2)
        return response

    @classmethod
    def tearDownClass(cls):
        cls.daemon._fwd_ports_thread.kill()
        cls.shutdown_daemon(cls.host)
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)


if __name__ == '__main__':
    unittest.main()
