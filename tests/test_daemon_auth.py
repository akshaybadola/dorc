import os
import random
import time
import shutil
import requests
import unittest
import pytest
from dorc.util import make_test_daemon


@pytest.mark.http
class DaemonHTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ".test_auth_dir"
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        if not os.path.exists(cls.data_dir):
            os.mkdir(cls.data_dir)
        cls.port = random.randint(24432, 25000)
        cls.hostname = "127.0.0.1"
        cls.daemon = make_test_daemon(cls.hostname, cls.port, cls.data_dir)
        cls.host = "http://" + ":".join([cls.hostname, str(cls.port) + "/"])
        time.sleep(.5)

    def test_sessions_view(self):
        response = requests.request("GET", self.host + "sessions", allow_redirects=False)
        self.assertEqual(response.status_code, 401)
        self.assertIn("not", response.content.decode().lower())
        self.assertIn("authorized", response.content.decode().lower())
        response = requests.request("POST", self.host + "login",
                                    json={"username": "admin", "password": "admin"})
        self.assertEqual(response.status_code, 401)
        response = requests.request("POST", self.host + "login",
                                    json={"username": "admin", "password": "AdminAdmin_33"})
        self.assertEqual(response.status_code, 200)
        cookies = response.cookies
        response = requests.request("GET", self.host + "sessions", allow_redirects=False,
                                    cookies=cookies)
        self.assertEqual(response.status_code, 200)

    @classmethod
    def shutdown_daemon(cls, host):
        cookies = requests.request("POST", host + "login",
                                   json={"username": "admin",
                                         "password": "AdminAdmin_33"}).cookies
        response = requests.request("GET", host + "_shutdown", cookies=cookies, timeout=2)
        return response

    @classmethod
    def tearDownClass(cls):
        cls.shutdown_daemon(cls.host)
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)


if __name__ == '__main__':
    unittest.main()
