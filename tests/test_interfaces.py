import os
import sys
import shutil
import requests
import unittest
import time
from threading import Thread
sys.path.append("../")
from trainer.interfaces import FlaskInterface


class InterfaceTest(unittest.TestCase):
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
        cls.iface = FlaskInterface(cls.hostname, cls.port, cls.data_dir)
        with open("_setup.py", "rb") as f:
            f_bytes = f.read()
            status, message = cls.iface.create_trainer(f_bytes)
        Thread(target=cls.iface.start).start()

    def test_iface_init(self):
        response = requests.request("GET", self.host + "start")
        self.assertTrue("start" in response.content.decode().lower())
        time.sleep(.5)
        response = requests.request("GET", self.host + "pause")
        self.assertTrue("pausing" in response.content.decode().lower())
        time.sleep(.5)
        self.assertTrue(self.iface.trainer.paused)
        response = requests.request("GET", self.host + "resume")
        self.assertTrue("resuming" in response.content.decode().lower())
        time.sleep(.5)
        self.assertFalse(self.iface.trainer.paused)
        self.iface.trainer.abort_loop()

    # def test_iface_shutdown(self):
    #     response = requests.request("GET", self.host + "abort_loop")
    #     print(response.content)
    #     self.assertTrue("aborted" in response.content.decode().lower())
    #     time.sleep(.5)

    # def test_trainer_crash(self):
    #     pass

    # def test_get(self):
    #     pass

    # def test_post(self):
    #     pass

    # def test_post_form(self):
    #     pass

    # def test_route(self):
    #     pass

    # def test_auth(self):
    #     pass

    # def test_helpers(self):
    #     pass

    # def test_props(self):
    #     pass

    # def test_extras(self):
    #     pass

    @classmethod
    def tearDownClass(cls):
        cls.iface.trainer.abort_loop()
        response = requests.request("GET", cls.host + "_shutdown")
        print(response.content)
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)


if __name__ == '__main__':
    unittest.main()
