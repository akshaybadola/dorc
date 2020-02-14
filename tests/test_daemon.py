import os
import shutil
import unittest
import sys
import requests
import time
import json
from _setup import config
from multiprocessing import Process
sys.path.append("../")
from trainer.interfaces import FlaskInterface
from trainer.trainer import Trainer
from trainer.daemon import create_daemon


class DaemonTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = ".test_dir"
        self.daemon = create_daemon(True, {"port": 23232, "data_dir": ".test_dir"})
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

    def test_daemon_start(self):
        p = Process(target=self.daemon.start)
        p.start()
        time.sleep(.2)
        host = "http://" + ":".join([self.daemon.hostname, str(self.daemon.port) + "/"])
        response = requests.request("GET", host)
        self.assertEqual(response.status_code, 404)
        response = requests.request("POST", host + "view_session")
        self.assertEqual(json.loads(response.content), "Doesn't do anything")
        p.terminate()

    def test_scan_sessions(self):
        # create a couple of session_dirs programmatically and it should scan
        # NOTE: config has to be eval code if it can be dumped.
        #       Or pickle. I'm not a big fan of pickle.
        import ipdb; ipdb.set_trace()
        pass

    # def test_create_session(self):
    #     "create session with existing name but different timestamp"
    #     # Should be able to create session on top of scanned and loaded sessions
    #     # Should also create session with different timestamp
    #     # session_name + timestamp then becomes equivalent to trainer.uid
    #     pass

    def tearDown(self):
        if hasattr(self, "test_dir"):
            shutil.rmtree(self.test_dir)

    # def test_list_sessions(self):
    #     pass

    # def test_destroy_session(self):
    #     pass

    # def test_auth(self):
    #     "auth is not yet available"
    #     pass


if __name__ == '__main__':
    unittest.main()
