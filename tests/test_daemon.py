import os
import shutil
import unittest
import sys
import requests
import time
import json
from multiprocessing import Process
sys.path.append("../")
from trainer.interfaces import FlaskInterface
from trainer.trainer import Trainer
from trainer.daemon import create_daemon


class DaemonTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = ".test_dir"
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
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

    def test_find_port(self):
        port = self.daemon._find_open_port()
        self.assertTrue(port)

    def test_create_session(self):
        "create session with existing name but different timestamp"
        # Should be able to create session on top of scanned and loaded sessions
        # Should also create session with different timestamp
        with open("_setup.py", "rb") as f:
            meh = f.read()
        data = {"name": "test_session", "config": meh}
        task_id = self.daemon._get_task_id_launch_func(self.daemon.create_session, data)
        time.sleep(1)
        self.assertTupleEqual(self.daemon._check_result(task_id), (1, True))

    # def test_scan_sessions(self):
    #     # create a couple of session_dirs programmatically and it should scan
    #     # NOTE: config has to be eval code if it can be dumped.
    #     #       Or pickle. I'm not a big fan of pickle.
    #     import ipdb; ipdb.set_trace()
    #     pass

    def tearDown(self):
        if hasattr(self, "test_dir"):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        for s in self.daemon._sessions.values():
            for s_name in s["sessions"]:
                if "process" in s["sessions"][s_name]:
                    s["sessions"][s_name]["process"].terminate()
                    print(f'Terminated {s["sessions"][s_name]["process"]}')

    # def test_list_sessions(self):
    #     pass

    # def test_destroy_session(self):
    #     pass

    # def test_auth(self):
    #     "auth is not yet available"
    #     pass


if __name__ == '__main__':
    unittest.main()
