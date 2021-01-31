import os
import shutil
import unittest
import sys
import time
sys.path.append("../")
from dorc.daemon import Daemon


class DaemonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ".test_dir"
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        if not os.path.exists(cls.data_dir):
            os.mkdir(cls.data_dir)
        cls.port = 23232
        cls.hostname = "127.0.0.1"
        cls.daemon = Daemon(cls.hostname, cls.port, ".test_dir")

    def test_find_port(self):
        port = self.daemon._find_open_port()
        self.assertTrue(port)

    def _create_session(self):
        with open("_setup.py", "rb") as f:
            meh = f.read()
        data = {"name": "test_session", "config": meh}
        task_id = self.daemon._get_task_id_launch_func(self.daemon.create_session, data)
        time.sleep(1)
        return self.daemon._check_result(task_id)[1]

    def test_create_session(self):
        "create session with existing name but different timestamp"
        # Should be able to create session on top of scanned and loaded sessions
        # Should also create session with different timestamp
        self.assertTrue(self._create_session())

    def test_scan_sessions(self):
        self._create_session()
        self._create_session()
        self._create_session()  # create three more sessions
        self.terminate_live_sessions()
        self.daemon._sessions = {}
        self.daemon.scan_sessions()
        for k, v in self.daemon._sessions["test_session"]["sessions"].items():
            with self.subTest(i=str(k)):
                self.assertIsInstance(v, dict)
                self.assertFalse("process" in v)

    @classmethod
    def terminate_live_sessions(cls):
        for s in cls.daemon._sessions.values():
            for s_name in s["sessions"]:
                if "process" in s["sessions"][s_name]:
                    s["sessions"][s_name]["process"].terminate()
                    print(f'Terminated {s["sessions"][s_name]["process"]}')

    @classmethod
    def tearDownClass(cls):
        cls.daemon._fwd_ports_thread.kill()
        cls.daemon.stop()
        cls.terminate_live_sessions()
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)


if __name__ == '__main__':
    unittest.main()
