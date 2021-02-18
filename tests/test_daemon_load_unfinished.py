import pytest
import os
import shutil
import unittest
import sys
import requests
import time
import json
sys.path.append("../")
from dorc.util import make_test_daemon
from dorc.daemon.models import SessionMethodResponseModel as smod


@pytest.mark.http
class DaemonHTTPTestLoadUnfinished(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ".test_dir"
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        if not os.path.exists(cls.data_dir):
            os.mkdir(cls.data_dir)
        cls.port = 23244
        cls.hostname = "127.0.0.1"
        cls.host = "http://" + ":".join([cls.hostname, str(cls.port) + "/"])
        cls.daemon = make_test_daemon(cls.hostname, cls.port, ".test_dir")
        time.sleep(.5)
        cls.cookies = requests.request("POST", cls.host + "login",
                                       data={"username": "joe", "password": "Monkey$20"}).cookies


    def test_load_unfinished_sessions(self):
        # """Restart daemon without removing all directories. Make sure unfinished
        # sessions are loaded

        # """
        # Create a couple of sessions
        from _setup import config
        data = {}
        data["name"] = "meh_session"
        data["config"] = config
        responses = []
        response = requests.request("POST", self.host + "create_session",
                                    json=data,
                                    cookies=self.cookies)
        responses.append(response)
        time.sleep(1)
        response = requests.request("POST", self.host + "create_session",
                                    json=data,
                                    cookies=self.cookies)
        responses.append(response)
        time.sleep(1)
        response = requests.get(self.host + "sessions", cookies=self.cookies)
        sessions = json.loads(response.content)
        assert sessions
        # shutdown the daemon
        response = self.shutdown_daemon(self.host)
        self.assertIn("shutting down", str(response.content).lower())
        time.sleep(.5)
        key = [*sessions.keys()][0]
        data_dir = os.path.join(self.daemon.root_dir, key)
        with open(os.path.join(data_dir, "session_state"), "r") as f:
            state = json.load(f)
        # switch state to finished
        state["epoch"] = state["max_epochs"]
        with open(os.path.join(data_dir, "session_state"), "w") as f:
            json.dump(state, f)
        # start new daemon
        daemon = make_test_daemon(self.hostname, self.port + 5, ".test_dir", no_clear=True)
        host = "http://" + ":".join([self.hostname, str(self.port + 5) + "/"])
        for _ in range(3):
            time.sleep(1)
            try:
                response = requests.get(host + "sessions", cookies=self.cookies)
                break
            except Exception:
                print(f"Checking again for {host}")
        sessions = json.loads(response.content)
        self.assertTrue(sessions)
        self.assertIn("loaded", sessions[key])
        self.assertFalse(sessions[key]["loaded"])
        responses = []
        for m in sessions.keys():
            responses.append(requests.get(host + f"purge_session?session_key={m}",
                                          cookies=self.cookies))
        # Give it some time to purge
        time.sleep(1)
        for r in responses:
            task_id = smod.parse_obj(json.loads(r.content)).task_id
            with self.subTest(i=str(task_id)):
                content = requests.request("GET", host + f"check_task?task_id={task_id}",
                                           cookies=self.cookies).content
                self.assertTrue(smod.parse_obj(json.loads(content)).status)
        self.shutdown_daemon(host)
        time.sleep(1)

    @classmethod
    def shutdown_daemon(cls, host):
        response = requests.request("GET", host + "_shutdown", timeout=2,
                                    cookies=cls.cookies)
        return response


if __name__ == '__main__':
    unittest.main()
