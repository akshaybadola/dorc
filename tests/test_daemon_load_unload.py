import os
import shutil
import unittest
import sys
import requests
import time
import json
sys.path.append("../")
from dorc.util import make_test_daemon


class DaemonHTTPTestLoadUnload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ".test_dir"
        if os.path.exists(cls.data_dir):
            shutil.rmtree(cls.data_dir)
        if not os.path.exists(cls.data_dir):
            os.mkdir(cls.data_dir)
        cls.port = 23232
        cls.hostname = "127.0.0.1"
        cls.cookies = requests.request("POST", cls.host + "login",
                                       data={"username": "joe", "password": "Monkey$20"}).cookies
        cls.host = "http://" + ":".join([cls.hostname, str(cls.port) + "/"])
        time.sleep(.5)

    def test_load_unfinished_sessions(self):
        # """Restart daemon without removing all directories. Make sure unfinished
        # sessions are loaded

        # """
        # Create a couple of sessions
        data = {}
        data["name"] = "meh_session"
        responses = []
        with open("_setup.py", "rb") as f:
            response = requests.request("POST", self.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")},
                                        cookies=self.cookies)
            responses.append(response)
        time.sleep(1)
        with open("_setup.py", "rb") as f:
            response = requests.request("POST", self.host + "create_session",
                                        files={"file": f},
                                        data={"name": json.dumps("meh_session")},
                                        cookies=self.cookies)
            responses.append(response)
        time.sleep(1)
        response = requests.request("GET", self.host + "sessions",
                                    cookies=self.cookies)
        status, sessions = json.loads(response.content)
        self.assertTrue(status)
        # shutdown the daemon
        response = self.shutdown_daemon(self.host)
        self.assertIn("shutting down", str(response.content).lower())
        time.sleep(.5)
        key = [*sessions.keys()][0]
        data_dir = os.path.join(self.daemon.data_dir, key)
        with open(os.path.join(data_dir, "session_state"), "r") as f:
            state = json.load(f)
        # switch state to finished
        state["epoch"] = state["trainer_params"]["max_epochs"]
        with open(os.path.join(data_dir, "session_state"), "w") as f:
            json.dump(state, f)
        # start new daemon
        daemon = make_test_daemon(self.hostname, self.port + 5, ".test_dir")
        if daemon._fwd_ports_thread is not None:
            daemon._fwd_ports_thread.kill()
        host = "http://" + ":".join([self.hostname, str(self.port + 5) + "/"])
        for _ in range(3):
            time.sleep(1)
            try:
                response = requests.request("GET", host + "sessions", cookies=self.cookies)
                break
            except Exception:
                print(f"Checking again for {host}")
        status, meh = json.loads(response.content)
        self.assertTrue(status)
        self.assertIn("loaded", meh[key])
        self.assertFalse(meh[key]["loaded"])
        responses = []
        for m in meh.keys():
            responses.append(requests.request("POST", host + "purge_session",
                                              json=json.dumps({"session_key": m}),
                                              cookies=self.cookies))
        print("RESPONSES", [r.content for r in responses])
        # Give it some time to purge
        time.sleep(1)
        for r in responses:
            status, resp = json.loads(r.content)
            task_id = resp["task_id"]
            with self.subTest(i=str(task_id)):
                content = requests.request("GET", host + f"check_task?task_id={task_id}",
                                           cookies=self.cookies).content
                status, resp = json.loads(content)
                self.assertTrue(resp["result"])
        # import ipdb; ipdb.set_trace()
        self.shutdown_daemon(host)
        # And some time to shutdown
        time.sleep(1)

    @classmethod
    def shutdown_daemon(cls, host):
        if cls.daemon._fwd_ports_thread is not None:
            cls.daemon._fwd_ports_thread.kill()
        response = requests.request("GET", host + "_shutdown", timeout=2,
                                    cookies=cls.cookies)
        return response


if __name__ == '__main__':
    unittest.main()
