import pytest
import os
import requests
import time
import json
from dorc import util
from dorc.daemon.models import SessionMethodResponseModel as smod


@pytest.mark.http
def test_load_unfinished_sessions(daemon_and_cookies, json_config):
    daemon, cookies = daemon_and_cookies
    data = {}
    data["name"] = "meh_session"
    data["config"] = json_config.copy()
    data["load"] = False
    hostname = daemon.hostname
    port = daemon.port
    host = f"http://{hostname}:{port}/"
    responses = []
    response = requests.request("POST", host + "create_session",
                                json=data,
                                cookies=cookies)
    responses.append(response)
    time.sleep(1)
    response = requests.request("POST", host + "create_session",
                                json=data,
                                cookies=cookies)
    responses.append(response)
    time.sleep(1)
    response = requests.get(host + "sessions", cookies=cookies)
    sessions = json.loads(response.content)
    assert sessions
    # shutdown the daemon
    response = util.stop_test_daemon(port)
    assert "shutting down" in str(response.content.lower())
    time.sleep(.5)
    key = [*sessions.keys()][0]
    data_dir = os.path.join(daemon.root_dir, key)
    with open(os.path.join(data_dir, "session_state"), "r") as f:
        state = json.load(f)
    # switch state to finished
    state["epoch"] = state["max_epochs"]
    with open(os.path.join(data_dir, "session_state"), "w") as f:
        json.dump(state, f)
    # start new daemon
    daemon = util.make_test_daemon(hostname, port + 5, ".test_dir", no_clear=True)
    host = "http://" + ":".join([hostname, str(port + 5) + "/"])
    for _ in range(3):
        time.sleep(1)
        try:
            response = requests.get(host + "sessions", cookies=cookies)
            break
        except Exception:
            print(f"Checking again for {host}")
    sessions = json.loads(response.content)
    assert sessions
    assert "loaded" in sessions[key]
    assert not sessions[key]["loaded"]
    responses = []
    for m in sessions.keys():
        responses.append(requests.get(host + f"purge_session?session_key={m}",
                                      cookies=cookies))
    # Give it some time to purge
    time.sleep(1)
    for r in responses:
        task_id = smod.parse_obj(json.loads(r.content)).task_id
        content = requests.request("GET", host + f"check_task?task_id={task_id}",
                                   cookies=cookies).content
        assert smod.parse_obj(json.loads(content)).status
    util.stop_test_daemon(port+5)
    time.sleep(1)
