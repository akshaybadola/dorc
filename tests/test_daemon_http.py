import pytest
import time
import json
import requests
from dorc.util import make_test_daemon
from util import _create_session, assertIn
from dorc.daemon.models import SessionMethodResponseModel as smod


@pytest.mark.http
def test_daemon_http_create_sessions_http(daemon_and_cookies, json_config):
    daemon, cookies = daemon_and_cookies
    data = {}
    data["name"] = "meh_session"
    data["config"] = json_config.copy()
    data["load"] = True
    host = f"http://{daemon.hostname}:{daemon.port}/"
    responses = []
    response = requests.post(host + "create_session",
                             json=data,
                             cookies=cookies)
    responses.append(response)
    time.sleep(1)
    response = requests.post(f"http://{daemon.hostname}:{daemon.port}/create_session",
                             json=data,
                             cookies=cookies)
    responses.append(response)
    time.sleep(2)
    response = requests.get(f"http://{daemon.hostname}:{daemon.port}/sessions",
                            cookies=cookies)
    assert len(json.loads(response.content)) == 2


@pytest.mark.http
def test_daemon_http_get_sessions(daemon_and_cookies, json_config):
    daemon, cookies = daemon_and_cookies
    host = f"http://{daemon.hostname}:{daemon.port}/"
    result = _create_session(daemon, json_config)
    assert result[1]
    time.sleep(1)
    result = _create_session(daemon, json_config)
    assert result[1]
    time.sleep(1)
    response = requests.get(host + "sessions",
                            cookies=cookies)
    json.loads(response.content)


@pytest.mark.http
def test_daemon_http_load_unload_session(daemon_and_cookies, json_config):
    daemon, cookies = daemon_and_cookies
    host = f"http://{daemon.hostname}:{daemon.port}/"
    _create_session(daemon, json_config, True)
    time.sleep(1)
    response = requests.request("GET", host + "sessions", cookies=cookies)
    response = json.loads(response.content)
    assert isinstance(response, dict)
    key = [*response.keys()][-1]
    assert response[key]["loaded"]
    response = requests.get(host + f"unload_session?session_key={key}",
                            cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    response = requests.request("GET", host + f"check_task?task_id={result.task_id}",
                                cookies=cookies)
    response = json.loads(response.content)
    assertIn("task_id", response)
    response = requests.get(host + "sessions", cookies=cookies)
    assert not json.loads(response.content)[key]["loaded"]


@pytest.mark.http
def test_daemon_http_unload_then_load_session(daemon_and_cookies, json_config):
    "First unload then load"
    daemon, cookies = daemon_and_cookies
    host = f"http://{daemon.hostname}:{daemon.port}/"
    _create_session(daemon, json_config, True)
    time.sleep(1)
    response = requests.request("GET", host + "sessions", cookies=cookies)
    response = json.loads(response.content)
    assert isinstance(response, dict)
    key = [*response.keys()][-1]
    assert response[key]["loaded"]
    response = requests.get(host + f"unload_session?session_key={key}",
                            cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    response = requests.request("GET", host + f"check_task?task_id={result.task_id}",
                                cookies=cookies)
    response = json.loads(response.content)
    assertIn("task_id", response)
    response = requests.get(host + "sessions", cookies=cookies)
    assert not json.loads(response.content)[key]["loaded"]
    response = requests.request("POST", host + "load_session",
                                json=json.dumps({"session_key": key}),
                                cookies=cookies)
    result = smod.parse_obj(json.loads(response.content))
    time.sleep(1)
    response = requests.request("GET", host + f"check_task?task_id={result.task_id}",
                                cookies=cookies)
    while "not yet" in json.loads(response.content)["message"].lower():
        time.sleep(1)
        response = requests.request("GET", host + f"check_task?task_id={result.task_id}",
                                    cookies=cookies)
    response = requests.get(host + "sessions", cookies=cookies)
    sessions = json.loads(response.content)
    assert sessions[key]["loaded"]
    assert "port" in sessions[key]

    # NOTE: DEBUG
    # def test():
    #     import sys
    #     sys.path.append("..")
    #     from dorc.trainer import Trainer
    #     if os.path.exists(".key"):
    #         import shutil
    #         shutil.rmtree(".key")
    #     os.mkdir(".key")
    #     from _setup import config
    #     config["dataloader_params"]["train"]["pin_memory"] = False
    #     config["dataloader_params"]["test"]["pin_memory"] = False
    #     trainer = Trainer(**{"data_dir": ".key", **config})
    #     trainer._init_all()
    #     return trainer
    # trainer = test()


@pytest.mark.http
@pytest.mark.todo
def test_daemon_http_purge():
    pass


@pytest.mark.http
@pytest.mark.todo
def test_daemon_http_upload_session():
    pass


@pytest.mark.http
@pytest.mark.todo
def test_daemon_http_archive():
    pass


@pytest.mark.http
@pytest.mark.todo
def test_daemon_http_clone():
    pass


@pytest.mark.http
@pytest.mark.todo
def test_daemon_http_reinit():
    pass
