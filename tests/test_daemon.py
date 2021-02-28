import os
import copy
import shutil
import pytest
import time
import json
import datetime
import requests
from dorc.daemon import Daemon, models
from dorc.interfaces import FlaskInterface
from dorc.util import dget
from util import _create_session


@pytest.mark.quick
def test_daemon_find_port(daemon_and_cookies):
    daemon, cookies = daemon_and_cookies
    port = daemon._find_open_port()
    assert port is not None


@pytest.mark.todo
def test_daemon_update_init_file(daemon_and_cookies):
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_daemon_modules(daemon_and_cookies):
    # module available to sessions
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_daemon_datasets(daemon_and_cookies):
    # load unload
    # dataset valid
    pass


@pytest.mark.quick
@pytest.mark.parametrize("indirect_config", ["json", "pybytes"], indirect=True)
def test_daemon_create_trainer(daemon_and_cookies, indirect_config):
    daemon, cookies = daemon_and_cookies
    name = "test_trainer"
    time_str = datetime.datetime.now().isoformat()
    data_dir = os.path.join(daemon._root_dir, name, time_str)
    os.makedirs(data_dir)
    daemon._create_trainer(0, name, time_str, data_dir, indirect_config)
    time.sleep(1)
    result = daemon._check_result(0)
    assert result[1]


@pytest.mark.quick
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_daemon_create_session(daemon_and_cookies, indirect_config):
    daemon, cookies = daemon_and_cookies
    if isinstance(indirect_config, dict):
        result = _create_session(daemon, indirect_config)
    else:
        result = _create_session(daemon, indirect_config)
    assert result[1]


@pytest.mark.quick
@pytest.mark.todo
def test_daemon_session_method_check(daemon_and_cookies):
    # check key in data, not key in data
    # check session valid works
    pass


@pytest.mark.quick
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_daemon_create_and_then_load_session(daemon_and_cookies, indirect_config):
    daemon, cookies = daemon_and_cookies
    result = _create_session(daemon, indirect_config)
    assert result[1]
    time_str = [*daemon._sessions["test_session"]["sessions"].keys()][-1]
    if isinstance(indirect_config, dict):
        assert os.path.exists(os.path.join(".test_dir", "test_session", time_str, "config.json"))
    else:
        assert (os.path.exists(os.path.join(".test_dir",
                                            "test_session",
                                            time_str,
                                            "session_config.py")) or
                os.path.exists(os.path.join(".test_dir",
                                            "test_session",
                                            time_str,
                                            "session_config")))
    daemon._load_session_helper(0, "test_session", time_str)
    result = daemon._check_result(0)
    assert os.path.exists(os.path.join(".test_dir", "test_session", time_str, "session_state"))
    assert result[1]


@pytest.mark.quick
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_daemon_load_session_without_state(daemon_and_cookies, indirect_config):
    daemon, cookies = daemon_and_cookies
    result = _create_session(daemon, indirect_config)
    assert result[1]
    time_str = [*daemon._sessions["test_session"]["sessions"].keys()][-1]
    state_path = os.path.join(".test_dir", "test_session", time_str, "session_state")
    assert os.path.exists(state_path)
    os.remove(state_path)
    daemon._testing = True
    daemon.scan_sessions()
    assert os.path.exists(state_path)


@pytest.mark.quick
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_daemon_create_and_load_session_with_overrides(daemon_and_cookies, indirect_config):
    daemon, cookies = daemon_and_cookies
    result = _create_session(daemon, indirect_config)
    assert result[1]
    time_str = [*daemon._sessions["test_session"]["sessions"].keys()][-1]
    data_dir = os.path.join(".test_dir", "test_session", time_str)
    if isinstance(indirect_config, dict):
        assert os.path.exists(os.path.join(data_dir, "config.json"))
        status, config = FlaskInterface.read_config(data_dir, "json")
        overrides_0 = [["trainer_params", "max_epochs", 120],
                       ["optimizers", "Adam", "function", "params", "lr", 0.1]]
        overrides_1 = [["trainer_params", "seed", 2222],
                       ["optimizers", "Adam", "function", "params", "lr", 0.2]]
    else:
        assert (os.path.exists(os.path.join(data_dir, "session_config.py")) or
                os.path.exists(os.path.join(data_dir, "session_config")))
        status, config = FlaskInterface.read_config(data_dir, "python")
        assert status
        overrides_0 = [["trainer_params", "max_epochs", 120],
                       ["optimizers", "Adam", "params", "lr", 0.1]]
        overrides_1 = [["trainer_params", "seed", 2222],
                       ["optimizers", "Adam", "params", "lr", 0.2]]
    FlaskInterface.update_config(config, overrides_0)
    with open(os.path.join(data_dir, "overrides.json.00"), "w") as f:
        json.dump(config["trainer_params"], f)
    FlaskInterface.update_config(config, overrides_1)
    with open(os.path.join(data_dir, "overrides.json.01"), "w") as f:
        json.dump(config["trainer_params"], f)
    daemon._load_session_helper(0, "test_session", time_str)
    result = daemon._check_result(0)
    assert os.path.exists(os.path.join(".test_dir", "test_session", time_str, "session_state"))
    assert result[1]
    assert dget(config, "trainer_params", "max_epochs") == 120
    assert dget(config, "trainer_params", "seed") == 2222
    if isinstance(indirect_config, dict):
        assert dget(config, "optimizers", "Adam", "function", "params", "lr") == 0.2
    else:
        assert dget(config, "optimizers", "Adam", "params", "lr") == 0.2


@pytest.mark.quick
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_daemon_list_sessions(daemon_and_cookies, indirect_config):
    daemon, _ = daemon_and_cookies
    if not daemon._sessions:
        result = _create_session(daemon, indirect_config)
        assert result[1]
    assert any("test_session" in x for x in daemon.sessions_list.keys())


@pytest.mark.http
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_daemon_load_and_unload_session(daemon_and_cookies, indirect_config):
    daemon, cookies = daemon_and_cookies
    if not daemon.sessions_list:
        result = _create_session(daemon, indirect_config)
        assert result[1]
    key = [*daemon.sessions_list.keys()][0]
    time.sleep(.5)
    assert not daemon.sessions_list[key].loaded
    daemon._load_session_helper(0, *key.split("/"))
    time.sleep(1)
    assert "port" in daemon.sessions_list[key].dict()
    port = daemon.sessions_list[key].port
    assert port is not None
    response = requests.get(f"http://127.0.0.1:{port}/_ping")
    assert "pong" in response.content.decode()
    daemon._unload_session_helper(0, *key.split("/"))
    time.sleep(.5)
    with pytest.raises(requests.ConnectionError):
        response = requests.get(f"http://127.0.0.1:{port}/_ping")


@pytest.mark.quick
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_daemon_purge_sessions(daemon_and_cookies, indirect_config):
    daemon, cookies = daemon_and_cookies
    _create_session(daemon, indirect_config)
    key = [*daemon.sessions_list.keys()][-1]
    assert os.path.exists(os.path.join(daemon.root_dir, key))
    daemon._purge_session_helper(0, *key.split("/"))
    assert key not in daemon.sessions_list
    assert not os.path.exists(os.path.join(daemon.root_dir, key))


@pytest.mark.quick
@pytest.mark.todo
def test_daemon_load_unfinished_sessions():
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_daemon_unload_finished_sessions():
    pass


@pytest.mark.todo
def test_daemon_scan_sessions(daemon_and_cookies, indirect_config):
    pass
    # daemon, cookies = daemon_and_cookies
    # config = copy.deepcopy(indirect_config)
    # _create_session(daemon)
    # _create_session(daemon)  # create three more sessions
    # terminate_live_sessions()
    # daemon._sessions = {}
    # daemon.scan_sessions()
    # for k, v in daemon._sessions["test_session"]["sessions"].items():
    #     assert isinstance(v, dict)
    #     assert not "process" in v
