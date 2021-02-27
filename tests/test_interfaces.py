import pytest
import requests
import shutil
import copy
import os
import json
import time
import base64
import util
from dorc.util import dget
from dorc.interfaces import FlaskInterface


@pytest.mark.http
def test_iface_init(params_and_iface):
    params, iface = params_and_iface
    iface.trainer.set_model({"net": "net"})
    host = params['host']
    response = requests.request("GET", host + "start")
    assert "start" in response.content.decode().lower()
    time.sleep(.5)
    response = requests.request("GET", host + "pause")
    assert "pausing" in response.content.decode().lower()
    time.sleep(.5)
    assert iface.trainer.paused
    response = requests.request("GET", host + "resume")
    assert "resuming" in response.content.decode().lower()
    time.sleep(.5)
    assert not iface.trainer.paused
    iface.trainer.pause()
    time.sleep(1)
    # assert iface.trainer.paused
    iface.trainer.abort_loop()


@pytest.mark.quick
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_iface_read_config(indirect_config):
    gm_dir = os.path.abspath("_some_modules_dir/global_modules")
    tc_dir = "test_config_dir"
    if isinstance(indirect_config, dict):
        with open("test_config_dir/config.json", "w") as f:
            json.dump(indirect_config, f)
        FlaskInterface.read_config(tc_dir, "json")
    else:
        status, msg = util.write_py_config(
            base64.b64decode(indirect_config), tc_dir, gm_dir)
        FlaskInterface.read_config(tc_dir, "python")


@pytest.mark.quick
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_iface_update_config(params_and_iface, indirect_config):
    params, iface = params_and_iface
    bleh = copy.deepcopy(iface._read_config())
    bleh["trainer_params"]["max_epochs"] = 100
    bleh["optimizers"]["Adam"]["params"]["lr"] = 0.01
    overrides = [["trainer_params", "max_epochs", 120],
                 ["optimizers", "Adam", "params", "lr", 0.1]]
    assert dget(bleh, *(overrides[0][:-1])) == 100
    assert dget(bleh, *(overrides[1][:-1])) == 0.01
    iface.update_config(bleh, overrides)
    assert dget(bleh, *(overrides[0][:-1])) == 120
    assert dget(bleh, *(overrides[1][:-1])) == 0.1


@pytest.mark.quick
@pytest.mark.todo
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_iface_write_config(indirect_config):
    pass


@pytest.mark.quick
@pytest.mark.todo
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_iface_get_config(indirect_config):
    pass


@pytest.mark.quick
@pytest.mark.todo
@pytest.mark.parametrize("indirect_config", ["json", "pystr"], indirect=True)
def test_iface_create_trainer(indirect_config):
    pass



@pytest.mark.todo
@pytest.mark.quick
def test_iface_update_config_handle_new_varnames(params_and_iface):
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_iface_update_config_handle_missing(params_and_iface):
    pass


@pytest.mark.todo
def test_trainer_crash():
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_iface_trainer_get():
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_iface_trainer_post():
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_iface_trainer_post_form():
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_iface_trainer_route():
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_iface_auth():
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_iface_trainer_helpers():
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_iface_trainer_props():
    pass


@pytest.mark.quick
@pytest.mark.todo
def test_iface_trainer_methods():
    pass
