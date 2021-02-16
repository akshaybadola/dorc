import pytest
import requests
import json
import time
from dorc.util import dget


# FIXME: In trying to instantiate an interface without a trainer, I get an
#        error. It's too coupled to the trainer. Should fix?
#        - But isn't the interface just a wrapper around the trainer?
#        - Actually it does a few more things like checking modules, loading/unloading
#          state etc. and stuff
@pytest.mark.http
def test_iface_init(params_and_iface):
    params, iface = params_and_iface
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


@pytest.mark.todo
def test_iface_write_config():
    pass


@pytest.mark.todo
def test_iface_load_config():
    # load with overrides
    pass


@pytest.mark.todo
def test_iface_get_config():
    pass


@pytest.mark.quick
def test_iface_update_config(params_and_iface):
    params, iface = params_and_iface
    with open(iface.config_file) as f:
        bleh = json.load(f)
    overrides = [["trainer_params", "max_epochs", 120],
                 ["optimizers", "Adam", "function", "params", "lr", 0.1]]
    assert dget(bleh, *(overrides[0][:-1])) == 100
    assert dget(bleh, *(overrides[1][:-1])) == 0.01
    iface.update_config(bleh, overrides)
    assert dget(bleh, *(overrides[0][:-1])) == 120
    assert dget(bleh, *(overrides[1][:-1])) == 0.1


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
