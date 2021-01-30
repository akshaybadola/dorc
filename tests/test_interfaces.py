import pytest
import requests
import time


# FIXME: In trying to instantiate an interface without a trainer, I get an
#        error. It's too coupled to the trainer. Should fix?
#        - But isn't the interface just a wrapper around the trainer?
#        - Actually it does a few more things like checking modules, loading/unloading
#          state etc. and stuff
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
    assert iface.trainer.paused
    iface.trainer.abort_loop()


def test_iface_shutdown(params_and_iface):
    params, iface = params_and_iface
    host = params['host']
    response = requests.request("GET", host + "abort_loop")
    print(response.content)
    assert "aborted" in response.content.decode().lower()
    time.sleep(.5)

# def test_trainer_crash():
#     pass

# def test_get():
#     pass

# def test_post():
#     pass

# def test_post_form():
#     pass

# def test_route():
#     pass

# def test_auth():
#     pass

# def test_helpers():
#     pass

# def test_props():
#     pass

# def test_extras():
#     pass
