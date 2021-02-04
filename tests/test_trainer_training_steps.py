import pytest
import torch
import sys
# from _setup_local import config
from util import get_model, get_batch
sys.path.append("../")
from dorc.device import all_devices


@pytest.mark.quick
def test_train_step_no_gpu(setup_and_net, get_step):
    config, Net = setup_and_net
    train_step = config["update_functions"]["train"]
    model = get_model("net", config, [])
    train_step.set_models({"net": model})
    cname = [*config["criteria"].keys()][0]
    train_step.set_criteria({"net": config["criteria"][cname]["function"](
        **config["criteria"][cname]["params"])})
    retval = train_step(get_batch())
    assert all(x in retval for x in train_step.returns)


@pytest.mark.quick
@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
def test_train_step_single_gpu(setup_and_net, get_step):
    config, _ = setup_and_net
    train_step = config["update_functions"]["train"]
    model = get_model("net", config, [0])
    train_step.set_models({"net": model})
    cname = [*config["criteria"].keys()][0]
    train_step.set_criteria({"net": config["criteria"][cname]["function"](
        **config["criteria"][cname]["params"])})
    retval = train_step(get_batch())
    assert all(x in retval for x in train_step.returns)


@pytest.mark.quick
@pytest.mark.skipif(len(all_devices()) < 2,
                    reason=f"Cannot run without at least 2 GPUs.")
def test_val_step_single_gpu(setup_and_net, get_step):
    config, _ = setup_and_net
    train_step = config["update_functions"]["train"]
    train_step.test = True
    model = get_model("net", config, [0])
    train_step.set_models({"net": model})
    cname = [*config["criteria"].keys()][0]
    train_step.set_criteria({"net": config["criteria"][cname]["function"](
        **config["criteria"][cname]["params"])})
    retval = train_step(get_batch())
    assert all(x in retval for x in train_step.returns)


# NOTE: how do we test that it is actually parallelized?
@pytest.mark.quick
@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
def test_dataparallel(setup_and_net, get_step):
    config, _ = setup_and_net
    train_step = config["update_functions"]["train"]
    train_step.test = True
    model = get_model("net", config, [0, 1])
    train_step.set_models({"net": model})
    cname = [*config["criteria"].keys()][0]
    train_step.set_criteria({"net": config["criteria"][cname]["function"](
        **config["criteria"][cname]["params"])})
    retval = train_step(get_batch())
    assert all(x in retval for x in train_step.returns)
    assert isinstance(model._model, torch.nn.DataParallel)
    assert retval["outputs"].device == torch.device(0)


def test_distributed_data_parallel():
    pass


@pytest.mark.skipif(not all_devices(), reason=f"Cannot run without GPUs.")
def test_model_parallel():
    pass
