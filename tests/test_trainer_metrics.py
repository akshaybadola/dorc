import pytest
import time
import sys
sys.path.append("../")
from dorc.trainer import Trainer
from util import SubTrainer


@pytest.mark.timeout(5, method="signal")
@pytest.mark.threaded
def test_trainer_capture_metrics_epoch(trainer_json_config):
    _, trainer = trainer_json_config
    trainer.reserved_gpus = []
    trainer.reserve_gpus = lambda x: [True, None]
    trainer.trainer_params.cuda = True
    print(id(trainer))
    trainer._init_all()
    trainer.set_model({"net": "net"})
    trainer.start()
    time.sleep(2)
    trainer.pause()
    time.sleep(1)
    retval = trainer.call_func_with_args("gather_metrics", trainer.epoch_runner)
    assert retval.status
    metrics = retval.data
    assert "train" in metrics
    assert "test" in metrics
    assert "val" not in metrics
    assert "loss" in metrics["train"]
    assert "num_datapoints" in metrics["train"]
    assert(metrics["train"]["num_datapoints"] > 0)
    assert(metrics["train"]["loss"] > 0)
    trainer.abort_loop()


@pytest.mark.timeout(5, method="signal")
@pytest.mark.threaded
def test_trainer_capture_metrics_iterations(trainer_json_config):
    _, trainer = trainer_json_config
    print(id(trainer))
    bleh = trainer.trainer_params.dict()
    new_config = {}
    new_config["training_type"] = "iterations"
    new_config["max_epochs"] = 0
    new_config["max_iterations"] = 800
    new_config["test_frequency"] = 200
    bleh.update(new_config)
    trainer.config.trainer_params = trainer.trainer_params.__class__(**bleh)
    trainer._init_all()
    trainer.set_model({"net": "net"})
    trainer.start()
    time.sleep(2)
    trainer.pause()
    result = trainer.call_func_with_args("gather_metrics", trainer.epoch_runner)
    assert(result.data["train"])
    assert "loss" in result.data["train"]
    assert "num_datapoints" in result.data["train"]
    assert result.data["train"]["loss"] != 0
    trainer.abort_loop()
