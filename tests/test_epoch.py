import pytest
import sys
import time
from threading import Thread
sys.path.append("../")
from dorc.trainer.epoch import EpochLoop
from dorc.trainer import Trainer


@pytest.mark.threaded
def test_epoch_device_poll(trainer_json_config):
    _, trainer = trainer_json_config
    trainer._init_all()
    with trainer._epoch_runner.device_poll.monitor():
        for x in range(1000):
            x = x ** 2
        time.sleep(2)
    assert "cpu_info" in trainer._epoch_runner.device_poll._data
    assert "memory_info" in trainer._epoch_runner.device_poll._data
    assert isinstance(trainer._epoch_runner.device_poll.cpu_util[0], float)
    assert isinstance(trainer._epoch_runner.device_poll.mem_util[0], float)
    if trainer._epoch_runner.device_poll._handles:
        assert isinstance(trainer._epoch_runner.device_poll.gpu_util[0], float)


@pytest.mark.threaded
def test_epoch_train(trainer_json_config):
    _, trainer = trainer_json_config
    def _debug():
        print(epoch_runner.train_loop.running)
        print(epoch_runner.train_loop.waiting)
        print(epoch_runner.train_loop.finished)

    def _batch_vars():
        print([x for x in epoch_runner.batch_vars])

    trainer._init_all()
    trainer.set_model({"net": "net"})
    epoch_runner = trainer._epoch_runner
    t = Thread(target=epoch_runner.run_train,
               args=[trainer._training_steps["train"][0], trainer.train_loader, "epoch"])
    t.start()
    trainer._running_event.set()
    time.sleep(2)
    assert epoch_runner.running
    assert not epoch_runner.waiting
    trainer._running_event.clear()
    time.sleep(.5)
    assert(epoch_runner.running)
    assert(epoch_runner.waiting)
    assert(epoch_runner.batch_vars.get("train", 1, "loss"))
    assert(epoch_runner.batch_vars.get("train", 1, "cpu_util"))
    assert(epoch_runner.batch_vars.get("train", 1, "mem_util"))
    assert(epoch_runner.batch_vars.get("train", 1, "time"))
    assert(epoch_runner.batch_vars.get("train", 1, "batch_time"))
    if epoch_runner.device_mon.gpu_util is not None:
        assert(epoch_runner.batch_vars.get("train", 1, "gpu_util"))
    assert len(epoch_runner.batch_vars[0]) == 4
    trainer._current_aborted_event.set()
    trainer._running_event.set()
    time.sleep(.5)
    assert not (epoch_runner.running)
    assert not (epoch_runner.waiting)
    trainer._current_aborted_event.clear()
    trainer._running_event.clear()


@pytest.mark.threaded
def test_epoch_test(trainer_json_config):
    _, trainer = trainer_json_config
    def _debug():
        print(epoch_runner.test_loop.running)
        print(epoch_runner.test_loop.waiting)
        print(epoch_runner.test_loop.finished)

    def _batch_vars():
        print([x for x in epoch_runner.batch_vars])

    trainer._init_all()
    trainer.set_model({"net": "net"})
    epoch_runner = trainer._epoch_runner
    t = Thread(target=epoch_runner.run_test,
               args=[trainer._training_steps["test"][0], trainer.test_loader])
    t.start()
    trainer._running_event.set()
    time.sleep(2)
    assert(epoch_runner.running)
    assert not (epoch_runner.waiting)
    trainer._running_event.clear()
    trainer.pause()
    time.sleep(.5)
    assert(epoch_runner.running)
    assert(epoch_runner.waiting)
    assert(epoch_runner.batch_vars.get("test", 1, "loss"))
    assert(epoch_runner.batch_vars.get("test", 1, "cpu_util"))
    assert(epoch_runner.batch_vars.get("test", 1, "mem_util"))
    assert(epoch_runner.batch_vars.get("test", 1, "time"))
    assert(epoch_runner.batch_vars.get("test", 1, "batch_time"))
    if epoch_runner.device_mon.gpu_util is not None:
        assert(epoch_runner.batch_vars.get("test", 1, "gpu_util"))
    assert len(epoch_runner.batch_vars[0]) == 4
    trainer._current_aborted_event.set()
    trainer._running_event.set()
    time.sleep(.5)
    assert not (epoch_runner.running)
    assert not (epoch_runner.waiting)
    trainer._current_aborted_event.clear()
    trainer._running_event.clear()
