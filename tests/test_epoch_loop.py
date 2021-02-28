import pytest
from functools import partial
from threading import Thread
import sys
import time
from dorc.trainer.epoch import EpochLoop
from dorc.device import all_devices
from util import SubTrainer


def train_one_batch(train_step, batch):
    get_raw = False
    if get_raw:
        raw, batch = batch[0], batch[1]
    received = train_step(batch)
    if get_raw:
        received["raw"] = raw
    return received


# NOTE: Consumer
@pytest.mark.threaded
def test_epoch_loop_run_task(params_and_trainer):
    params, trainer = params_and_trainer
    trainer = SubTrainer(False, **params)
    trainer.config.model_params['net'].loaded = True
    trainer._init_all()
    signals = trainer._epoch_runner.signals
    signals.paused.clear()
    hooks_with_args = [*trainer._epoch_runner._post_batch_hooks_with_args["train"].values()]
    step = trainer.update_functions.train
    step.models["net"].load_into_memory()
    train_loop = EpochLoop(partial(train_one_batch, step), signals,
                           trainer.train_loader, [], hooks_with_args,
                           trainer._epoch_runner.device_mon)
    t = Thread(target=train_loop.run_task)
    t.start()
    signals.paused.set()
    time.sleep(1)
    signals.paused.clear()
    assert t.is_alive()
    time.sleep(.5)
    assert len(trainer.epoch_runner.batch_vars)
    assert train_loop.paused
    train_loop.finish()
    signals.paused.set()
    time.sleep(.5)
    assert train_loop.finished
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


# NOTE: Consumer
@pytest.mark.threaded
def test_epoch_loop_run_task_have_gpus(params_and_trainer):
    params, trainer = params_and_trainer
    trainer = SubTrainer(False, **params)
    trainer.config.trainer_params.gpus = [0]
    trainer.config.trainer_params.cuda = True
    trainer.config.model_params['net'].loaded = True
    trainer._init_all()
    signals = trainer._epoch_runner.signals
    signals.paused.clear()
    hooks_with_args = [*trainer._epoch_runner._post_batch_hooks_with_args["train"].values()]
    step = trainer.update_functions.train
    step.models["net"].load_into_memory()
    train_loop = EpochLoop(partial(train_one_batch, step), signals,
                           trainer.train_loader, [], hooks_with_args,
                           trainer._epoch_runner.device_mon)
    t = Thread(target=train_loop.run_task)
    t.start()
    signals.paused.set()
    time.sleep(1)
    signals.paused.clear()
    assert t.is_alive()
    time.sleep(.5)
    assert len(trainer.epoch_runner.batch_vars)
    if all_devices():
        assert any("gpu" in x[2] for x in trainer.epoch_runner.batch_vars)
    assert train_loop.paused
    train_loop.finish()
    signals.paused.set()
    time.sleep(.5)
    assert train_loop.finished
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)


# NOTE: Producer
@pytest.mark.threaded
def test_epoch_loop_fetch_data(params_and_trainer):
    _, trainer = params_and_trainer
    trainer.config.model_params['net'].loaded = True
    trainer._init_all()
    signals = trainer._epoch_runner.signals
    signals.paused.clear()
    hooks_with_args = [*trainer._epoch_runner._post_batch_hooks_with_args["train"].values()]
    step = trainer.update_functions.train
    step.models["net"].load_into_memory()
    train_loop = EpochLoop(partial(train_one_batch, step), signals,
                           trainer.train_loader, [], hooks_with_args,
                           trainer._epoch_runner.device_mon)
    signals.paused.clear()
    assert not signals.paused.is_set()
    assert train_loop._data_q.empty()
    signals.paused.set()
    time.sleep(.5)
    assert not train_loop._data_q.empty()
    time.sleep(2)
    assert train_loop._data_q.full()
    train_loop._init = False
    train_loop.finish()
    assert train_loop.finished
    for handler in trainer._logger.handlers:
        handler.close()
        trainer._logger.removeHandler(handler)
