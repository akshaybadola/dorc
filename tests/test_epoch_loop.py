import os
import shutil
from threading import Thread
from datetime import datetime
import unittest
import sys
import time
from _setup_local import config
sys.path.append("../")
from dorc.trainer.epoch import EpochLoop
from dorc.trainer import Trainer


class EpochLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup a simple trainer with MNIST dataset."""
        cls.config = config
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")
        os.mkdir(".test_dir")
        os.mkdir(".test_dir/test_session")
        time_str = datetime.now().isoformat()
        os.mkdir(f".test_dir/test_session/{time_str}")
        cls.data_dir = f".test_dir/test_session/{time_str}"
        cls.trainer = Trainer(**{"data_dir": cls.data_dir, **cls.config})
        cls.trainer._init_all()
        cls.train_loader = cls.trainer.train_loader
        cls.dev_mon = cls.trainer._epoch_runner.device_mon
        cls.hooks = [*cls.trainer._epoch_runner._post_batch_hooks["train"].values()]
        cls.signals = cls.trainer._epoch_runner.signals
        cls.train_step = cls.trainer.train_step_func

    def train_one_batch(self, batch):
        get_raw = False
        if get_raw:
            raw, batch = batch[0], batch[1]
        received = self.train_step(batch)
        if get_raw:
            received["raw"] = raw
        return received

    # NOTE: Consumer
    def test_epoch_loop_run_task(self):
        self.signals.paused.clear()
        train_loop = EpochLoop(self.train_one_batch, self.signals,
                               self.train_loader, self.hooks, self.dev_mon)
        t = Thread(target=train_loop.run_task)
        t.start()
        self.signals.paused.set()
        time.sleep(1)
        self.signals.paused.clear()
        self.assertTrue(t.is_alive())
        time.sleep(.5)
        self.assertTrue(train_loop.paused)
        train_loop.finish()
        self.signals.paused.set()
        time.sleep(.5)
        self.assertTrue(train_loop.finished)

    # NOTE: Producer
    def test_epoch_loop_fetch_data(self):
        self.signals.paused.clear()
        train_loop = EpochLoop(self.train_one_batch, self.signals,
                               self.train_loader, self.hooks, self.dev_mon)
        self.assertFalse(self.signals.paused.is_set())
        self.assertTrue(train_loop._data_q.empty())
        self.signals.paused.set()
        time.sleep(.5)
        self.assertFalse(train_loop._data_q.empty())
        time.sleep(2)
        self.assertTrue(train_loop._data_q.full())
        train_loop._init = False
        train_loop.finish()
        self.assertTrue(train_loop.finished)

    def tearDown(self):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
