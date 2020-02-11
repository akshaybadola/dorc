import unittest
import sys
import time
from setup import config
from threading import Thread
sys.path.append("../")
from trainer.trainer import Trainer


class EpochTest(unittest.TestCase):
    def setUp(self):
        """Setup a simple trainer with MNIST dataset."""
        self.config = config
        self.trainer = Trainer(**self.config)
        self.trainer._init_all()

    def test_device_poll(self):
        with self.trainer._epoch_runner.device_poll.monitor():
            for x in range(1000):
                x = x ** 2
            time.sleep(2)
        self.assertTrue("cpu_info" in self.trainer._epoch_runner.device_poll._data)
        self.assertTrue("memory_info" in self.trainer._epoch_runner.device_poll._data)
        self.assertIsInstance(self.trainer._epoch_runner.device_poll.cpu_util[0], float)
        self.assertIsInstance(self.trainer._epoch_runner.device_poll.mem_util[0], float)
        if self.trainer._epoch_runner.device_poll._handles:
            self.assertIsInstance(self.trainer._epoch_runner.device_poll.gpu_util[0], float)

    def test_epoch(self):
        epoch_runner = self.trainer._epoch_runner
        t = Thread(target=epoch_runner.run_train,
                   args=[self.trainer.train_step_func, self.trainer.train_loader, "epoch"])
        t.start()
        self.trainer._running_event.clear()
        self.assertTrue(epoch_runner.running)
        self.assertFalse(epoch_runner.waiting)
        self.trainer._running_event.set()
        time.sleep(.5)
        self.assertFalse(epoch_runner.waiting)
        self.trainer._current_aborted_event.set()
        time.sleep(.5)


if __name__ == '__main__':
    unittest.main()
