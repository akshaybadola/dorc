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

    def test_epoch_train(self):
        def _debug():
            print(epoch_runner.train_loop.running)
            print(epoch_runner.train_loop.waiting)
            print(epoch_runner.train_loop.finished)

        def _batch_vars():
            print([x for x in epoch_runner.batch_vars])

        epoch_runner = self.trainer._epoch_runner
        t = Thread(target=epoch_runner.run_train,
                   args=[self.trainer.train_step_func, self.trainer.train_loader, "epoch"])
        t.start()
        self.trainer._running_event.set()
        time.sleep(2)
        self.assertTrue(epoch_runner.running)
        self.assertFalse(epoch_runner.waiting)
        self.trainer._running_event.clear()
        time.sleep(.5)
        self.assertTrue(epoch_runner.running)
        self.assertTrue(epoch_runner.waiting)
        self.assertTrue(epoch_runner.batch_vars.get(1, "loss"))
        self.assertTrue(epoch_runner.batch_vars.get(1, "cpu_util"))
        self.assertTrue(epoch_runner.batch_vars.get(1, "mem_util"))
        self.assertTrue(epoch_runner.batch_vars.get(1, "time"))
        self.assertTrue(epoch_runner.batch_vars.get(1, "batch_time"))
        if epoch_runner.device_mon.gpu_util is not None:
            self.assertTrue(epoch_runner.batch_vars.get(1, "gpu_util"))
        self.assertEqual(len(epoch_runner.batch_vars[0]), 4)
        self.trainer._current_aborted_event.set()
        self.trainer._running_event.set()
        time.sleep(.5)
        self.assertFalse(epoch_runner.running)
        self.assertFalse(epoch_runner.waiting)


if __name__ == '__main__':
    unittest.main()
