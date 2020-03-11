import os
import shutil
from datetime import datetime
import unittest
import sys
import time
from _setup_local import config
import torch
sys.path.append("../")
from trainer.trainer import Trainer


class StateTest(unittest.TestCase):
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

    def test_main_loop(self):
        # Should have subtest for a set of transitions I guess. I can generate
        # predicates and combinatorial states according to that
        # This link has a good post about it
        # https://www.caktusgroup.com/blog/2017/05/29/subtests-are-best/
        # none -> running
        self.assertIsInstance(self.trainer.train_loader, torch.utils.data.DataLoader)
        self.trainer._transition(self.trainer.current_state, "main_running_train")
        self.assertFalse(self.trainer.paused)
        time.sleep(2)
        self.assertTrue("main" in self.trainer._threads)
        self.trainer.pause()
        time.sleep(1)
        self.assertTrue(self.trainer.paused)
        self.assertTrue(self.trainer._epoch_runner.running)
        self.assertTrue(self.trainer._epoch_runner.waiting)
        time.sleep(1)
        # running -> aborted/finished without gathering results
        self.trainer._abort_current("test")
        time.sleep(1)
        self.assertFalse(self.trainer._epoch_runner.running)
        self.assertFalse(self.trainer._epoch_runner.waiting)
        self.assertTrue([x for x in self.trainer._epoch_runner.batch_vars])
        self.assertFalse(self.trainer._threads["main"].is_alive())
        time.sleep(1)
        self.assertFalse(self.trainer._metrics["train"]["loss"])
        # aborted/finished -> running
        self.trainer._run_new_if_finished()
        self.assertFalse(self.trainer.paused)
        time.sleep(1)
        # running -> paused
        self.trainer.pause()
        self.assertTrue(self.trainer.paused)
        # running -> finish AND gather results
        self.trainer._abort_current_run_cb()
        self.assertTrue(self.trainer._metrics["train"]["loss"])
        self.assertFalse(self.trainer._threads["main"].is_alive())
        # start again: finished -> running
        self.trainer._run_new_if_finished()
        # maybe after val/test hooks are run?
        time.sleep(2)
        self.assertFalse(self.trainer.paused)
        print("HERE in TEST CASE")
        self.trainer._abort_current("test")

    def test_main_adhoc_back(self):
        print(self.trainer._task_runners)
        self.trainer._start_if_not_running()
        time.sleep(.5)
        self.assertFalse(self.trainer.paused)
        time.sleep(.5)
        self.assertFalse(self.trainer._epoch_runner.train_loop.paused)
        self.assertFalse(self.trainer._epoch_runner.train_loop.finished)
        self.assertFalse(self.trainer._epoch_runner.train_loop.init)
        self.assertEqual(self.trainer.current_state, "main_running_train")
        self.assertTrue(self.trainer._epoch_runner.running)
        self.assertTrue("main" in self.trainer._threads)
        self.trainer._user_funcs["test_func"] = lambda: None
        time.sleep(1)
        self.trainer.adhoc_eval({"test": {"epoch": "current",
                                          "num_or_fraction": 100,
                                          "device": "cpu",
                                          "data": "test",
                                          "callback": "test_func"}})
        time.sleep(1)
        self.assertTrue("adhoc" in self.trainer._task_runners)
        self.assertTrue(self.trainer._epoch_runner.train_loop.paused)
        self.assertFalse(self.trainer._epoch_runner.train_loop.finished)
        self.assertFalse(self.trainer._epoch_runner.train_loop.init)
        self.assertIsNotNone(self.trainer._task_runners["adhoc"])
        self.assertTrue(len(self.trainer._task_runners["adhoc"].batch_vars) > 0)
        self.trainer.pause()    # NOTE: pauses adhoc loop now
        time.sleep(.5)
        self.assertTrue(self.trainer._task_runners["adhoc"].test_loop.paused)
        self.assertFalse(self.trainer._task_runners["adhoc"].test_loop.finished)
        self.assertFalse(self.trainer._task_runners["adhoc"].test_loop.init)
        self.trainer._abort_current_run_cb()
        time.sleep(.5)
        self.assertTrue(self.trainer._task_runners["adhoc"].test_loop.finished)
        # self.trainer.pause()
        # time.sleep(.5)
        # er = self.trainer._task_runners["main"]
        # ar = self.trainer._task_runners["adhoc"]
        # import ipdb; ipdb.set_trace()
        self.assertFalse(self.trainer._task_runners["adhoc"].running)
        self.assertTrue(self.trainer._task_runners["main"].running)
        self.assertEqual(self.trainer.current_state, "main_running_train")
        self.trainer._abort_current("test")
        # import ipdb; ipdb.set_trace()

    # def test_user_adhoc_main(self):
    #     pass

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
