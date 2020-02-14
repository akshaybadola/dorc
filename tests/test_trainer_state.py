import unittest
import sys
import time
from setup import config
import torch
sys.path.append("../")
from trainer.trainer import Trainer


class StateTest(unittest.TestCase):
    def setUp(self):
        """Setup a simple trainer with MNIST dataset."""
        self.config = config
        self.trainer = Trainer(**self.config)
        self.trainer._init_all()

    # def tearDown(self):
    #     self.trainer._abort_current()

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
        self.trainer._abort_current()
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
        self.trainer._abort_current()

    def test_main_adhoc_back(self):
        print(self.trainer._task_runners)
        self.trainer._start_if_not_running()
        time.sleep(.5)
        self.assertFalse(self.trainer.paused)
        self.assertEqual(self.trainer.current_state, "main_running_train")
        print(self.trainer._task_runners)
        time.sleep(1)
        self.assertTrue(self.trainer._epoch_runner.running)
        self.assertTrue("main" in self.trainer._threads)
        self.trainer._user_funcs["test_func"] = lambda x: None
        time.sleep(1)
        self.trainer.adhoc_eval({"test": {"epoch": "current",
                                          "num_or_fraction": 100,
                                          "device": "cpu",
                                          "data": "test",
                                          "callback": "test_func"}})
        time.sleep(1)
        self.assertTrue("adhoc" in self.trainer._task_runners)
        self.assertIsNotNone(self.trainer._task_runners["adhoc"])
        self.assertTrue(len(self.trainer._task_runners["adhoc"].batch_vars) > 0)
        self.trainer.pause()
        time.sleep(.5)
        self.assertTrue(self.trainer._task_runners["adhoc"].test_loop.paused)
        # er = self.trainer._task_runners["main"]
        # ar = self.trainer._task_runners["adhoc"]
        # ar.test_loop.signals.paused.clear()
        # import ipdb; ipdb.set_trace()
        self.trainer._abort_current_run_cb()
        time.sleep(.5)
        self.assertFalse(self.trainer._task_runners["adhoc"].running)

        # import ipdb; ipdb.set_trace()

    def test_main_not_in_task_runner(self):
        pass

    # def test_illegal_states(self):
    #     results = [False, True, False, True, False, True, True, True, False, True,
    #                False, True, False, False, True, False, True, True, True,
    #                True, True, True, True, False]
    #     for i, x in enumerate([
    #             ("normal_paused_none", "normal_running_none"),
    #             ("normal_paused_none", "normal_paused_train"),
    #             ("normal_paused_train", "normal_paused_train"),
    #             ("normal_paused_train", "normal_running_train"),
    #             ("normal_running_train", "normal_running_train"),
    #             ("normal_running_train", "normal_paused_train"),
    #             ("normal_paused_none", "normal_paused_train"),
    #             ("normal_paused_none", "normal_running_train"),
    #             ("normal_paused_train", "normal_paused_eval"),
    #             ("normal_finished_train", "normal_paused_eval"),
    #             ("normal_paused_train", "force_paused_adhoc"),
    #             ("normal_paused_train", "force_running_adhoc"),
    #             ("normal_running_train", "force_running_adhoc"),
    #             ("normal_running_train", "force_running_adhoc"),
    #             ("normal_paused_train", "force_running_adhoc"),
    #             ("force_running_adhoc", "normal_paused_train"),
    #             ("force_finished_adhoc", "normal_paused_train"),
    #             ("normal_paused_train", "normal_finished_train"),
    #             ("normal_paused_train", "force_finished_stop"),
    #             ("force_finished_stop", "normal_running_train"),
    #             ("normal_running_train", "force_paused_train"),
    #             ("force_paused_train", "force_finished_stop"),
    #             ("force_paused_train", "force_running_adhoc"),
    #             ("normal_running_train", "normal_running_adhoc")]):
    #         with self.subTest(i=str(x)):
    #             self.assertEqual(self.trainer._sm.allowed_transition(*x, True), results[i])


if __name__ == '__main__':
    unittest.main()
