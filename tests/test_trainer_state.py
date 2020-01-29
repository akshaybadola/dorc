import unittest
import sys
import time
from setup import config
import torch
sys.path.append("../")
from trainer.trainer import Trainer


class TrainerTest(unittest.TestCase):
    def setUp(self):
        """Setup a simple trainer with MNIST dataset."""
        self.config = config
        self.trainer = Trainer(**self.config)
        self.trainer._init_all()

    def test_train_transitions(self):
        # Should have subtest for a set of transitions I guess. I can generate
        # predicates and combinatorial states according to that
        # This link has a good post about it
        # https://www.caktusgroup.com/blog/2017/05/29/subtests-are-best/
        self.assertIsInstance(self.trainer.train_loader, torch.utils.data.DataLoader)
        self.trainer._transition(self.trainer.current_state, "normal_running_train")
        self.assertFalse(self.trainer.paused)
        time.sleep(3)
        self.assertTrue("main" in self.trainer._threads)
        self.trainer._transition(self.trainer.current_state, "normal_paused_train")
        time.sleep(1)
        self.assertTrue(self.trainer.paused)
        self.assertTrue(self.trainer._epoch_runner.running)
        self.assertTrue(self.trainer._epoch_runner.waiting)
        time.sleep(1)
        self.trainer._finish_if_paused_or_running(True)
        time.sleep(1)
        print(self.trainer._epoch_runner)
        self.assertFalse(self.trainer._epoch_runner.running)
        self.assertFalse(self.trainer._epoch_runner.waiting)
        self.assertTrue([x for x in self.trainer._epoch_runner.batch_vars])
        self.assertFalse(self.trainer._threads["main"].is_alive())
        time.sleep(1)
        self.trainer._run_new_if_finished()
        print(self.trainer._epoch_runner)
        # self.assertEqual(self.trainer._epoch_runner.current_loop, "train")
        self.trainer.pause()
        import ipdb; ipdb.set_trace()
        # assert hooks were run?

    def test_state_machine_transitions(self):
        results = [True, False, True, False, True, True, True, False, True, False, True,
                   False, False, True, False, True, True, True, True]
        for i, x in enumerate([
                self.trainer._allowed_transition("normal_paused_none", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_train", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_train", "normal_running_train"),
                self.trainer._allowed_transition("normal_running_train", "normal_running_train"),
                self.trainer._allowed_transition("normal_running_train", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_none", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_none", "normal_running_train"),
                self.trainer._allowed_transition("normal_paused_train", "normal_paused_eval"),
                self.trainer._allowed_transition("normal_finished_train", "normal_paused_eval"),
                self.trainer._allowed_transition("normal_paused_train", "force_paused_eval"),
                self.trainer._allowed_transition("normal_paused_train", "force_running_eval"),
                self.trainer._allowed_transition("normal_running_train", "force_running_eval"),
                self.trainer._allowed_transition("normal_running_train", "force_running_eval"),
                self.trainer._allowed_transition("normal_paused_train", "force_running_eval"),
                self.trainer._allowed_transition("force_running_eval", "normal_paused_train"),
                self.trainer._allowed_transition("force_finished_eval", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_train", "normal_finished_train"),
                self.trainer._allowed_transition("normal_paused_train", "force_finished_stop"),
                self.trainer._allowed_transition("force_finished_stop", "normal_running_train")]):
            with self.subTest(i=i):
                self.assertEqual(x, results[i])


if __name__ == '__main__':
    unittest.main()
