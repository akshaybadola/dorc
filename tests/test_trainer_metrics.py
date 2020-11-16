import os
import shutil
import time
from datetime import datetime
import unittest
import sys
from _setup_local import config
sys.path.append("../")
from trainer.device import all_devices, useable_devices
from trainer.trainer import Trainer


class SubTrainer(Trainer):
    def __init__(self, _cuda, *args, **kwargs):
        self._cuda = _cuda
        super().__init__(*args, **kwargs)

    @property
    def have_cuda(self):
        return self._cuda


class TrainerTestInitLoadSave(unittest.TestCase):
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
        cls.params = {"data_dir": cls.data_dir, **cls.config}
        cls.trainer = SubTrainer(False, **cls.params)
        cls.trainer.reserved_gpus = []
        cls.trainer.reserve_gpus = lambda x: [True, None]
        cls.trainer._trainer_params["cuda"] = True

    def test_trainer_init(self):
        self.trainer._init_all()
        self.assertFalse(self.trainer._have_resumed)
        self.assertTrue(self.trainer.paused)

    def test_trainer_epoch_captures_metrics(self):
        self.trainer._init_all()
        self.trainer.start()
        time.sleep(2)
        self.trainer.pause()
        time.sleep(1)
        metrics = self.trainer.gather_metrics(self.trainer._epoch_runner)
        self.assertIn("train", metrics)
        self.assertIn("test", metrics)
        self.assertNotIn("val", metrics)
        self.assertIn("loss", metrics["train"])
        self.assertIn("num_datapoints", metrics["train"])
        self.assertTrue(metrics["train"]["num_datapoints"] > 0)
        self.assertTrue(metrics["train"]["loss"] > 0)
        self.trainer.abort_loop()

    def test_trainer_log_iter(self):
        self.new_config = self.config.copy()
        self.new_config["trainer_params"]["training_steps"] = ["iterations"]
        self.new_config["trainer_params"]["max_iterations"] = 800
        self.new_config["trainer_params"]["hooks_run_iter_frequency"] = 200
        self.new_config["dataloader_params"]["train"]["batch_size"] = 64
        self.new_trainer = Trainer(**{"data_dir": self.data_dir, **self.new_config})
        self.new_trainer._init_all()
        self.new_trainer.start()
        while not self.new_trainer.iterations:
            time.sleep(1)
        self.new_trainer.pause()
        time.sleep(2)
        self.assertTrue(self.new_trainer.metrics["train"])
        self.assertIn("loss", self.new_trainer.metrics["train"])
        self.assertTrue(self.new_trainer.metrics["train"]["loss"][0])
        self.new_trainer.abort_loop()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
