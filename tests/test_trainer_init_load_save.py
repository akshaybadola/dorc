import os
import shutil
import torch
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


class TrainerTestDevice(unittest.TestCase):
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

    def setUp(self):
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer._trainer_params["cuda"] = True

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
