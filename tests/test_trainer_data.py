import os
import shutil
from datetime import datetime
import unittest
import sys
from _setup_local import config
sys.path.append("../")
from trainer.device import all_devices, useable_devices
from trainer.trainer import Trainer
from trainer.trainer import ParamsError


class TrainerTestData(unittest.TestCase):
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

    def test_trainer_init_data_check_params(self):
        self.params = {"data_dir": self.data_dir, **self.config}
        self.trainer = Trainer(**self.params)  # no error
        self.params["dataloader_params"]["val"] = {"batch_size": 32,
                                                   "num_workers": 0,
                                                   "shuffle": False,
                                                   "pin_memory": False}
        try:
            self.trainer = Trainer(**self.params)
        except Exception as e:
            self.assertIsInstance(e, ParamsError)
            self.assertEqual(e.params_type, "data")

    def test_trainer_init_data_function_in_dataloader(self):
        pass

    def test_trainer_dataloaders(self):
        pass

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
