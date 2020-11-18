import os
import shutil
import json
import time
from datetime import datetime
import unittest
import sys
from _setup_local import config
sys.path.append("../")
from trainer.device import all_devices, useable_devices
from trainer.trainer import Trainer


class FakeRequest:
    def __init__(self):
        self.form = {}
        self.json = None
        self.files = {}


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

    def test_trainer_load_saves_bad_params(self):
        data = {}
        self.assertEqual(self.trainer.load_saves(data),
                         (False, "[load_saves()] Missing params \"weights\""))
        data = {"weights": "meh"}
        self.assertEqual(self.trainer.load_saves(data),
                         (False, "[load_saves()] Invalid or no such method"))
        data = {"weights": "meh", "method": "load"}
        self.assertEqual(self.trainer.load_saves(data),
                         (False, "[load_saves()] No such file"))

    # def test_trainer_load_weights(self):
    #     req = FakeRequest()
    #     req.form["model_names"] = json.dumps(["net", "net_2"])
    #     req.files["file"] = None
    #     status, response = self.trainer.load_weights(req)
    #     self.assertFalse(status)
    #     with open()
    #     req.files["file"] = None

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
