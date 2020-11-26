import os
import shutil
from datetime import datetime
import unittest
import sys
import json
import torch
from _setup_local import config, Net
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

    def test_trainer_load_saves_bad_params(self):
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer._trainer_params["cuda"] = True
        data = {}
        self.assertEqual(self.trainer.load_saves(data),
                         (False, "[load_saves()] Missing params \"weights\""))
        data = {"weights": "meh"}
        self.assertEqual(self.trainer.load_saves(data),
                         (False, "[load_saves()] Invalid or no such method"))
        data = {"weights": "meh", "method": "load"}
        self.assertEqual(self.trainer.load_saves(data),
                         (False, "[load_saves()] No such file"))

    def test_trainer_load_weights(self):
        self.params["model_params"] = {"net_1": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"},
                                       "net_2": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"}}
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer._trainer_params["cuda"] = True
        self.trainer._init_device()
        self.trainer._init_models()
        req = FakeRequest()
        with self.subTest(i="no_file_given"):
            req.form["model_names"] = json.dumps(["net_1", "net_2"])
            status, response = self.trainer.load_weights(req)
            self.assertFalse(status)
        with self.subTest(i="weights_for_only_one_model_given"):
            tmp = open("_temp_weights.pth", "rb")
            req.form["model_names"] = json.dumps(["net_1", "net_2"])
            req.files["file"] = tmp
            status, response = self.trainer.load_weights(req)
            tmp.close()
            self.assertFalse(status)
            self.assertIn("given weights", response.lower())
        with self.subTest(i="correct_params"):
            net = torch.load("_test_weights.pth")
            weights = net["net"]
            torch.save({"net_1": weights, "net_2": weights}, "_temp_weights.pth")
            tmp = open("_temp_weights.pth", "rb")
            status, response = self.trainer.load_weights(req)
            tmp.close()
            self.assertTrue(status)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
