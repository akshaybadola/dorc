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

    def test_trainer_load_weights_no_gpu(self):
        self.params["model_params"] = {"net_1": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"},
                                       "net_2": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"}}
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer._trainer_params["gpus"] = []
        self.trainer._trainer_params["cuda"] = True
        self.trainer._init_device()
        self.trainer._init_models()
        req = FakeRequest()
        with self.subTest(i="no_file_given"):
            req.form["model_names"] = json.dumps(["net_1", "net_2"])
            status, response = self.trainer.load_weights(req)
            self.assertFalse(status)
        with self.subTest(i="weights_for_only_one_model_given"):
            tmp = open("_test_weights.pth", "rb")
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
            req.form["model_names"] = json.dumps(["net_1", "net_2"])
            req.files["file"] = tmp
            status, response = self.trainer.load_weights(req)
            tmp.close()
            self.assertTrue(status)

    @unittest.skipIf(len(all_devices()) < 2, f"Cannot run without at least 2 GPUs.")
    def test_trainer_load_weights_two_models_two_gpus(self):
        self.params["model_params"] = {"net_1": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"},
                                       "net_2": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"}}
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer._trainer_params["gpus"] = [0, 1]
        self.trainer._trainer_params["cuda"] = True
        self.trainer._init_device()
        self.trainer._init_models()
        req = FakeRequest()
        with self.subTest(i="no_file_given"):
            req.form["model_names"] = json.dumps(["net_1", "net_2"])
            status, response = self.trainer.load_weights(req)
            self.assertFalse(status)
        with self.subTest(i="weights_for_only_one_model_given"):
            tmp = open("_test_weights.pth", "rb")
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
            req.form["model_names"] = json.dumps(["net_1", "net_2"])
            req.files["file"] = tmp
            status, response = self.trainer.load_weights(req)
            tmp.close()
            self.assertTrue(status)
            self.assertTrue(all(x.device == torch.device("cuda:1")
                                for x in self.trainer._models["net_1"].weights.values()))
            self.assertTrue(all(x.device == torch.device("cuda:0")
                                for x in self.trainer._models["net_0"].weights.values()))

    def test_trainer_get_state(self):
        self.params["model_params"] = {"net_1": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"},
                                       "net_2": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"}}
        self.trainer = Trainer(**self.params)
        self.trainer._init_all()
        state = self.trainer._get_state()
        for x in self.trainer._save_and_load_keys:
            with self.subTest(i=x):
                self.assertIn(x, state)

    # FIXME: This test should be better
    def test_trainer_fix_state(self):
        self.params["model_params"] = {"net_1": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"},
                                       "net_2": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"}}
        self.trainer = Trainer(**self.params)
        self.trainer._init_all()
        state = self.trainer._get_state()
        self.trainer.fix_state(state, "_bleh_bleh")

    def test_trainer_save_to_path(self):
        self.params["model_params"] = {"net_1": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"},
                                       "net_2": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"}}
        self.trainer = Trainer(**self.params)
        self.trainer._init_all()
        self.trainer._save("_bleh_bleh")

    def test_trainer_resume_from_path(self):
        self.params["model_params"] = {"net_1": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"},
                                       "net_2": {"model": Net, "optimizer": "Adam",
                                                 "params": {}, "gpus": "auto"}}
        self.trainer = Trainer(**self.params)
        self.trainer._init_all()
        self.trainer._save("_bleh_bleh")
        self.trainer._resume_from_path("_bleh_bleh.pth")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")
        if os.path.exists("_temp_weights.pth"):
            os.remove("_temp_weights.pth")
        if os.path.exists("_bleh_bleh.pth"):
            os.remove("_bleh_bleh.pth")


if __name__ == '__main__':
    unittest.main()
