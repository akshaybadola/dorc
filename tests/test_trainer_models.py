import os
import pytest
import shutil
import torch
from datetime import datetime
import unittest
import sys
from _setup_local import config
sys.path.append("../")
from dorc.device import all_devices, useable_devices
from dorc.trainer import Trainer


class SubTrainer(Trainer):
    def __init__(self, _cuda, *args, **kwargs):
        self._cuda = _cuda
        super().__init__(*args, **kwargs)

    @property
    def have_cuda(self):
        return self._cuda


@pytest.mark.quick
class TrainerTestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup a simple trainer with MNIST dataset."""
        import importlib
        import _setup_local as setup
        importlib.reload(setup)
        config = setup.config
        cls.config = config
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")
        os.mkdir(".test_dir")
        os.mkdir(".test_dir/test_session")
        time_str = datetime.now().isoformat()
        os.mkdir(f".test_dir/test_session/{time_str}")
        cls.data_dir = f".test_dir/test_session/{time_str}"
        cls.params = {"data_dir": cls.data_dir, **cls.config}
        cls.Net = cls.config["model_params"]["net"]["model"]

    def test_trainer_init_models_one_model_no_gpus(self):
        with self.subTest(i="no_gpus_trainer_params"):
            self.params["model_params"] = {"net": {"model": self.Net,
                                                   "optimizer": "Adam",
                                                   "params": {},
                                                   "gpus": []}}
            self.trainer = SubTrainer(False, **self.params)
            self.trainer.reserved_gpus = []
            self.trainer.reserve_gpus = lambda x: [True, None]
            self.trainer._init_device()
            self.trainer._init_models()
            self.assertIn("net", self.trainer._models)
            self.assertEqual(self.trainer._models["net"].gpus, [])
        with self.subTest(i="gpus_in_trainer_params"):
            self.params["trainer_params"]["gpus"] = [0]
            self.trainer = SubTrainer(False, **self.params)
            self.trainer.reserved_gpus = []
            self.trainer.reserve_gpus = lambda x: [True, None]
            self.trainer._init_device()
            self.trainer._init_models()
            self.assertIn("net", self.trainer._models)
            self.assertEqual(self.trainer._models["net"].gpus, [])

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_trainer_init_models_one_model_one_gpu(self):
        self.params["model_params"] = {"net": {"model": self.Net,
                                               "optimizer": "Adam",
                                               "params": {},
                                               "gpus": [0]}}
        self.params["trainer_params"]["gpus"] = [0]
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer.trainer_params.cuda = True
        self.trainer._init_device()
        self.trainer._init_models()
        self.assertIn("net", self.trainer._models)
        self.assertEqual(self.trainer._models["net"].gpus, [0])

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_trainer_init_models_many_models_no_auto_no_parallel_gpus_sufficient(self):
        self.params["model_params"] = {"net_1": {"model": self.Net,
                                                 "optimizer": "Adam",
                                                 "params": {},
                                                 "gpus": [0]},
                                       "net_2": {"model": self.Net,
                                                 "optimizer": "Adam",
                                                 "params": {},
                                                 "gpus": [1]}}
        self.params["trainer_params"]["gpus"] = [0, 1]
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer.trainer_params.cuda = True
        self.trainer._init_device()
        self.trainer._init_models()
        self.assertIn("net_1", self.trainer._models)
        self.assertIn("net_2", self.trainer._models)
        self.assertEqual(self.trainer._models["net_1"].gpus, [0])
        self.assertEqual(self.trainer._models["net_2"].gpus, [1])

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_trainer_init_models_many_models_no_auto_no_parallel_gpus_deficient(self):
        self.params["model_params"] = {"net_1": {"model": self.Net,
                                                 "optimizer": "Adam",
                                                 "params": {},
                                                 "gpus": [0, 1]},
                                       "net_2": {"model": self.Net,
                                                 "optimizer": "Adam",
                                                 "params": {},
                                                 "gpus": [2, 3]}}
        self.params["trainer_params"]["gpus"] = [0, 1]
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer.trainer_params.cuda = True
        self.trainer._init_device()
        self.trainer._init_models()
        self.assertIn("net_1", self.trainer._models)
        self.assertIn("net_2", self.trainer._models)
        self.assertEqual(self.trainer._models["net_1"].gpus, [0, 1])
        self.assertEqual(self.trainer._models["net_2"].gpus, [])

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_trainer_init_models_many_models_no_auto_no_parallel_gpus_conflict(self):
        self.params["model_params"] = {"net_1": {"model": self.Net,
                                                 "optimizer": "Adam",
                                                 "params": {},
                                                 "gpus": [0, 1]},
                                       "net_2": {"model": self.Net,
                                                 "optimizer": "Adam",
                                                 "params": {},
                                                 "gpus": [0]}}
        self.params["trainer_params"]["gpus"] = [0, 1]
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer.trainer_params.cuda = True
        self.trainer._init_device()
        self.trainer._init_models()
        self.assertIn("net_1", self.trainer._models)
        self.assertIn("net_2", self.trainer._models)
        self.assertEqual(self.trainer._models["net_1"].gpus, [0, 1])
        self.assertEqual(self.trainer._models["net_2"].gpus, [])

    @unittest.skipIf(len(all_devices()) < 2, f"Cannot run without at least 2 GPUs.")
    def test_trainer_init_models_two_models_two_gpus_auto_no_parallel(self):
        bleh = self.Net

        class Net2(torch.nn.Module):
            def __init__(fles):
                super().__init__()
                fles.a = bleh()
                fles.b = bleh()

            def forward(fles, x):
                x_a = fles.a(x)
                x_b = fles.b(x)
                return x_a + x_b

        self.params["model_params"] = {"net_1": {"model": self.Net,
                                                 "optimizer": "Adam",
                                                 "params": {},
                                                 "gpus": "auto"},
                                       "net_2": {"model": Net2,
                                                 "optimizer": "Adam",
                                                 "params": {},
                                                 "gpus": "auto"}}
        self.params["trainer_params"]["gpus"] = [0, 1]
        self.trainer = SubTrainer(False, **self.params)
        self.trainer.reserved_gpus = []
        self.trainer.reserve_gpus = lambda x: [True, None]
        self.trainer.trainer_params.cuda = True
        self.trainer._init_device()
        self.trainer._init_models()
        self.assertIn("net_1", self.trainer._models)
        self.assertIn("net_2", self.trainer._models)
        self.assertEqual(self.trainer._models["net_1"].gpus, [1])
        self.assertEqual(self.trainer._models["net_2"].gpus, [0])

    def test_trainer_load_models_state(self):
        pass

    def test_trainer_get_new_models(self):
        pass

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
