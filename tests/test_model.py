import pytest
import os
import shutil
import unittest
import torch
import sys
from _setup_local import config
sys.path.append("../")
from dorc.device import all_devices
from dorc.trainer.model import Model


def get_model(name, config, gpus):
    _name = "net"
    model_def = config["model_params"][_name]["model"]
    params = config["model_params"][_name]["params"]
    optimizer = {"name": "Adam",
                 **config["optimizers"]["Adam"]}
    return Model(name, model_def, params, optimizer, gpus)


@pytest.mark.quick
class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib
        import _setup_local as setup
        importlib.reload(setup)
        config = setup.config
        cls.config = config
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")
        os.mkdir(".test_dir")
        os.mkdir(".test_dir/test_session")

    def test_model_init_check_gpus(self):
        for i, gpus in enumerate([[], [-1]]):
            with self.subTest(i=i):
                model = get_model("net", self.config, gpus)

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_model_to_single_gpu(self):
        model = get_model("net", self.config, [0])
        x = torch.rand(100, 100)
        x = model.to_(x)
        self.assertEqual(x.device, model.device)

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    @unittest.skipIf(len(all_devices()) < 2, f"Cannot run without at least 2 gpus.")
    def test_model_to_multi_gpu(self):
        model = get_model("net", self.config, [0, 1])
        x = torch.rand(100, 100)
        x = model.to_(x)
        self.assertEqual(x.device, model.device)

    def test_model_dump_not_loaded_no_gpus(self):
        model = get_model("net", self.config, [])
        dump = model.dump(True)
        expected = {"name": str, "params": dict,
                    "optimizer": dict, "gpus": list,
                    "state_dict": type(None)}
        for i in expected.keys():
            with self.subTest(i=i):
                self.assertIn(i, dump)
                self.assertIsInstance(dump[i], expected[i])

    def test_model_dump_loaded_no_gpus(self):
        model = get_model("net", self.config, [])
        model.load_into_memory()
        dump = model.dump()
        expected = {"name": str, "params": dict,
                    "optimizer": dict, "gpus": list,
                    "state_dict": dict}
        for i in expected.keys():
            with self.subTest(i=i):
                self.assertIn(i, dump)
                self.assertIsInstance(dump[i], expected[i])
        self.assertTrue(all(isinstance(x, torch.Tensor) for x in dump["state_dict"].values()))

    @unittest.skipIf(len(all_devices()) < 2, f"Cannot run without at least 2 gpus.")
    def test_model_dump_multi_gpus(self):
        model = get_model("net", self.config, [0, 1])
        model.load_into_memory()
        dump = model.dump()
        expected = {"name": str, "params": dict,
                    "optimizer": dict, "gpus": list,
                    "state_dict": dict}
        for i in expected.keys():
            with self.subTest(i=i):
                self.assertIn(i, dump)
                self.assertIsInstance(dump[i], expected[i])
            self.assertIn("state_dict", expected)
            self.assertTrue(all(isinstance(x, torch.Tensor) for x in dump["state_dict"].values()))

    def test_model_load_weights_no_gpus(self):
        model = get_model("net", self.config, [])
        model.load_into_memory()
        weights = model.dump()["state_dict"]
        status, message = model.load_weights({"name": "net", "weights": weights})
        self.assertTrue(status)

    @unittest.skipIf(not all_devices(), f"Cannot run without gpus.")
    def test_model_load_weights(self):
        model = get_model("net", self.config, [0, 1])
        model.load_into_memory()
        weights = model.dump()["state_dict"]
        status, message = model.load_weights({"name": "net", "weights": weights})
        self.assertTrue(status)

    def test_model_load_not_loaded_into_memory(self):
        subtests = {"no_gpus": [], "have_gpus": all_devices()[:1]}
        for sub_test in subtests:
            devices = subtests[sub_test]
            if sub_test == "have_gpus" and not devices:
                continue
            with self.subTest(i=sub_test):
                model = get_model("net", self.config, devices)
                status, message = model.load({"name": "net",
                                              "optimizer": {"name": "Adam",
                                                            **config["optimizers"]["Adam"],
                                                            "state_dict": None},
                                              "params": {}, "gpus": [0],
                                              "state_dict": None})
                self.assertFalse(status)
                self.assertIn("not loaded", message.lower())

    def test_model_load_no_gpus(self):
        model = get_model("net", self.config, [])
        model.load_into_memory()
        status, message = model.load({"name": "net",
                                      "optimizer": {"name": "Adam",
                                                    **config["optimizers"]["Adam"],
                                                    "state_dict": None},
                                      "params": {}, "gpus": [0],
                                      "state_dict": model.dump()["state_dict"]})
        self.assertTrue(status)
        self.assertIsInstance(message, str)
        self.assertTrue("different gpus" in message.lower())
        self.assertTrue("optimizer" in message.lower())

    @unittest.skipIf(len(all_devices()) < 1, f"Cannot run without gpus.")
    def test_model_load_gpus(self):
        model = get_model("net", self.config, all_devices()[:2])
        model.load_into_memory()
        status, message = model.load({"name": "net",
                                      "optimizer": {"name": "Adam",
                                                    **config["optimizers"]["Adam"],
                                                    "state_dict": None},
                                      "params": {}, "gpus": [0],
                                      "state_dict": model.dump()["state_dict"]})
        self.assertTrue(status)
        self.assertTrue("different gpus" in message.lower())
        self.assertTrue("optimizer" in message.lower() and "not given" in message.lower())

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
