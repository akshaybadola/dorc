import os
import shutil
import unittest
import torch
import sys
from _setup_local import config
sys.path.append("../")
from trainer.model import Model


def get_model(name, config, gpus):
    _name = "net"
    model_def = config["model_defs"][_name]["model"]
    params = config["model_params"][_name]["params"]
    optimizer = {"name": "adam",
                 **config["optimizer"]["Adam"]}
    return Model(name, model_def, params, optimizer, gpus)


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup a simple trainer with MNIST dataset."""
        cls.config = config
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")
        os.mkdir(".test_dir")
        os.mkdir(".test_dir/test_session")

    def test_model_init_check_gpus(self):
        for i, gpus in enumerate([[], [-1]]):
            with self.subTest(i=i):
                model = get_model("net", self.config, gpus)

    def test_model_to_single_gpu(self):
        model = get_model("net", self.config, [0])
        x = torch.rand(100, 100)
        x = model.to_(x)
        self.assertEqual(x.device, model.device)

    def test_model_to_multi_gpu(self):
        model = get_model("net", self.config, [0, 1])
        x = torch.rand(100, 100)
        x = model.to_(x)
        self.assertEqual(x.device, model.device)

    def test_model_dump(self):
        model = get_model("net", self.config, [0, 1])
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

    def test_model_load_weights(self):
        model = get_model("net", self.config, [0, 1])
        weights = model.dump()["state_dict"]
        status, message = model.load_weights({"name": "net", "weights": weights})
        self.assertTrue(status)

    def test_model_load(self):
        model = get_model("net", self.config, [0, 1])
        status, message = model.load({"name": "net",
                                      "optimizer": {"name": "Adam",
                                                    **config["optimizer"]["Adam"],
                                                    "state_dict": None},
                                      "params": {}, "gpus": [0],
                                      "state_dict": model.dump()["state_dict"]})
        self.assertTrue(status)
        self.assertIsInstance(message, list)
        self.assertTrue(len(message))
        self.assertTrue(any("different gpus" in x.lower() for x in message))
        self.assertTrue(any("optimizer" in x.lower() for x in message))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
