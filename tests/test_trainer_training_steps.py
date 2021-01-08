import os
import shutil
import unittest
import pytest
import torch
import sys
# from _setup_local import config
from util import get_step, get_model, get_model_batch, get_batch
sys.path.append("../")
from trainer.device import all_devices


@pytest.mark.ci
class TrainingStepsTest(unittest.TestCase):
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

    def test_train_step_no_gpu(self):
        model = get_model("net", self.config, [])
        train_step = get_step({"net": model}, self.config, "train")
        retval = train_step(get_batch())
        self.assertTrue(all(x in retval for x in train_step.returns))

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_train_step_single_gpu(self):
        model, batch = get_model_batch("net", self.config, [0])
        train_step = get_step({"net": model}, "train")
        retval = train_step(get_batch())
        self.assertTrue(all(x in retval for x in train_step.returns))

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_val_step_single_gpu(self):
        model, batch = get_model_batch("net", self.config, [0])
        val_step = get_step({"net": model}, "val")
        retval = val_step(batch)
        self.assertTrue(all(x in retval for x in val_step.func.returns))

    # NOTE: how do we test that it is actually parallelized?
    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_dataparallel(self):
        model, batch = get_model_batch("net", self.config, [0, 1])
        train_step = get_step({"net": model}, "train")
        retval = train_step(batch)
        self.assertTrue(all(x in retval for x in train_step.returns))
        self.assertIsInstance(model._model, torch.nn.DataParallel)
        self.assertEqual(retval["outputs"].device, torch.device(0))

    def test_distributed_data_parallel(self):
        pass

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_model_parallel(self):
        pass

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".test_dir"):
            shutil.rmtree(".test_dir")


if __name__ == '__main__':
    unittest.main()
