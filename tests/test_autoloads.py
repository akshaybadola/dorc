import pytest
import torch
import unittest
import sys
from _setup_local import config
from util import get_batch
sys.path.append("../")
from dorc.autoloads import (ClassificationStep, accuracy, CheckFunc,
                               CheckGreater, CheckGreaterName, CheckLesserName,
                               CheckAccuracy)
from dorc.autoloads import ModelStep
from dorc.trainer.model import Model
from dorc.device import all_devices


Net = config["model_params"]["net"]["model"]


class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Net()
        self.b = Net()

    def forward(self, x):
        x_a = self.a(x)
        x_b = self.b(x)
        return x_a + x_b


class ClassificationStepTwoModels(ModelStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns = {"losses", "outputs", "labels", "total"}

    def __call__(self, batch):
        model_1 = self.models["net_1"]
        model_2 = self.models["net_2"]
        criterion_1 = self.criteria["net_1"]
        criterion_2 = self.criteria["net_2"]
        inputs, labels = batch
        inputs = model_1.to_(inputs)
        labels = model_2.to_(labels)
        if self.train:
            model_1.train()
            model_2.train()
            model_1.optimizer.zero_grad()
            model_2.optimizer.zero_grad()
        outputs_1 = model_1(inputs)
        outputs_2 = model_2(inputs)
        loss_1 = criterion_1(outputs_1, labels)
        loss_2 = criterion_2(outputs_2, labels)
        if self.train:
            loss_1.backward()
            loss_2.backward()
        return {"losses": {"loss_1": loss_1.detach().item(), "loss_2": loss_2.detach().item()},
                "outputs": {"outputs_1": outputs_1.detach(), "outputs_2": outputs_2.detach()},
                "labels": labels.detach(), "total": len(labels)}


@pytest.mark.quick
class AutoloadsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup a simple trainer with MNIST dataset."""
        cls.config = config
        cls.config["model_params"] = {"net_1": {"model": Net,
                                                "optimizer": "Adam",
                                                "params": {},
                                                "gpus": "auto"},
                                      "net_2": {"model": Net2,
                                                "optimizer": "Adam",
                                                "params": {},
                                                "gpus": "auto"}}
        cls.models = {"net_1": Model("net_1", Net, {},
                                     optimizer={"name": "Adam",
                                                "function": torch.optim.Adam,
                                                "params": {}},
                                     gpus=[]),
                      "net_2": Model("net_1", Net2, {},
                                     optimizer={"name": "Adam",
                                                "function": torch.optim.Adam,
                                                "params": {}},
                                     gpus=[])}
        cls.checks = {"net_1": lambda x: True, "net_2": lambda x: True}
        cls.criteria = {"ce_loss": config["criteria"]["criterion_ce_loss"]["function"](
            **config["criteria"]["criterion_ce_loss"]["params"])}

    def test_autoloads_classification_train_step_one_model_no_gpus(self):
        self.step_func = ClassificationStep(models={"net": self.models["net_1"]},
                                            criteria_map={"net": "ce_loss"},
                                            checks={"net": self.checks["net_1"]},
                                            logs=["loss"])
        self.step_func.set_criteria({"net": self.criteria["ce_loss"]})
        self.models["net_1"].load_into_memory()
        self.step_func.train = True
        batch = get_batch()
        retval = self.step_func(batch)
        self.assertEqual([*retval.keys()], ["loss", "outputs", "labels", "total"])

    def test_autoloads_classification_train_step_two_models(self):
        self.step_func = ClassificationStepTwoModels(models=self.models,
                                                     criteria_map={"net_1": "ce_loss",
                                                                   "net_2": "ce_loss"},
                                                     checks=self.checks,
                                                     logs=["losses"])
        self.step_func.set_criteria({"net_1": self.criteria["ce_loss"],
                                     "net_2": self.criteria["ce_loss"]})
        self.models["net_1"].load_into_memory()
        self.models["net_2"].load_into_memory()
        self.step_func.train = True
        batch = get_batch()
        retval = self.step_func(batch)
        self.assertEqual([*retval.keys()], ["losses", "outputs", "labels", "total"])

    @unittest.skipIf(not all_devices(), f"Cannot run without GPUs.")
    def test_autoloads_classification_train_step_one_model_one_gpu(self):
        self.step_func = ClassificationStepTwoModels(models=self.models,
                                                     criteria_map={"net_1": "ce_loss",
                                                                   "net_2": "ce_loss"},
                                                     checks=self.checks,
                                                     logs=["losses"])
        self.step_func.set_criteria({"net_1": self.criteria["ce_loss"],
                                     "net_2": self.criteria["ce_loss"]})
        self.models["net_1"].load_into_memory()
        self.models["net_2"].load_into_memory()
        self.step_func.train = True
        batch = get_batch()
        retval = self.step_func(batch)
        self.assertEqual([*retval.keys()], ["losses", "outputs", "labels", "total"])


if __name__ == '__main__':
    unittest.main()
