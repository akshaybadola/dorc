import unittest
import sys
import torch
from torchvision import datasets, transforms
sys.path.append("../")
from trainer.trainer import Trainer


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output


class TrainerTest(unittest.TestCase):
    class ClassificationTrainStep:
        def __init__(self, model_name, criterion_name):
            self._model_name = model_name
            self._criterion_name = criterion_name
            self.returns = {("metric", "loss"), ("", "outputs"), ("", "labels"), ("", "total")}

        def __call__(self, models, criteria, batch):
            model = models[self._model_name]
            inputs, labels = batch
            inputs = model.to_(inputs)
            labels = model.to_(labels)
            model.train()
            model._optimizer.zero_grad()
            outputs = model(inputs)
            loss = criteria[self._criterion_name](outputs, labels)
            loss.backward()
            return {"loss": loss.detach().item(), "outputs": outputs.detach(),
                    "labels": labels.detach(), "total": len(labels)}

    class ClassificationTestStep:
        def __init__(self, model_name, criterion_name):
            self._model_name = model_name
            self._criterion_name = criterion_name
            self.returns = {("metric", "loss"), ("", "outputs"), ("", "labels"), ("", "total")}

        def __call__(self, models, criteria, batch):
            model = models[self._model_name]
            inputs, labels = batch
            model.eval()
            with torch.zero_grad():
                inputs = model.to_(inputs)
                labels = model.to_(labels)
                outputs = model(inputs)
                loss = criteria[self._criterion_name](outputs, labels)
                return {"loss": loss.detach().item(), "outputs": outputs.detach(),
                        "labels": labels.detach(), "total": len(labels)}

    def setUp(self):
        """Setup a simple trainer with MNIST dataset."""
        config = {}
        config["optimizer"] = {"Adam": {"function": torch.optim.Adam,
                                        "params": {"lr": 0.01,
                                                   "weight_decay": 0}}}
        config["criteria"] = {"criterion_ce_loss":
                              {"function": torch.nn.CrossEntropyLoss, "params": {}}}
        config["uid"] = "test_trainer"
        config["extra_metrics"] = None
        config["trainer_params"] = {"gpus": "0,1", "cuda": True, "seed": 1111,
                                    "resume": False, "resume_best": False,
                                    "resume_weights": False, "init_weights": False,
                                    "training_steps": ["train", "val", "test"],
                                    "check_func": None, "max_epochs": 100}
        config["data"] = {"train": datasets.MNIST('.data',
                                                  train=True,
                                                  download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                  ])),
                          "val": None,
                          "test": datasets.MNIST('.data',
                                                 train=False,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.1307,), (0.3081,))
                                                 ]))}
        config["dataloader_params"] = {"train": {"batch_size": 32,
                                                 "num_workers": 0,
                                                 "shuffle": True,
                                                 "pin_memory": True},
                                       "val": None,
                                       "test": {"batch_size": 32,
                                                "num_workers": 0,
                                                "shuffle": False,
                                                "pin_memory": True}}
        config["model_params"] = {"net": {}}
        config["model_defs"] = {"net": {"model": Net, "optimizer": "Adam"}}
        config["update_functions"] = {"train": self.ClassificationTrainStep("net",
                                                                            "criterion_ce_loss"),
                                      "val": self.ClassificationTestStep("net",
                                                                         "criterion_ce_loss"),
                                      "test": self.ClassificationTestStep("net",
                                                                          "criterion_ce_loss")}
        self.config = config
        self.trainer = Trainer(**self.config)

    def test_trainer_init(self):
        self.trainer._init_all()
        self.assertFalse(self.trainer._have_resumed)
        self.assertTrue(self.trainer._paused)

    def test_dataparallel(self):
        pass

    def test_iterations_only(self):
        pass

    def test_post_epoch_hooks(self):
        pass

    def test_update_funcs(self):
        pass

    def test_add_module(self):
        pass

    def test_device_logging(self):
        pass

    def test_load_saves(self):
        data = {}
        self.assertEqual(self.trainer.load_saves(data), (False, "[load_saves()] Missing params \"weights\""))
        data = {"weights": "meh"}
        self.assertEqual(self.trainer.load_saves(data), (False, "[load_saves()] Invalid or no such method"))
        data = {"weights": "meh", "method": "load"}
        self.assertEqual(self.trainer.load_saves(data), (False, "[load_saves()] No such file"))

    def test_trainer_transitions(self):
        # Should have subtest for a set of transitions I guess. I can generate
        # predicates and combinatorial states according to that
        # This link has a good post about it
        # https://www.caktusgroup.com/blog/2017/05/29/subtests-are-best/
        pass

    def test_state_machine_transitions(self):
        results = [True, False, True, False, True, True, True, False, True, False, True,
                   False, False, True, False, True, False]
        for i, x in enumerate([
                self.trainer._allowed_transition("normal_paused_none", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_train", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_train", "normal_running_train"),
                self.trainer._allowed_transition("normal_running_train", "normal_running_train"),
                self.trainer._allowed_transition("normal_running_train", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_none", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_none", "normal_running_train"),
                self.trainer._allowed_transition("normal_paused_train", "normal_paused_eval"),
                self.trainer._allowed_transition("normal_finished_train", "normal_paused_eval"),
                self.trainer._allowed_transition("normal_paused_train", "force_paused_eval"),
                self.trainer._allowed_transition("normal_paused_train", "force_running_eval"),
                self.trainer._allowed_transition("normal_running_train", "force_running_eval"),
                self.trainer._allowed_transition("normal_running_train", "force_running_eval"),
                self.trainer._allowed_transition("normal_paused_train", "force_running_eval"),
                self.trainer._allowed_transition("force_running_eval", "normal_paused_train"),
                self.trainer._allowed_transition("force_finished_eval", "normal_paused_train"),
                self.trainer._allowed_transition("normal_paused_train", "normal_finished_train")]):
            with self.subTest(i=i):
                self.assertEqual(x, results[i])

    # Need to test
    # multi_models, multi_criteria like setup
    # sampling
    # recurrent and other such models
    # rest of the functions


if __name__ == '__main__':
    unittest.main()
