# import torch
# from torchvision import datasets, transforms
# import global_modules as gm


source = """
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
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
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
"""

# src_identity = """
# def identity(x):
#     return x
# """

src_identity = "lambda x: x"


# NOTE: This is entirely JSON serializable
config = {}                     # type: ignore
config["optimizers"] = {"Adam": {"function": {"path": "torch.optim.Adam",
                                              "params": {"lr": 0.01,
                                                         "weight_decay": 0}}}}
config["criteria"] = {"criterion_ce_loss":
                      {"function": {"path": "torch.nn.CrossEntropyLoss", "params": {}}}}
config["extra_metrics"] = {}
config["trainer_params"] = {"gpus": "", "cuda": False, "seed": 1111,
                            "resume": False, "resume_best": None,
                            "resume_dict": None, "init_weights": None,
                            "training_steps": ["train", "val", "test"],
                            "check_func": None, "max_epochs": 100, "load_all": True}
config["data_params"] = {"name": "mnist",
                         "train": {"function": {"path": "torchvision.datasets.MNIST",
                                                "params": {"root": ".data",
                                                           "train": True,
                                                           "download": True,
                                                           "transform": {"expr": """transforms.Compose([
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))
                                                           ])"""}}}},
                         "val": None,
                         "test": {"function": {"path": "torchvision.datasets.MNIST",
                                               "params": {"root": ".data",
                                                          "train": False,
                                                          "transform": {"expr": """transforms.Compose([
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))
                                                          ])"""}}}}}
config["dataloader_params"] = {"train": {"batch_size": 32,
                                         "num_workers": 0,
                                         "shuffle": True,
                                         "pin_memory": False},
                               "val": None,
                               "test": {"batch_size": 32,
                                        "num_workers": 0,
                                        "shuffle": False,
                                        "pin_memory": False}}
config["model_params"] = {"net": {"model": {"function": {"name": "Net",
                                                         "params": {},
                                                         "source": source}},
                                  "optimizer": "Adam",
                                  "gpus": "auto"}}
config["model_step_params"] = {"function":
                               {"path": "ClassificationStep",
                                "params": {"models": ["net"],
                                           "criteria_map": {"net": "criterion_ce_loss"},
                                           "checks": {"net": {"expr": src_identity}},
                                           "logs": ["loss"]}}}
config["log_levels"] = {"file": "error", "stream": "error"}
config["load_modules"] = {"module": {"source": """
import torch
import torchvision
from dorc.autoloads import ClassificationStep
from dorc.trainer.model import Model
from torchvision import transforms
module_exports = {"torch": torch, "torchvision": torchvision, "transforms": transforms,
                  "ClassificationStep": ClassificationStep, "Model": Model}
"""
}}
