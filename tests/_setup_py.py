import torch
from torchvision import datasets, transforms
from global_modules.autoloads import ClassificationStep


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


def identity(x):
    return True


config = {}
config["optimizers"] = {"Adam": {"function": torch.optim.Adam,
                                 "params": {"lr": 0.01,
                                            "weight_decay": 0}}}
config["criteria"] = {"criterion_ce_loss":
                      {"function": torch.nn.CrossEntropyLoss, "params": {}}}
config["extra_metrics"] = {}
config["trainer_params"] = {"gpus": "", "cuda": False, "seed": 1111,
                            "resume": False, "resume_best": None,
                            "resume_dict": None, "init_weights": None,
                            "training_steps": ["train", "val", "test"],
                            "training_type": "epoch",
                            "check_func": None, "max_epochs": 100, "load_all": True}
config["data_params"] = {"name": "mnist",
                         "train": datasets.MNIST('.data',
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
                                         "pin_memory": False},
                               "val": None,
                               "test": {"batch_size": 32,
                                        "num_workers": 0,
                                        "shuffle": False,
                                        "pin_memory": False}}
config["model_params"] = {"net": {"model": Net, "optimizer": "Adam", "params": {}, "gpus": "auto"}}
config["update_functions"] = {"function": ClassificationStep,
                              "params": {"models": ["net"],
                                         "criteria_map": {"net": "criterion_ce_loss"},
                                         "checks": {"net": identity},
                                         "logs": ["loss"]}}
config["log_levels"] = {"file": "error", "stream": "error"}
