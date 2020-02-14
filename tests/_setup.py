import torch
from torchvision import datasets, transforms


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
        with torch.no_grad():
            inputs = model.to_(inputs)
            labels = model.to_(labels)
            outputs = model(inputs)
            loss = criteria[self._criterion_name](outputs, labels)
            return {"loss": loss.detach().item(), "outputs": outputs.detach(),
                    "labels": labels.detach(), "total": len(labels)}


config = {}
config["optimizer"] = {"Adam": {"function": torch.optim.Adam,
                                "params": {"lr": 0.01,
                                           "weight_decay": 0}}}
config["criteria"] = {"criterion_ce_loss":
                      {"function": torch.nn.CrossEntropyLoss, "params": {}}}
config["uid"] = "test_trainer"
config["extra_metrics"] = None
config["trainer_params"] = {"gpus": "", "cuda": False, "seed": 1111,
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
config["update_functions"] = {"train": ClassificationTrainStep("net", "criterion_ce_loss"),
                              "val": ClassificationTestStep("net", "criterion_ce_loss"),
                              "test": ClassificationTestStep("net", "criterion_ce_loss")}
