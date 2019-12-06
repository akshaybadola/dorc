import torch


class Models:
    def __init__(self, models, optimizers, devices, gpus, logger):
        assert all(isinstance(x, dict) for x in [models, optimizers, devices])
        assert set(models.keys()) == set(optimizers.keys())
        assert set(models.keys()) == set(devices.keys())
        self._optimizers = optimizers
        self._devices = devices
        self._models = {}
        self._gpus = gpus
        self._logger = logger
        # model attributes are added to torch.nn.Module directly
        for k, model in models.items():
            assert hasattr(model, "forward"), f"No forward call in model {model.__qualname__}"
            assert hasattr(model, "_device"), f"No \"_device\" attr in model {model.__qualname__}."
            assert hasattr(model, "to_"), f"No \"to_\" attr in model {model.__qualname__}."
            model._name = k
            model._optimizer = optimizers[k]["optimizer"]
            model._optimizer_name = optimizers[k]["name"]
            # FIXME: `device`  is referred by the model which shouldn't be the case
            model._device = devices[k]
            self._models[k] = model
            self.set_device(k, devices[k])

    def set_device(self, model_name, device):
        # FIXME: `device`  is referred by the model which shouldn't be the case
        if str(device) == "parallel":
            # NOTE: Doesn't work in captioning case as lengths cannot be split across GPUs
            # TODO: It should be model[devices][0]
            self._models[model_name].to_ = lambda x: x.cuda(self._gpus[0])
            self._models[model_name].gpus = self._gpus
            self._models[model_name] = torch.nn.DataParallel(self._models[model_name], self._gpus)
            self._models[model_name].to_ = lambda x: x.cuda(self._gpus[0])
            self._models[model_name] = self._models[model_name].to_(self._models[model_name])
            # AGAIN!
            self._models[model_name]._device = "parallel"
            self._models[model_name]._name = model_name
            self._models[model_name].gpus = self._gpus
            self._models[model_name]._optimizer = self._optimizers[model_name]["optimizer"]
            self._models[model_name]._optimizer_name = self._optimizers[model_name]["name"]
        else:
            self._models[model_name] = self._models[model_name].to(device)
            self._models[model_name].to_ = lambda x: x.to(device)

    def __getitem__(self, key):
        return self._models[key]

    @property
    def names(self):
        return [*self._models.keys()]

    @property
    def devices(self):
        return {k: v._device for k, v in self._models.items()}

    def load_weights(self, model_name, state_dict):
        self._models[model_name].load_state_dict(state_dict)

    # FIXME: There could be issues in dumping and loading with device allocation
    def dump(self):
        temp = {}
        for name, model in self._models.items():
            _model_state = {"name": model._name, "optimizer": model._optimizer_name,
                            "optimizer_state": model._optimizer.state_dict(),
                            "device": str(model._device), "state_dict": model.state_dict()}
            temp[name] = _model_state
        return temp

    def load(self, model_state):
        assert all(k in self.names for k in model_state)
        for name, model in self._models.items():
            if model._optimizer_name == model_state[model._name]["optimizer"]:
                model._optimizer.load_state_dict(model_state[model._name]["optimizer_state"])
            else:
                self._logger.warn("Optimizers are different. Could not load state dict")
            model._device = model_state[model._name]["device"]
            model.load_state_dict(model_state[model._name]["state_dict"])
