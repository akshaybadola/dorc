class Models:
    def __init__(self, models, optimizers, devices, gpus, logger):
        assert all(isinstance(x, dict) for x in [models, optimizers, devices])
        assert set(models.keys()) == set(optimizers.keys())
        assert set(models.keys()) == set(devices.keys())
        self._models = {}
        self._gpus = gpus
        self._logger = logger
        # model attributes are added to torch.nn.Module directly
        for k, model in models.items():
            assert hasattr(model, "forward"), f"No forward call in model {model.__qualname__}"
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
            self._models[model_name] = self._models[model_name].cuda()
            self._models[model_name].to_ = lambda x: x.cuda()
            self._models[model_name].gpus = self._gpus
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

    def load_model(self, model_name, state_dict):
        self._models[model_name].load_state_dict(state_dict)

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
