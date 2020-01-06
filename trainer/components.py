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
            assert hasattr(model, "forward"), f"No forward call in model {type(model).__qualname__}"
            if not hasattr(model, "_device"):
                self._logger.warn(f"No \"_device\" attr in model {type(model).__qualname__}. ")
            if not hasattr(model, "to_"):
                self._logger.warn(f"No \"to_\" attr in model {type(model).__qualname__}. ")
            model._name = k
            model._optimizer = self._optimizers[k]["optimizer"]
            model._optimizer_name = self._optimizers[k]["name"]
            # FIXME: `device` is referred by the model which shouldn't be the
            #        case In models_old.py, the model forks execution based on
            #        the device.  _device is patched into the model later. It
            #        can be queried externally but that would add additonal
            #        complexity.
            #
            #        A better solution would be to use decorators. Inject the
            #        code automatically, based on decorators.  e.g., @device
            #
            #        Functions in modern computer programming are meant as
            #        demarcations. Instead of trying to inject code in arbitrary
            #        places it would be better to segregate those areas into
            #        functions and then use decorators. Another way would be a
            #        device context.
            model._device = self._devices[k]
            self._models[k] = model
            self.set_device(k, devices[k])

    def set_device(self, model_name, device):
        """Sets the device and patches the model with _device and to_ attrs

        :param model_name: 
        :param device: 
        :returns: 
        :rtype: 

        """
        
        # FIXME: `device` is referred by the model which shouldn't be the case
        #        In models_old.py, Decoder refers to the _device as
        #        self._device.  Ideally _device should be totally outside the
        #        model, but torch.nn.DataParallel may not always work as not
        #        everything in batch may have to be put on to the devices.
        #
        #        In the case below, the "_device" and "_to" attributes are
        #        patched on to the model. Only way out of this is to implement a
        #        custom dataparallel which allows different batch sizes AND non
        #        device params.
        if str(device) == "parallel":
            # NOTE: model = torch.nn.DataParallel doesn't work in certain cases
            #       e.g., if not all attributes of the batch can be split across
            #       GPUs.
            # TODO: It should be model[devices][0] (NOTE Why? [Thu Jan  2 01:53:12 IST 2020])
            self._models[model_name].to_ = lambda x: x.cuda(self._gpus[0])
            self._models[model_name].gpus = self._gpus
            self._models[model_name] = self._models[model_name].to_(self._models[model_name])
            self._models[model_name]._device = "parallel"
        elif str(device) == "dataparallel":
            self._models[model_name] = torch.nn.DataParallel(
                self._models[model_name], device_ids=self._gpus)
            self._models[model_name].gpus = self._gpus
            self._models[model_name].to_ = lambda x: x.cuda(self._gpus[0])
            self._models[model_name] = self._models[model_name].to_(self._models[model_name])
            self._models[model_name]._device = "dataparallel"
            self._models[model_name]._name = model_name
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

    def add(self, model, params):
        """Add model to self and initialize it according to the params

        :param model_name: Name of the model
        :param model: The model object :class: `torch.nn.Module` for now
        :param params: params for the model
        :returns: None
        :rtype: None

        """
        model_name = params["model_name"]
        self._models[model_name] = model
        self._models[model_name]._optimizer = params["optimizer"]
        self._models[model_name]._optimizer_name = params["optimizer_name"]
        self._models[model_name]._device = params["device"]

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
