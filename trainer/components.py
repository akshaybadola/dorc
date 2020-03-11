from typing import List, Dict, Iterable, Any, Union, Tuple
import torch
import logging
import traceback


class Models:
    def __init__(self, models: Dict[str, torch.nn.Module],
                 # optimizers: Dict[str, torch.optim.optimizer.Optimizer],
                 optimizers: Dict[str, Any],
                 devices: Dict[str, Union[str, torch.device]],
                 gpus, logger: logging.Logger):
        assert all(isinstance(x, dict) for x in [models, optimizers, devices])
        assert set(models.keys()) == set(optimizers.keys())
        assert set(models.keys()) == set(devices.keys())
        self._optimizers = optimizers
        self._devices = devices
        self._models: Dict[str, torch.nn.Module] = {}
        self._gpus = gpus
        self._logger = logger
        # model attributes are added to torch.nn.Module directly
        self._status = []
        for k, model in models.items():
            status, message = self._check(model)
            if status:
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
            self._status.append((status, message))

    @property
    def status(self) -> Iterable[Tuple[bool, str]]:
        return self._status

    def _check(self, model) -> Tuple[bool, str]:
        if not hasattr(model, "forward"):
            return False, f"No forward call in model {type(model).__qualname__}"
        elif not hasattr(model, "_device"):
            return True, f"No \"_device\" attr in model {type(model).__qualname__}. "
        elif not hasattr(model, "to_"):
            return True, f"No \"to_\" attr in model {type(model).__qualname__}. "
        else:
            return True, "Edge Case?"

    def set_device(self, model_name: str, device: Union[str, torch.device]):
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

    def __iter__(self):
        return iter(self._models)

    @property
    def names(self) -> List[str]:
        return [*self._models.keys()]

    @property
    def devices(self) -> Dict[str, torch.device]:
        return {k: v._device for k, v in self._models.items()}

    def load_weights(self, model_name: str, state_dict: Dict[str, torch.Tensor]) ->\
            Tuple[bool, Union[str, None]]:
        try:
            for k in state_dict:
                state_dict[k] = self._models[model_name].to_(state_dict[k])
            self._models[model_name].load_state_dict(state_dict)
            return True, None
        except Exception as e:
            return False, f"{e}" + f"\n{traceback.format_exc()}"

    def add(self, model: torch.nn.Module, params: Dict[str, Any]):
        """Add model to self and initialize it according to the params

        :param params: params for the model. :class:`dict` with keys
        ``["name", "optimizer", "optimizer_state", "device"]``
        :returns: None
        :rtype: None

        """
        if self._check(model)[0]:
            name = params["name"]
            self._models[name] = model
            self._models[name]._name = name
            self._models[name]._optimizer = params["optimizer"]
            self._models[name]._optimizer_name = params["optimizer_name"]
            self._models[name]._device = params["device"]

    # FIXME: There could be issues in dumping and loading with device allocation
    def dump(self):
        temp = {}
        for name, model in self._models.items():
            _model_state = {"name": model._name, "optimizer_name": model._optimizer_name,
                            "optimizer_state": model._optimizer.state_dict(),
                            "device": str(model._device), "state_dict": model.state_dict()}
            temp[name] = _model_state
        return temp

    # CHECK: Maybe optimizer can be none? In case there's nothing to optimize? Not sure.
    #        Also how to exclude parameters of model?
    def load(self, model_states: Dict[str, Any]) -> Tuple[bool, Union[str, None]]:
        for name, state in model_states.items():
            params = {"name": state["name"], "optimizer": state["optimizer"],
                      "optimizer_name": state["optimizer_name"], "device": state["device"]}
            if name not in self.names:
                self._logger.debug(f"New model {name}")
                self.add(name, params)
            if self._models[name]._optimizer_name == state["optimizer_name"]:
                self._models[name]._optimizer.load_state_dict(state["optimizer_state"])
            else:
                self._logger.warn(f"Optimizers are different." +
                                  " Could not load state dict for model {name}")
            self._models[name].load_state_dict(state["state_dict"])
            self._models[name]._device = state["device"]
        return True, None
