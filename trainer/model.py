from typing import List, Dict, Any, Union, Tuple, Callable
import torch
import traceback


class Model:
    """An abstraction for a model.

    Currently only :class:`torch.nn.Module` models are supported but the
    abstraction means that it can easily be extended.

    Args:
        name: Name of the model
        model_def: A :class:`callable` which returns a model
        params: parameters which are fed to model_def
        optimizer: An dictionary containing name, params and function for optimizer
        gpus: A list of gpu indices

    """
    def __init__(self, name: str, model_def: Callable[..., torch.nn.Module],
                 params: Dict,
                 optimizer: Dict[str, Union[Callable[..., torch.optim.optimizer.Optimizer],
                                            str, Dict]],
                 gpus: List[int]):
        self._name = name
        self._model_def = model_def
        self._model_params = params
        self._optimizer_name = optimizer["name"]
        self._optimizer_func = optimizer["function"]
        self._optimizer_params = optimizer["params"]
        self._gpus = gpus
        self._init()

    def __call__(self, x):
        return self._model(x)

    def forward(self, x):
        return self.model.forward(x)

    def _init(self):
        self._model = self._model_def(**self._model_params)
        self._optimizer = self._optimizer_func(**self._optimizer_params)
        if self._gpus == [-1] or self._gpus == []:
            self._device = torch.device("cpu")
        elif len(self._gpus) == 1:
            self._model._to_device = lambda x: x.cuda(self._gpus[0])
            # TODO:
            # 1. one model many gpus -> dataparallel
            # 2. one model one gpu -> obvious
            # 3. many models many gpus -> tricky
            pass

    def reinit(self):
        self._init()

    @property
    def optimizer_name(self):
        return self._optimizer_name

    @property
    def optimizer(self) -> torch.optim.optimizer.Optimizer:
        return self._optimizer

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> torch.device:
        """Return the primary device for the model.

        In case no gpu or single gpu are specified, it's
        straightforward. Otherwise, return the first device from the
        :attr:`_gpus` list.

        """
        if self._gpus == [-1]:
            return torch.device("cpu")
        else:
            return torch.device("cuda:" + self._gpus[0])

    @property
    def gpus(self):
        return self._gpus

    def to(self, x: Union[torch.Tensor, torch.nn.Module]) ->\
            Union[torch.Tensor, torch.nn.Module]:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        else:  # What to do for a module?
            return x

    def _check(self) -> Tuple[bool, str]:
        if not hasattr(self._model, "forward"):
            return False, f"No forward call in model {type(self._model).__qualname__}"
        # FIXME: Check if the code is commented
        # if "cuda()" in inspect.getsource(model.__class__):
        #     return False, f"cuda() call in model code {type(model).__qualname__}"
        # if "import ipdb" in inspect.getsource(model.__class__) or\
        #    "import pdb" in inspect.getsource(model.__class__):
        #     return False, f"i/pdb imported in model code {type(model).__qualname__}"
        else:
            return True, "Edge Case?"

    def load_weights(self, state_dict: Dict[str, torch.Tensor]) ->\
            Tuple[bool, Union[str, None]]:
        # FIXME
        # Full state dict has to be allocated correctly
        try:
            self._model.load_state_dict(self.to(state_dict))
            return True, None
        except Exception as e:
            return False, f"{e}" + f"\n{traceback.format_exc()}"

    def dump(self):
        return {"name": self._name,
                "params": self._model_params,
                "optimizer": {"name": self._optimizer_name,
                              "state_dict": self._optimizer.state_dict(),
                              "params": self._optimizer_params},
                "gpus": self._gpus,
                "state_dict": self._model.state_dict()}

    # TODO: Allow loading a different optimizer
    def load(self, state: Dict[str, Any]) -> Tuple[bool, Union[str, None]]:
        try:
            # status = {"model": None, "optimizer": None, "gpus": None}
            if state["name"] != self._name:
                return False, "Different model name in state"
            if state["optimizer"]["name"] != self._optimizer_name:
                return False, "Different optimizer name in state"
            if state["gpus"] != self._gpus:
                return False, "Different gpus in state"
            self._optimizer = self._optimizer_func(**state["optimizer"]["params"])
            self._optimizer.load_state_dict(state["optimizer"]["state_dict"])
            self._model = self._model_def(**state["params"])
            status, message = self.load_weights(state["state_dict"])
            if not status:
                return status, message
            else:
                return True, None
        except Exception as e:
            return False, f"{e}" + f"\n{traceback.format_exc()}"
