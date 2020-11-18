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
                 optimizer: Dict[str, Union[Callable[..., torch.optim.Optimizer],
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

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def forward(self, x):
        return self.model.forward(x)

    def _init(self):
        self._model = self._model_def(**self._model_params)
        self._optimizer = self._optimizer_func(self._model.parameters(), **self._optimizer_params)
        if self._gpus == [-1] or self._gpus == []:
            self._device = torch.device("cpu")
        elif len(self._gpus) == 1:
            self._device = torch.device(f"cuda:{self._gpus[0]}")
            self._model = self._model.to(self._device)
            self._model._to_device = lambda x: x.cuda(self._gpus[0])
        else:
            self._device = "dataparallel"
            self._model = self._model.to(torch.device(f"cuda:{self._gpus[0]}"))
            self._model = torch.nn.DataParallel(self._model, device_ids=self._gpus)
            self._model._to_device = lambda x: x.cuda(self._gpus[0])

    def reinit(self):
        self._init()

    def to(self, x):
        return self.to_(x)

    @property
    def optimizer_name(self):
        return self._optimizer_name

    @property
    def optimizer(self) -> torch.optim.Optimizer:
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
            return torch.device(f"cuda:{self._gpus[0]}")

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        "Return model state_dict"
        return self._model.state_dict()

    @property
    def gpus(self):
        return self._gpus

    def to_(self, x: Union[torch.Tensor, torch.nn.Module]) ->\
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

    def load_weights(self, state_dict: Dict[str, Dict[str, torch.Tensor]],
                     force: bool = False) -> Tuple[bool, Union[str, None]]:
        """Load given weights.

        `name` and `weights` must be present and `name` must match with
        model.

        """
        # CHECK Full state dict is allocated correctly
        try:
            if self.name != state_dict["name"] and not force:
                return False, "Names don't match. Set force=True to force loading weights"
            self._model.load_state_dict(self.to(state_dict["weights"]))
            return True, None
        except Exception as e:
            return False, f"{e}" + f"\n{traceback.format_exc()}"

    def dump(self) -> Dict[str, Any]:
        """Dump the current state.

        Returns a dictionary with keys ``[name, params, optimizer, gpus,
        state_dict]``

        """
        return {"name": self._name,
                "params": self._model_params,
                "optimizer": {"name": self._optimizer_name,
                              "state_dict": self._optimizer.state_dict(),
                              "params": self._optimizer_params},
                "gpus": self._gpus,
                "state_dict": self._model.state_dict()}

    def load(self, state: Dict[str, Any]) -> Tuple[bool, Union[str, None]]:
        """Load the entire model state.

        `model_name` and `state_dict` must be present in the state.  `state`
        should be of the same form as :meth:`self.dump`. state["name"] has to be
        the same as the current model. `optimizer` (and name) can be different
        and optimizer state_dict need not be given, but that implies that the
        optimizer will start from the beginning.

        If optimizer name is different or state not given or gpus are different,
        the state is still loaded but appropriate warnings are also returned.

        """
        warnings = []
        try:
            # status = {"model": None, "optimizer": None, "gpus": None}
            if state["name"] != self._name:
                return False, "Different model name in state"
            if state["optimizer"]["name"] != self._optimizer_name:
                warnings.append("Different gpus in state")
            if state["gpus"] != self._gpus:
                warnings.append("Different gpus in state")
            self._model = self._model_def(**state["params"])
            self._optimizer = self._optimizer_func(self._model.parameters(),
                                                   **state["optimizer"]["params"])
            # can be None if a new optimizer is specified
            if state["optimizer"]["state_dict"]:
                self._optimizer.load_state_dict(state["optimizer"]["state_dict"])
            else:
                warnings.append("optimizer state dict not given")
            status, message = self.load_weights({"name": state["name"],
                                                 "weights": state["state_dict"]})
            if not status:
                return status, message
            else:
                return True, warnings
        except Exception as e:
            return False, f"{e}" + f"\n{traceback.format_exc()}"
