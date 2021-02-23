from typing import List, Dict, Union, Iterable, Callable, Tuple, Any, Optional
from collections.abc import Sequence
import abc
import torch
import pathlib
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
                 optimizer: Dict[str, Any],
                 gpus: List[int]):
        self._name = name
        self._model_def = model_def
        self._model_params = params
        self._optimizer_name = optimizer["name"]
        self._optimizer_func = optimizer["function"]
        self._optimizer_params = optimizer["params"]
        self._gpus = gpus
        self._model = None
        self._optimizer = None
        self._loaded = False
        self._backup_path = ""

    def __call__(self, x):
        return self._model(x)

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    # NOTE: I tried a from_dump but optimizer["function"] has to be given which
    #       can't be reliably dumped.
    #
    # @classmethod
    # def from_dump(cls, dump) -> Optional["Model"]:
    #     required_keys = ["name", "model_def", "params", "optimizer", "gpus"]
    #     if any(x not in dump for x in required_keys):
    #         if "params" not in dump and "model_params" not in dump:
    #             return None
    #         elif "params" in dump:
    #             params = dump["params"]
    #         elif "model_params" in dump:
    #             params = dump["model_params"]
    #         else:
    #             return None
    #         model = cls(**{**{k: dump[k] for k in required_keys}, "params": params})
    #     else:
    #         model = cls(**{k: dump[k] for k in required_keys})
    #     if "state_dict" in dump:
    #         model.load_into_memory()
    #         model.load_weights(dump["state_dict"])
    #     return model

    def forward(self, x):
        return self.model.forward(x)

    def load_into_memory(self, force: bool = False) -> Tuple[bool, str]:
        if self.loaded and not force:
            return True, "Already loaded. Use force=True to force reload"
        try:
            self._model = self._model_def(**self._model_params)
            self._optimizer = self._optimizer_func(self._model.parameters(),
                                                   **self._optimizer_params)
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
            self._loaded = True
            return True, f"Loaded {self.name}"
        except Exception as e:
            return False, f"Error occured {e}"

    def unload(self, backup_path: Union[pathlib.Path, str]):
        """Unload the model.

        Unloading frees up all resources and deletes all state. Model is
        backedup up to a default path in case it's needed to be loaded again.

        """
        if self.loaded:
            try:
                if backup_path == "RAM":
                    # simply backup to RAM
                    self._model = self._model.cpu()
                    self._device = torch.device("cpu")
                    self._loaded = False
                    self._backup_path = "RAM"
                    return True, f"Unloaded {self.name} to RAM"
                else:
                    torch.save(self.dump(), backup_path)
                    del self._model
                    del self._optimizer
                    self._model = None
                    self._optimizer = None
                    self._loaded = False
                    self._backup_path = backup_path
                    return True, f"Unloaded {self.name} to {backup_path}"
            except Exception as e:
                return False, f"Error occured {e}"
        else:
            return True, f"Already unloaded {self.name}"

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
    def backup_path(self) -> str:
        return self._backup_path

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> torch.device:
        """Return the primary device for the model.

        In case no gpu or single gpu are specified, it's
        straightforward. Otherwise, return the first device from the
        :attr:`_gpus` list.

        """
        if self._gpus == [-1] or self._gpus == [] or not self.loaded:
            return torch.device("cpu")
        else:
            return torch.device(f"cuda:{self._gpus[0]}")

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        "Return model state_dict"
        if self._model is None:
            raise ValueError("Model not initialized")
        else:
            return self._model.state_dict()

    @property
    def gpus(self):
        return self._gpus

    def to_(self, x: Union[torch.Tensor, torch.nn.Module]) ->\
            Union[torch.Tensor, torch.nn.Module]:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        # TODO: Test if this is correct for a module
        elif isinstance(x, torch.nn.Module):
            return x.to(self.device)
        else:
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
            if not self.loaded:
                self.load_into_memory()
            self._model.load_state_dict({k: self.to(v)
                                         for k, v in state_dict["weights"].items()})
            return True, None
        except Exception as e:
            return False, f"{e}" + f"\n{traceback.format_exc()}"

    def dict(self) -> Dict[str, Any]:
        """Dump the current state.

        Returns a dictionary with keys ``[name, params, optimizer, gpus,
        state_dict]``

        If the model is backed up to :attr:`backup_path` then that state is
        returned.

        """
        return {"name": self._name,
                "params": self._model_params,
                "optimizer": {"name": self._optimizer_name,
                              "params": self._optimizer_params},
                "gpus": self._gpus}

    def dump(self, include_def=False) -> Optional[Dict[str, Any]]:
        """Dump the current state.

        Returns a dictionary with keys :code:`[name, params, optimizer, gpus,
        state_dict]` and optional :code:`model_def`

        If the model is backed up to :attr:`backup_path` then that state is
        returned.

        """
        if self.backup_path and self.backup_path != "RAM":
            return torch.load(self.backup_path)
        else:
            dump = {"name": self._name,
                    "params": self._model_params,
                    "optimizer": {"name": self._optimizer_name,
                                  "state_dict": self._optimizer and self._optimizer.state_dict(),
                                  "params": self._optimizer_params},
                    "gpus": self._gpus,
                    "state_dict": self._model and self._model.state_dict()}
            if include_def:
                dump["model_def"] = self._model_def
            return dump

    def to_params(self):
        return {"name": self._name,
                "params": self._model_params,
                "optimizer": self._optimizer_name,
                "gpus": self._gpus,
                "model": self._model_def,
                "loaded": self.loaded}

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
            if not self.loaded:
                return False, "Model not loaded into memory"
            if state["name"] != self._name:
                return False, f"Different model name in state {state['name']}, {self._name}"
            if state["optimizer"]["name"] != self._optimizer_name:
                state_name = state["optimizer"]["name"]
                self_name = self._optimizer_name
                warnings.append(f"Different optimizers in state {state_name}, {self_name}")
            if state["gpus"] != self._gpus:
                warnings.append(f"Different gpus in state {state['gpus']}, {self._gpus}")
            if not state["state_dict"]:
                return False, "No point loading an empty state_dict"
                warnings.append("Was not loaded. Loading into memory")
            if state["optimizer"]["state_dict"]:
                try:
                    self._optimizer.load_state_dict(state["optimizer"]["state_dict"])
                except Exception as e:
                    return False, f"{e}"
            else:
                warnings.append("optimizer state dict not given")
            self_keys = self.weights.keys()
            new_keys = state["state_dict"].keys()
            if set(self_keys) == set(new_keys):
                status, message = self.load_weights({"name": state["name"],
                                                     "weights": state["state_dict"]})
            elif (len(self_keys) == len(new_keys) and
                  [f"module.{x}" in new_keys for x in self_keys]):
                weights = {x: state["state_dict"][f"module.{x}"] for x in self_keys}
                status, message = self.load_weights({"name": state["name"],
                                                     "weights": weights})
            else:
                status, message = False, f"Keys differ"
            if not status:
                return status, message
            else:
                return True, "\n".join(warnings)
        except Exception as e:
            return False, f"{e}" + f"\n{traceback.format_exc()}"


# TODO: Automatic separate code for parallel and non parallel
#       execution
class ModelStep(abc.ABC):
    """A ModelStep is an abstract class for data processing by the model.

    It's a class around a function by which inputs are sent through the model,
    outputs and losses are collected, loss is sent backward and other such tasks
    in a systematic manner. It provides a standard interface for sending and
    collecting data from the models with the :class:`Trainer`.

    Args:
        models: A list of model names OR
                A dictionary of model_names and models.
        criteria_map: A dictionary of model_names to criteria_names
        checks: A dictionary of model_names to validation functions
        logs: Which values to log. Can be anything in :attr:`returns`.
              These values are those which will be reported, rest
              will be discarded.

    A contrived example demonstrating the need and execution flow::

        class ExampleModelStep(ModelStep):
            def __call__(self, batch):
                # assuming self._model_names["foo"] is in self.models
                model_1 = self.models["foo"]
                model_2 = self.models["bar"]
                criterion_1 = self.criteria["foo"]  # criterion for foo
                criterion_2 = self.criteria["bar"]  # criterion for bar
                inputs, labels = batch
                inputs = model_1.to_(inputs)
                labels = model_1.to_(labels)
                if not self.test:
                    model_1.train()
                    model_1._optimizer.zero_grad()
                # NOTE: These should be checked for errors as order of execution may be important
                inter_vals = model_1(inputs)
                loss_1 = criterion_1(inter_vals, labels)
                final_vals = model_2(inter_vals, inputs)
                loss_2 = criterion_2(final_vals, inputs)  # maybe reconstruction loss
                if not self.test:
                    loss_1.backward()
                    loss_2.backward()
                # NOTE: values are model and criterion specific
                return {"losses": {"foo": loss_1.detach().item(), "bar": loss_2.detach().item()},
                        "outputs": {"foo": inter_vals.detach(), "bar": final_vals.detach()},
                        "labels": labels.detach(), "total": len(labels)}


        def check_foo(foo):
            try:
                foo(torch.randn(some_shape))
                return True
            except Exception:
                return False

        # etc.

    Assuming trainer.models is `{"Foo": Foo, "Bar": Bar, "OtherFoo": OtherFoo}`
    and in model_params the criteria are given as `{"foo": "ce_loss", "bar": "mse_loss"}`
    with critera as `{"ce_loss": torch.nn.CrossEntropyLoss, "mse_loss": torch.nn.MSELoss}`
    then::

        example_step = ExampleModelStep(models={"foo": Foo, "bar": Bar},
                                        criteria_map={"foo": "ce_loss", "bar": "mse_loss"},
                                        checks={"foo": check_foo, "bar": check_bar})
        example_step.returns = {"losses", "outputs", "labels", "total"}

        # In trainer the models and criteria will always be thus:
        example_step.train = True  # set to train

        # Executed anywhere with a batch
        retval = example_step(batch)

        # later at test time
        example_step.test = True
        retval = example_step(batch)

        # much later, change only one model, checks performed automatically
        example_step.set_models({"foo": NewModelFoo})
        example_step.set_criteria({"foo": trainer.criteria["some_other_loss"]})
        retval = example_step(batch)

    """

    def __init__(self, models: Union[List[str], Dict[str, Model]],
                 criteria_map: Dict[str, str],
                 checks: Dict[str, Callable[[Model], bool]],
                 criteria: Dict[str, Callable] = {},
                 **kwargs):
        self._test = False
        self._models = models
        self._criteria_map: Dict[str, str] = criteria_map
        self._criteria: Dict[str, Callable] = criteria
        # NOTE: This is set from config by the trainer
        if not all(x in criteria_map for x in self._models):
            raise AttributeError("Must have item in criteria_map for each model")
        if not all(x in criteria_map for x in self._models):
            raise AttributeError("Must have item in criteria_map for each model")
        if not all(x in checks for x in self._models):
            raise AttributeError("Must have checks for each model")
        self._checks = checks
        self._modes = ["train", "val", "test"]
        self._mode = None
        self.__logs: Dict[str, List[str]] = {}
        self.__returns: Dict[str, List[str]] = {}
        if "logs" in kwargs:
            self._logs = kwargs["logs"]
        if "returns" in kwargs:
            self._returns = kwargs["returns"]

    @abc.abstractmethod
    def __call__(self, batch: Iterable) -> Dict:
        """Call the Step

        Args:
            batch: A data specific iterable of values

        :meth:`__call__` is provided by the user and can have different modes.
        Standard modes are `train` and `test`.

        The execution flow and artefacts accumulated can depend on the
        modes. They have to be implemented by the user.

        """
        pass

    def set_criteria(self, criteria: Dict[str, Callable]) ->\
            Union[bool, Dict[str, bool]]:
        """Set the criteria for the models.
        """
        if not all(x in self._models for x in criteria):
            return False
        statuses: Dict[str, bool] = {}
        for key, val in criteria.items():
            if key not in self.models:
                statuses[key] = False
            else:
                self.criteria[key] = val
        return statuses


    def set_models(self, models: Dict[str, Model]) ->\
            Union[bool, Dict[str, bool]]:
        """Set the models which will be used.

        It's only a name mapping. `models` are handled by the trainer, but which
        model will be called is determined dynamically at run time. However
        because :attr:`checks` are `model` and `step` specific so they're
        checked here.

        Args:
            models: :class:`dict` of models which will be set

        ``models`` must be a :class:`dict` like ``{"internal_name": {"external_name": model}}``.

        """
        if not all(x in self._models for x in models):
            return False
        statuses: Dict[str, bool] = {}
        for key, val in models.items():
            status = self.check_model(key, val)
            if status:
                self.models[key] = val
        return statuses

    def set_checks(self, checks: Dict[str, Callable[[Model], bool]]):
        """Set the checks for the models and criteria.

        Args:
            checks: A :class:`dict` of {model_name: check_func} where check_func
                    is a function which takes a :class:`torch.nn.Module`
                    as input and returns an instance of :class:`bool`

        For example, one can verify that the output from each model is of a
        certain shape.  Criteria are more dynamic and are not checked here.

        """
        if not all(x in self._models for x in checks):
            raise ValueError("All model names should be in checks")
        self._checks = checks

    def check_model(self, model_name: str, model: Model) -> bool:
        return self._checks[model_name](model)

    @property
    def model_names(self) -> List[str]:
        """Return the model names."""
        return [*self._models.keys()]

    @property
    def models(self) -> Dict[str, Model]:
        """Return the models."""
        return self._models

    @property
    def criteria_map(self) -> Dict[str, str]:
        return self._criteria_map

    @property
    def criteria(self) -> Dict[str, Callable]:
        return self._criteria

    @property
    def mode(self):
        """:attr:`mode` can be other than :attr:`test` and :attr:`train`"""
        if self._mode is not None:
            return self._mode
        else:
            return "train" if self.train else "test"

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    @property
    def train(self):
        return not self._test

    @train.setter
    def train(self, x):
        self._test = not x

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, x):
        self._test = x

    def returns(self, mode: str) -> List[str]:
        """Names of artefacts returned by :meth:`__call__`.

        The step function can be different modes according to the modes
        supported. :attr:`returns` can be different for different modes.::

            self._returns("train") == {"loss", "total"}
            self._returns("test") == {"loss", "outputs", "label", "total"}
            self._returns("other") == {"loss", "outputs", "label", "total", "other_metric"}

        "total" should always be returned by the step function and is implied.

        """
        if mode not in self._modes:
            raise ValueError(f"Unknown mode {mode}")
        else:
            return self.__returns[mode]

    @property
    def _returns(self) -> Dict[str, List[str]]:
        return self.__returns

    @_returns.setter
    def _returns(self, x: Union[Dict[str, Iterable[str]], List[str]]):
        if isinstance(x, dict) and all(k in self._modes for k in x.keys()):
            for m in x:
                if "total" not in x[m]:
                    raise AttributeError(f"'total' must be present for {m} in {x[m]}")
            self.__returns = x
        elif iter(x):
            if "total" not in x:
                raise AttributeError(f"'total' must be present in returns")
            for m in self._modes:
                self.__returns[m] = [*x]
        else:
            raise TypeError(f"{x} is not iterable")

    def logs(self, mode) -> List[str]:
        """Which items to log from :meth:`returns`

        Args:
            mode: Return logs for that mode. See :attr:`mode`.

        """
        if mode not in self._modes:
            raise ValueError(f"Unknown mode {mode}")
        else:
            return self.__logs[mode]

    @property
    def _logs(self) -> Dict[str, List[str]]:
        """Set :attr:`_logs`
        Setting to `None` means log nothing (unusual).
        """
        return self.__logs

    @_logs.setter
    def _logs(self, x: Union[Dict[str, List[str]], List[str]]):
        if isinstance(x, dict) and all(k in self._modes for k in x.keys()):
            self.__logs = x
        elif isinstance(x, Sequence):
            for m in self._modes:
                self.__logs[m] = x
        else:
            raise ValueError(f"Unknown type of {x}")
