from typing import List, Dict, Iterable, Any, Union, Tuple, Callable, Optional, cast
import re
import io
import os
import base64
import copy
import time
import json
import torch
import pathlib
import inspect
import traceback
import flask
from functools import partial
from threading import Thread, Event
from PIL import Image
import numpy as np
from multiprocessing.pool import ThreadPool
from torch.utils.data import Dataset, DataLoader

from ..device import (init_nvml, gpu_ranking, gpu_util, gpu_name,
                      cpu_info, memory_info, DeviceMonitor)
from ..util import gen_file_and_stream_logger, deprecated, _dump, concat, diff_as_sets, BasicType
from ..task import Signals
from ..mods import Modules
from ..overrides import MyDataLoader, default_tensorify
from .._log import Log
from ..helpers import (Tag, ProxyDataset, PropertyProxy, HookDict, Hook,
                       GET, POST, Exposes)
from ..version import __version__

from .epoch import Epoch
from .model import Model, ModelStep
from .models import Return, ReturnBinary, ReturnExtraInfo, AdhocEvalParams, TrainerState, StateEnum
from . import config
from .config import Metric
from . import hooks as hooks_module


PathType = Union[pathlib.Path, str]
control = Tag("control")
prop = Tag("prop")
state_var = Tag("state_var")
extras = Tag("extras")
methods = Tag("methods")
objects = Tag("objects")
objects.__doc__ = """Objects can be queried for properties just like the trainer, although
their methods cannot be called. We can fetch each exposed property via a
@prop tag. A sequence of @prop tags can expose nested props.

For example, Trainer._epoch_runner is an instance of Epoch. Now say
Trainer has an "object" epoch_runner like below which can be exposed.  So
we can access the property "info" of Epoch like trainer/epoch_runner/info
from the HTTP API. If batch_vars is another "object" of type BatchVars and
that is also exposed, then we should be able to access,
trainer/epoch_runner/batch_vars/{prop}, for some "prop" of batch_vars.
"""

internals = Tag("internals")
prop_names = {"saves", "gpus", "system_info", "devices", "models", "active_model",
              "epoch", "max_epochs", "iterations", "max_iterations",
              "updatable_params", "all_attrs", "all_params", "metrics",
              "post_epoch_hooks_to_run", "all_post_epoch_hooks", "items_to_log_dict",
              "current_run", "paused", "best_save", "props", "controls", "methods",
              "extras"}

# Protocol:
# 1. "control" is defined as any method which changes the state of the
#    wrapper, but doesn't require any arguments, therefore doesn't change
#    the attrs of the wrapper
# 2. "update" is any operation that changes the attrs of the wrapper
# 3. "property" is any operation the retrieves an attribute
# TODO: Change the whole "returns" and "expects" paradigm to "requires" and "provides".
# TODO: The trainer should return the controls and properties in a more wholesome way.
#       Currently it's very hacky and will be error prone in the future.

# TODO: Diagnostics:
#       1. Find optimal batch size.
#       2. Report if and where there's shape mismatch between model and data
#       3. Report any crashes with the ability to start again with updated
#          parameters

# TODO: Meh
#       According to the parameters given, there can be different checks which
#       can be applied. That is very very hard to generalize and will simply
#       need a lot of heuristics to solve, which is what we do while we program.

# TODO: Code inspection
#       This is a bit easier than code execution. I have to simply fetch the relevant code
#       and show it in a dialog. I can also fetch the module code if required.

# CHECK from this link
# https://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread-in-python
# class PropagatingThread(Thread):
#     def run(self):
#         self.exc = None
#         try:
#             if hasattr(self, '_Thread__target'):
#                 # Thread uses name mangling prior to Python 3.
#                 self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
#             else:
#                 self.ret = self._target(*self._args, **self._kwargs)
#         except BaseException as e:
#             self.exc = e

#     def join(self):
#         super(PropagatingThread, self).join()
#         if self.exc:
#             raise self.exc
#         return self.ret


def return_image(status: bool, image: str) -> ReturnBinary:
    return ReturnBinary(status=status, image=image)


def make_return(status: bool, message: str) -> Return:
    return Return(status=status, message=message)


def reterr(message: str) -> Return:
    return Return(status=False, message=message)


def make_info(status: bool, message: str, data: Dict = {}) -> ReturnExtraInfo:
    return ReturnExtraInfo(status=status, message=message, data=data)


# TODO: autoprop decorator, expand name to different properties
#       based on their name, remove leading "_"
#       e.g., autoprop(self._training_step) becomes
#       @property
#       def training_step(self):
#           return self._training_step
class Trainer:
    __version__ = __version__

    def __init__(self, model_params, criteria, optimizers, update_functions,
                 extra_metrics, trainer_params, data_params, dataloader_params, data_dir,
                 global_modules_dir, global_datasets_dir,
                 log_levels={"file": "debug", "stream": "info"}):
        """Initializes the :class:`Trainer` object. This is supposed to be a catch all
        trainer which is robust and easy to train and can generate graphs
        automatically etc.

        `model_params`, `criteria`, `trainer_params`, `dataloader_params` are
        stateless parameters.

        `optimizers`, `update_functions` contain callables and as
        such aren't part of config but model and training definitions.

        :param model: model which is a :class:`torch.nn.Module`
        :param model_params: model params where (k, v) are (:class:`str` model_name,
        `list` of model params) :class:`dict`
        :param criteria: `dict` where (k, v) are (`str`, :class:`torch.nn.Module`)
        :param optimizer: `dict` where (k, v) are (`str`, :class:`torch.optim.Optimizer`)
        :param model_init: `dict` where (k, v) are (`str` model_name, :function:
                            returns the initialized model)
        :param train_step_func: :function: which is called for running each batch forward iteration
        :param trainer_params: TODO
        :param train_loader: a train data loader usually :class:`torch.utils.data.Dataloader`
        :param val_loader: a validation data loader usually :class:`torch.utils.data.Dataloader`
        :param test_loader: a test data loader usually :class:`torch.utils.data.Dataloader`

        """
        # DONE: model, train_loader, val_loader should be resettable from the interface
        #       Say, trainer.reset() is called, then the interface should place a hook there
        #       that automatically resets the trainloader and the valloader
        #       Mostly Done.
        # Basic assign parameters

        if "params" in update_functions and update_functions["params"]:
            update_functions = config.UpdateFunctions(
                function=update_functions["function"],
                params=config.UpdateFunctionsParams(**update_functions["params"]))
        # else:
        #     update_functions = update_functions
        #     update_functions = config.UpdateFunctions(**update_functions)
        self.config = config.Config(model_params={k: config.ModelParams(**v)
                                                  for k, v in model_params.items()},
                                    trainer_params=config.TrainerParams(**trainer_params),
                                    criteria={k: config.Criterion(**v)
                                              for k, v in criteria.items()},
                                    optimizers={k: config.Optimizer(**v)
                                                for k, v in optimizers.items()},
                                    data_params=config.DataParams(name=data_params["name"],
                                                                  train=data_params["train"],
                                                                  val=data_params["val"],
                                                                  test=data_params["test"]),
                                    dataloader_params=config.DataLoaderParams(
                                        **dataloader_params),
                                    update_functions=update_functions,
                                    extra_metrics={k: Metric(**v)
                                                   for k, v in extra_metrics.items()},
                                    log_levels=config.LogLevelParams(**log_levels),
                                    data_dir=data_dir,
                                    global_modules_dir=global_modules_dir,
                                    global_datasets_dir=global_datasets_dir)
        # static attributes
        self._data_dir = data_dir
        # logging is started first of all
        self._init_logging()
        # then modules and save dir
        self._init_saves_and_modules()
        # resume initially is false
        self._have_resumed = False
        # FIXME: NEW Remove this
        self._init_static_vars()
        self._init_property_vars()
        if self.config.trainer_params.resume or self.config.trainer_params.init_weights:
            self._init_device()
            self._init_models()
            self._check_resume_or_init_weights()

    def _init_saves_and_modules(self):
        self._savedir = os.path.join(self._data_dir, "savedir")
        self._modules_dir = os.path.join(self._data_dir, "modules")
        if not os.path.exists(self._savedir):
            os.mkdir(self._savedir)
        self._logi(f"Savedir is {os.path.abspath(self._savedir)}")

    def _init_logging(self):
        self._logdir = os.path.join(self._data_dir, "logs")
        if not os.path.exists(self._logdir):
            os.mkdir(self._logdir)
        self._logfile, self._logger = gen_file_and_stream_logger(
            self._logdir, "trainer", self.config.log_levels.file,
            self.config.log_levels.stream)
        log = Log(self._logger)
        self._logd = log._logd
        self._loge = log._loge
        self._logi = log._logi
        self._logw = log._logw
        self._logi(f"Initialized logger in {os.path.abspath(self._logdir)}")

    def init(self, force=False):
        """Initialize everything.

        If `force==True` then initialize even if we were initialized before.

        """
        if self._have_resumed and not force:
            self._logw("\"init\" cannot be called after resume. Use \"force\"")
            self._init_all()
        elif self._have_resumed and force:
            self._logw("forcing \"init\" call after resume")
            self._init_all()
        else:
            self._init_all()

    def _check_resume_or_init_weights(self):
        if self.trainer_params.init_weights:
            self._logw("Warning! Loading weights directly to model")
            load_state = torch.load(self.trainer_params.init_weights)
            for name in self._models.names:
                self._models.load_weights(name, load_state["models"][name])
            self._resume_path = None
        elif self.trainer_params.resume:
            pass
    # def _check_resume_or_init_weights(self):
    #     """Check for resume or initial weights and set :attr:`_resume_path`.

    #     :code:`config` is already loaded and will not be loaded again. Only
    #     state will be restored.

    #     """
    #     if self.trainer_params.init_weights:
    #         self._logw("Warning! Loading weights directly to model")
    #         load_state = torch.load(self.trainer_params.init_weights)
    #         for name in self._models.names:
    #             self._models.load_weights(name, load_state["models"][name])
    #         self._resume_path = None
    #     elif self.trainer_params.resume:  # implies resume from somewhere
    #         if self.trainer_params.resume_best:
    #             # try to find and resume best weights
    #             self._loge("Resume from best is not yet implemented")
    #             self._resume_path = None
    #         elif self.trainer_params.resume_dict:
    #             if os.path.exists(self.trainer_params.resume_dict):
    #                 self._resume_path = self.trainer_params.resume_dict
    #             else:
    #                 self._logw("Given resume weights do not exist")
    #                 self._resume_path = None  # set appropriate path
    #         else:
    #             # if both resume_best and resume_dict are false
    #             # AND resume is true AND checkpoint_path is valid
    #             if os.path.exists(self._checkpoint_path):
    #                 self._logi("Checkpoint exists. Will resume from there")
    #                 self._resume_path = self._checkpoint_path
    #             else:
    #                 self._logi("No checkpoint found. Will train from beginninng")
    #                 self._resume_path = None
    #     else:
    #         # Don't resume
    #         self._resume_path = None
    #     if self.trainer_params["resume"] and self._resume_path:
    #         self._logi(f"Resuming from {self._resume_path}")
    #         self._resume_from_path(self._resume_path)


    def _init_all(self):
        self._logi("Initializing trainer")
        self._init_device()
        self._init_models()
        self._init_data_and_dataloaders()
        self._init_update_funcs()
        self._init_training_steps()
        self._init_metrics()
        self._init_state_vars()
        self._init_task_runners()
        self._init_modules()
        self._init_hooks()
        self._dump_state()
        # self._init_extra_controls()

    # # NOTE: Shouldn't export this to Checks as name will be mangled
    # def _check_exports(self):
    #     """Checks the API as exported endpoints.

    #     All the properties not beginning with _ are exported except extras and
    #     methods.

    #     Controls and other export checks are to be added.

    #     :returns: None
    #     :rtype: None

    #     """
    #     missing = []
    #     for x in prop_names:
    #         if x not in prop.names:
    #             missing.append(x)
    #     additional = set(prop.names) - prop_names
    #     self._logw(f"Some properties not correctly exported {prop}, {missing}")
    #     self._logw(f"Additional properties {additional}")
    #     import ipdb; ipdb.set_trace()

    # START: Init Funcs
    # CHECK: It does beg the questions that what happens when the system
    #        reboots? Or when a reinitialization occurs.
    def _init_device(self):
        """Initialize the devices according to :attr:`trainer_params` and system
        configuration.

        If `gpus` are requested in :attr:`trainer_params` and the system
        contains `gpus` then they're allocated to the trainer. In addition, the
        devices to the models are assigned in accordance with
        :attr:`model_params`. Otherwise the models reside on the CPU (and system
        VRAM).

        Two (or more) models in this case can share the same device if there's
        enough room on the device VRAM.

        :meth:`reserve_gpus` and :attr:`reserved_gpus` are added to the
        :class:`Trainer` by :class:`~trainer.interfaces.FlaskInterface` as that manages the device
        reservation.

        """
        # CHECK: Why's this commented?
        # What if we're resuming and devices are already initialized?
        # if self._devices_initialized:
        #     return
        self._maybe_init_gpus()
        self._set_device()

    def _maybe_init_gpus(self):
        if self.gpus == []:
            self.gpus = [-1]
        if self.gpus != [-1]:
            available_gpus = [x for x in self.gpus if x not in self.reserved_gpus]
            unvailable_gpus = set(available_gpus) - set(self.gpus)
            self.gpus = available_gpus
            if not self.reserve_gpus(self.gpus)[0]:
                self._loge(f"Could not reserve gpus {self.gpus}")
                self.gpus = [-1]
            else:
                self._logd(f"Reserved gpus {self.gpus}")
            try:
                self._device_handles, removed = init_nvml(self.gpus)
                self._loge(f"Devices {unvailable_gpus} are already in use. " +
                           f"Will only set {available_gpus}")
                self._logd(f"Initialized devices {[*self._device_handles.keys()]} with names:\n" +
                           f"{[gpu_name(x) for x in self._device_handles.values()]}")
                if removed:
                    self._logw(f"Devices {removed} are not supported")
                self.gpus = [*self._device_handles.keys()]
            except Exception as e:
                self._loge(f"Could not initialize devices {self.gpus}. Error {e}")
                self._device_handles = None
        else:
            self._device_handles = None

    def _set_device(self):
        have_cuda = torch.cuda.is_available()
        gpus_given = self.gpus and (not self.gpus == [-1])
        cuda_given = self.trainer_params.cuda
        if not gpus_given:
            self._logd("No gpus given. Will run on cpu")
            # self._device = torch.device("cpu")
            self.gpus = [-1]
        elif cuda_given and not have_cuda:
            self._logw("cuda specified but not available. Will run on cpu")
            # self._device = torch.device("cpu")
            self.gpus = [-1]
        elif gpus_given and not cuda_given:
            self._logw("cuda not specified but gpus given. Will run on cpu")
            # self._device = torch.device("cpu")
            self.gpus = [-1]
        elif cuda_given and have_cuda and self.gpus != [-1] and len(self.gpus) == 1:
            self._logi(f"GPU {self.gpus[0]} detected and specified")
            # self._device = torch.device(f"cuda:{self._gpus[0]}")
        elif cuda_given and have_cuda and len(self.gpus) > 1:
            self._logi(f"Multipule gpus {self.gpus} specified." +
                       " Will allocate according to model_params")
        #     if torch.cuda.device_count() >= len(self._gpus):
        #         self._logi(f"{torch.cuda.device_count()} gpus are available")
        #         if "parallel" in self.trainer_params:
        #             # NOTE: I always get confused by this statement It's
        #             #       somewhhat mirthful and one has to see the next line
        #             #       to make sense of it.
        #             self._logi(f"Parallel call be functional {self.trainer_params['parallel']}")
        #             # self._device = self.trainer_params["parallel"]
        #         else:
        #             self._logi("Parallel call be Module dataparallel")
        #             # self._device = "dataparallel"
        #     else:
        #         self._loge(f"{torch.cuda.device_count()} gpus are not available")
        #         raise AttributeError
        # else:
        #     self._logi("cuda not specified. Using cpu")
        #     # self._device = torch.device("cpu")
        #     self._gpus = [-1]
        torch.cuda.manual_seed(self.trainer_params.seed)
        # for t, v in self.trainer_params.items():
        #     if t in self.__class__.__dict__:
        #         self._logw(f"Tried overwriting attribute {t}! Denied.")
        #     elif t != "gpus":
        #         self.__dict__[t] = v
        self._devices = {}
        self._devices_initialized = True

    # FIXME: NEW Replace with pydantic
    def _init_static_vars(self):
        self.adhoc_error_dict = {
            "required_oneof_[function]": ["train", "val", "test", "user_func_name"],
            "required_for_[function]": {"epoch": "[int|string]_which_epoch",
                                        "data": "[string]_train_val_or_test",
                                        "num_or_fraction":
                                        "[int|float]_number_of_points_or_fraction_of_dataset",
                                        "callback": "[string]_name_of_callback_function"}}

    def _init_state_vars(self):
        """Initialize default state variables.

        :attr:`epoch` always remains 0 if training only with iterations and
        :attr:`iterations` increase.

        post_epoch_hooks are run after a specified number of iterations which is
        :attr:`_hooks_run_iter_frequency`

        """
        # params and state properties

        self._logi("Initializing State Variables")

        # NOTE: These should only be modified by the transition function
        self._running_event = Event()  # something is running
        self._current_aborted_event = Event()  # current task was aborted
        self._session_aborted_event = Event()  # entire session was aborted

        self._adhoc_running_event = Event()  # adhoc task is running
        self._adhoc_aborted_event = Event()  # adhoc task was aborted
        self._userfunc_running_event = Event()  # userfunc is running
        self._userfunc_aborted_event = Event()  # userfunc was aborted

        self._threads = {"main": Thread(target=self.train)}

        # NOTE: Default setting for tasks
        self._aborted = []  # prev_loop_aborted

        # Initialize hooks. Validate test and save can never be removed, LOL
        #
        # TODO: Not sure if hooks are state or property vars
        #       Maybe they're just hooks
        # TODO: There should be a better way as I should be able to disable
        #       validate and test That can only be if I specify order in certain
        #       hooks.
        self._post_epoch_hooks_to_run = Hook(["validate", "test", "update_metrics"])

        if not getattr(self, "_hooks_run_iter_frequency", None):
            self._hooks_run_iter_frequency = self.trainer_params.test_frequency
        # FIXME: validate, test, update_metrics is mandatory for now,
        #        unless val_loader and test_loader are none of course
        self._post_epoch_hooks_to_run.append("save_history")
        self._post_epoch_hooks_to_run.append("save_best")
        self._post_epoch_hooks_to_run.append("save_checkpoint")
        self._post_epoch_hooks_to_run.append("log")

        # NOTE: _log_metrics is a function so "metrics" defines a way to log it
        #       rather than just copying the values.
        # self._items_to_log_dict = {"metrics": self._log_metrics}
        # CHECK: Why am I initializing device again?
        # self._init_device()
        self._epoch = 0
        self._iterations = 0
        # self._init_nvml()
        steps = self.trainer_params.training_steps
        # NOTE: In theory user func can be a task, but not right now
        if "iterations" in steps:
            self._transition_steps = {"main": {"train", "val", "test", "none"},
                                      "adhoc": {"val", "test", "extra", "none"},
                                      "user": {"func", "none"}}
        else:
            self._transition_steps = {"main": set(steps).union({"none"}),
                                      "adhoc": {"val", "test", "none"},
                                      "user": {"func", "none"}}
        self._loop_states = {"main": {"paused", "running", "finished"},
                             "adhoc": {"paused", "running", "finished"},
                             "user": {"running", "finished"}}
        self._task_flags = []
        self._task_args = {"main": [],
                           "adhoc": [],
                           "user": []}
        self._task_kargs = {"main": [],
                            "adhoc": [],
                            "user": []}
        self._current_state = "main_paused_none"
        self._prev_paused_state = None

    # TODO: For each such variable i.e., static, property etc. add a decorator
    #       or a function such that they're added to that list e.g.,
    #       property_vars and are initialized correctly or raise error if not
    #       initialized so that while adding adhoc vars in the middle of the
    #       code, I don't forget to initialize them somewhere
    #
    #       In the middle of file:
    #
    #       >>> self.add_to_property_vars(meh)
    #
    #       In _init_property_vars:
    #
    #       >>> assert all(getattr(x) for x in self._property_vars)
    #
    #       Or something like that.
    def _init_property_vars(self):
        self._logi("Initializing Property Variables")
        self.extra_report = {}
        self._user_funcs = {}
        self._current_user_func_name = None
        self._current_user_func_params = None
        # self.load_weights.__dict__["content_type"] = "form"
        # self.add_model.__dict__["content_type"] = "form"
        # self.add_user_funcs.__dict__["content_type"] = "form"
        # self.load_image.__dict__["content_type"] = "form"

    def _init_modules(self):
        """Instantiate :class:`Modules`.

        Modules serves as a single entry point for loading and unloading dynamic
        modules.

        """
        self._modules = Modules(self._modules_dir,
                                {"logd": self._logd, "loge": self._loge,
                                 "logi": self._logi, "logw": self._logw})
        # NOTE: Data is always available
        #       Other things can be nvml, device and stuff.
        self._user_func_env_params = {"train_loader": self.train_loader,
                                      "val_loader": self.val_loader,
                                      "test_loader": self.test_loader,
                                      "torch": torch}

    def _init_models(self):
        """Initialize models with a :class:`~trainer.model.Model` class

        Criteria are are initialized also.  :attr:`Trainer.model_params` should
        be a :class:`dict` of {model_name: :class:`config.ModelParams`}.  Device
        allocation can be specified with giving `gpus` in model_params. There
        can be overlap between models and GPUs and it is left up to the user.

        `gpus` can also be specified as "auto" or "parallel".  If "auto" is
        given then certain heuristics are applied according which of the models
        are to be loaded initially the the gpus available.

        Devices are allocated for the models in alphabetical order except `auto`
        and `parallel` which are allocated last. So if there's a conflict
        between devices priority is given alphabetically.

        models which are to be loaded should have {"loaded": True} in their
        params. Alternatively one can specify {"load_all": True} in
        :attr:`trainer_params`

        See :ref:`source/dorc:Device Allocation` for details

        """
        self._logi("Initializing Criteria and Models.")
        self.criteria = {}
        for k, v in self.criteria_params.items():
            self.criteria[k] = v.function(**v.params)
        # TODO: Model parallel and sharding
        # NOTE: if only one model load it
        self._models = {}
        if self.trainer_params.load_all:
            load_models = self.model_params.keys()
        else:
            load_models = [k for k, v in self.model_params.items() if v.load]
        if self.gpus and self.gpus != [-1]:
            self.allocate_devices(load_models)
        else:
            for model_name in load_models:
                self.devices[model_name] = []
        for model_name in load_models:
            self._models[model_name] = self._model_init_helper(model_name)
            if self._models[model_name]:
                status, response = self._models[model_name].load_into_memory()
                if not status:
                    self._loge(response)
            else:
                self._models[model_name] = None

    def _init_hooks(self):
        """Initialize :attr:`hooks` and :attr:`hooks_with_args`.

        All initial hooks are loaded from :mod:`hooks_module` but can be added
        by the user.

        """
        self._hooks = {}
        self._hooks_with_args = {}
        for x in hooks_module.__dict__:
            if x.endswith("_hook"):
                self._hooks[x] = hooks_module.__dict__[x]
            if x.endswith("_hook_with_args"):
                self._hooks_with_args[x] = hooks_module.__dict__[x]

    def load_models_state(self, model_state: Dict) -> Return:
        """Load state of all :attr:`loaded_models` from :code:`model_state`.

        Args:
            model_state: A dictionary of model names and state
                         See :meth:`model.Model.load` for the components of state.

        Return:
            Status and message
        """
        for model in self.loaded_models:
            status, message = self._models[model].load(model_state[model])
            if not status:
                self._loge(f"Loading model {model} failed. Error {message}")
                return make_return(False, message)
        return make_return(True, "")

    def allocate_devices(self, load_models: Union[str, List[str]] = [],
                         unload_models: Union[str, List[str]] = []):
        """Spread the devices over the `load_models` models.

        First remove the devices from :attr:`devices` and then allocate
        according to parameters.

        Args:
            load_models: names and params of models which should be loaded
            into system memory

        For allocation, explicitly mentioned gpus are given preference
        first. After that `auto` is given preference over `parallel`.

        """
        if isinstance(load_models, str):
            load_models = [load_models]
        if isinstance(unload_models, str):
            unload_models = [unload_models]
        load_models = [*load_models]
        unload_models = [*unload_models]
        for model_name in unload_models:
            if model_name in self.model_params:
                self.devices[model_name] = []
            else:
                self._logw(f"{model_name} not in known models")
        if len(load_models) == 1:
            model_name = load_models[0]
            if model_name in self.model_params:
                gpus = self.model_params[model_name].gpus
                remaining = set(self.gpus) - set(self.allocated_devices)
                if gpus == "auto" or gpus == "parallel":
                    self.devices[model_name] = [x for x in remaining]
                else:
                    self.devices[model_name] = [x for x in gpus if x in remaining]
            else:
                self._logw(f"{model_name} not in known models")
        else:
            auto = []
            parallel = []
            for model_name in load_models:
                if model_name in self.model_params:
                    if self.model_params[model_name].gpus == "auto":
                        auto.append(model_name)
                    elif self.model_params[model_name].gpus == "parallel":
                        parallel.append(model_name)
                    else:
                        self.devices[model_name] = [x for x in self.model_params[model_name].gpus
                                                    if x in self.gpus and
                                                    x not in self.allocated_devices]
                else:
                    self._logw(f"{model_name} not in known models")
            remaining = set(self.gpus) - set(self.allocated_devices)
            if len(remaining) > 0:
                # FIXME: NEW
                ranking = [*gpu_ranking(self._device_handles).items()]
                num_params = {x: 0 for x in auto}
                for i, model_name in enumerate(auto):
                    self.devices[model_name] = []
                    temp_model = self._model_init_helper(model_name)
                    if temp_model is not None:
                        temp_model.load_into_memory()
                    else:
                        raise ValueError()
                    num_params[model_name] = np.sum([np.prod(x.shape)
                                                     for x in temp_model.weights.values()])
                    del temp_model
                num_params = [*num_params.items()]
                num_params.sort(key=lambda x: x[1], reverse=True)
                ranking = sorted(ranking, key=lambda x: (x[1]["memory"], x[1]["compute"]),
                                 reverse=True)
                # NOTE: `auto` takes precedence over `parallel``
                if len(auto) < len(remaining):
                    # FIXME: need some to be parallel and some over a single gpu
                    raise NotImplementedError
                elif len(auto) >= len(remaining):
                    # NOTE: put the bigger model on to the GPU with more ram
                    for i, (model_name, _) in enumerate(num_params):
                        if i < len(ranking):
                            self.devices[model_name] = [ranking[i][0]]
                        else:
                            self.devices[model_name] = []
                remaining = set(self.gpus) - set(self.allocated_devices)
                if len(parallel) >= len(remaining):
                    # parallel_devices = self.best_parallel_devices(parallel)
                    for model_name in parallel:
                        if model_name in self.model_params:
                            # self.devices[name] = parallel_devices[name]
                            pass
                        else:
                            self._logw(f"{model_name} not in known models")
            else:
                for model_name in auto:
                    if model_name in self.model_params:
                        self.devices[model_name] = []
                    else:
                        self._logw(f"{model_name} not in known models")
                for model_name in parallel:
                    if model_name in self.model_params:
                        self.devices[model_name] = []
                    else:
                        self._logw(f"{model_name} not in known models")

    def _model_init_helper(self, model_name) -> Optional[Model]:
        model = self.model_params[model_name].model
        model_params = self.model_params[model_name].params
        try:
            optim_name: str = self.model_params[model_name].optimizer
            if optim_name in self.optimizers:
                optim_func: Callable = self.optimizers[optim_name].function
                optim_params: Dict = self.optimizers[optim_name].params
            # FIXME: how give optim params if optimizer not in self.optimizers?
            elif hasattr(torch.optim, optim_name):
                optim_func = getattr(torch.optim, optim_name)
                optim_params = {}
            else:
                optim_params = {}
            optimizer = {"name": optim_name,
                         "function": optim_func,
                         "params": optim_params}
            gpus = self.devices[model_name]
            return Model(model_name, model, model_params, optimizer, gpus)
        except Exception as e:
            self._loge(f"Error occured {e}")
            return None

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def loaded_models(self) -> List[str]:
        return [name for name, model in self._models.items() if model.loaded]

    @POST
    @methods
    def load_model(self, name: str) -> Return:
        """Load a model into system and device memory with given `name`."""
        if name in self.model_params:
            return make_return(*self._models[name].load_into_memory())
        else:
            return make_return(False, f"Model {name} not present")

    @POST
    @methods
    def unload_model(self, name: str) -> Return:
        """Unload a model from system with given `name`.

        Frees up resources allocated to the model."""
        if name in self.model_params:
            return make_return(*self._models[name].unload())
        else:
            return make_return(False, f"Model {name} not present")

    def _init_data_helper(self, params: Union[Dict, Iterable]) -> Iterable:
        if isinstance(params, dict):
            if "function" in params and "params" in params:
                return params["function"](**params["params"])
            else:
                raise AttributeError(f"Unknown or bad parameters {params}")
        elif iter(params) and len(params) and not isinstance(params, str):  # type: ignore
            return params
        else:
            raise AttributeError(f"Unknown or bad parameters {params}")

    def _init_data_and_dataloaders(self):
        """Initialize the dataloaders.

        Dataloaders are initialized from a `{step, data, params}` in which the
        corresponding dataloaders are initialized with the given data.

        `step` here corresponds to the `train\\val\\test` step and so are
        `data` and `params` respectively. As such separate datasets can
        always be initialized with different paramters at any given instance.

        They can also be initialized from a function like `get_dataloader`
        with certain parameters, so that a complex dataloader with various
        transformations and arrangements on the data can be loaded `ab initio`.

        In case `train data` is available, then `train_step` has to be
        available. Arbitrary custom named steps aren't supported as of now.

        As of now, only a torch :class:`~torch.utils.data.DataLoader` is
        integrated, but more may be added as the data loading and feeding part
        is agnostic of the framework.

        Data can be given as either:

            a. `data["train"]` etc. in the parameters
            b. Or for a custom dataloader can be given as a function which
               provides the dataloader along with params.

        Data and dataloader can be skipped entirely (aside from `train`) if both
        `data[step]` and `dataloader_params[step]` are `None`.

        If however e.g., `data["val"]` is `None` and
        `dataloader_params["val"]` is not `None`, it implies that "val" data
        was not to be skipped. Then the second case is checked and a "function"
        has to be present which will provide the dataloader.

        """
        self._logi("Initializing Dataloaders")

        def _check_raw(loader, name):
            if loader and not hasattr(loader, "dataset"):
                self._logw(name + " loader doesn't have a dataset")
            elif loader and not hasattr(loader.dataset, "_get_raw"):
                self._logw(name + " dataset doesn't define \"_get_raw\"" +
                           " Drawing samples from validation data will not be available.")
            elif loader and hasattr(loader.dataset, "_get_raw"):
                self._logw(name + " dataset has \"_get_raw\"" +
                           " Drawing samples from validation data is available!")

        if self.data_params.train:
            train_data = self._init_data_helper(self.data_params.train)
            self.train_loader = DataLoader(train_data,
                                           **self.dataloader_params.train.__dict__)
            _check_raw(self.train_loader, "Train")
            if self.data_params.val:
                val_data = self._init_data_helper(self.data_params.val)
                self.val_loader = DataLoader(val_data,
                                             **self.dataloader_params.val.__dict__)
                _check_raw(self.train_loader, "Val")
            else:
                self._logi("No Val loader. Will not do validation")
                self.val_loader = None
            if self.data_params.test:
                test_data = self._init_data_helper(self.data_params.test)
                self.test_loader = DataLoader(test_data,
                                              **self.dataloader_params.test.__dict__)
                _check_raw(self.test_loader, "Test")
            else:
                self._logi("No Test loader. Will not do testing")
                self.test_loader = None
        else:
            self.train_loader = self.data_params.loader.train(
                **self.data_params.loader.train_params)
            _check_raw(self.train_loader, "Train")
            if self.data_params.loader.val:
                self.val_loader = self.data_params.loader.val(
                    **self.data_params.loader.val_params)
                _check_raw(self.train_loader, "Val")
            else:
                self._logi("No Val loader. Will not do validation")
                self.val_loader = None
            if self.data_params.loader.test:
                self.test_loader = self.data_params.loader.test(
                    **self.data_params.loader.test_params)
                _check_raw(self.test_loader, "Test")
            else:
                self._logi("No Test loader. Will not do testing")
                self.test_loader = None

    def _restore_dataloader_params(self, params):
        self._logd("Restoring dataloader_params")
        if any([(y is not None and "collate_fn" in y or callable(y))
                for x, y in params.items()]):
            self._logw("collate_fn will not be restored")
        for k, v in params.items():
            if v is not None and set(v.keys()) != set(self.dataloader_params[k].keys()):
                self._logw(f"Dataloader params for {k} differ. Not restoring")
            elif v is not None:
                for a, b in v.items():
                    if a != "collate_fn":
                        value = params[k][a]
                        if isinstance(value, dict):
                            for x, y in value.items():
                                if isinstance(y, str) and not y.startswith("callable_"):
                                    self.dataloader_params[k][a][x] = y
                        else:
                            if isinstance(value, str) and not value.startswith("callable_"):
                                self.dataloader_params[k][a] = value

    def _init_update_funcs(self):
        self._logi("Initializing Update Functions")
        self._model_step_func = None
        if self.update_functions.train is not None:
            for x in self.trainer_params.training_steps:
                func = self.update_functions.__dict__[x]
                if func is not None and not func.criteria:
                    criteria = {m: self.criteria[c] for m, c in func.criteria_map.items()}
                    self.update_functions.__dict__[x].set_criteria(criteria)
        else:
            func = self.update_functions.function
            params = self.update_functions.params.__dict__.copy()
            # models and criteria already exist with the trainer.
            # They are to be updated according to maps in the func.
            models = {m: self._models[m] for m in params["models"]}
            criteria = {m: self.criteria[c] for m, c in params["criteria_map"].items()}
            params["models"] = models
            params["criteria"] = criteria
            self._model_step_func = func(**params)

    def _init_training_steps(self):
        """Which training steps will be run.

        Depends on a combination of `training_steps`, `dataloaders` and
        `update_functions`.

        """
        self._training_steps = {}
        if self.trainer_params.training_type == "iterations":
            loaders = ["train", "val", "test"]
        else:
            loaders = self.trainer_params.training_steps
        for x in loaders:
            if getattr(self, x + "_loader", None):
                if self._model_step_func:
                    self._training_steps[x] = (self._model_step_func,
                                               self._model_step_func.returns(x),
                                               self._model_step_func.logs(x))
                else:
                    func = getattr(self.update_functions, x)
                    self._training_steps[x] = func, func.returns, func.logs

    def _init_metrics(self):
        """Intializes and checks the metrics.

        Anything returned by the `step` having the first element `metric` is a
        default metric and is logged.

        Other metrics are specified in `extra_metrics` have to conform to the
        format of type :class:`~config.Metric`.

        """
        self._logi("Initializing Metric")
        self._metrics = {}
        for step, (func, retvals, logs) in self._training_steps.items():  # step = training step
            self._metrics[step] = dict((name, {}) for name in retvals       # name = name of metric
                                       if name in logs)
            self._metrics[step]["num_datapoints"] = {}
        for name, metric in self.extra_metrics.items():
            for step in metric.steps:
                if step not in self._metrics:
                    self._metrics[step] = {}
                self._metrics[step][name] = {}
                func_retvals = ((self._model_step_func and self._model_step_func.returns(step))
                                or self.update_functions.__dict__[step].returns)
                if metric.when == "BATCH":
                    if not all(_ in func_retvals for _ in metric.inputs):
                        raise ValueError(
                            self._loge(f"Not all metric inputs {metric.inputs} for {name} " +
                                       f"and functions return values {retvals}"))
                elif metric.when == "EPOCH":
                    # CHECK: Is this good practice?
                    #        I meant
                    # NOTE: all attrs in self.props + all attrs in func
                    # FIXME: Change to all named variables (props)
                    retvals = [*self.__dict__.keys(), *prop.members, *func_retvals]
                    if not all(_ in retvals for _ in metric.inputs):
                        raise ValueError(
                            self._loge(f"Not all metric inputs {metric.inputs} for {name} " +
                                       f"and functions return values {retvals}"))

    def _init_task_runners(self):
        """Initialize the :attr:`task_runners`

        A `task_runner` is an :class:`Epoch` instance. This method initializes
        three kinds of task runners:

          1. epoch_runner
          2. adhoc_runner
          3. user_func_runner

        Corresponding to each task runner there are threads with the same names
        and the references are further stored in `_task_thread_keys`,
        `_task_callbacks`.

        """
        device_monitor, signals = self._task_runner_helper("main")
        self._logi("Initializing Epoch Runner")
        self._epoch_runner = Epoch({"metrics": self._metrics, "extra_metrics": self.extra_metrics},
                                   signals, device_monitor, self.extra_report,
                                   **{"logd": self._logd, "loge": self._loge,
                                      "logi": self._logi, "logw": self._logw})
        self._epoch_runner.name = "epoch_runner"
        self._task_runners = {"main": self._epoch_runner,
                              "adhoc": None,
                              "user": None}
        self._task_thread_keys = {"main": "main",
                                  "adhoc": "adhoc",
                                  "user": "user"}
        # FIXME: For val and test maybe update in separate variables
        self._task_callbacks = {"main": self._run_post_epoch_hooks,
                                "adhoc": None,
                                "user": None}

    def _init_default_task_runner(self, device_monitor: DeviceMonitor,
                                  signals: Signals) -> Epoch:
        task_runner = Epoch({"metrics": self._metrics, "extra_metrics": self.extra_metrics},
                            signals, device_monitor, self.extra_report,
                            **{"logd": self._logd, "loge": self._loge,
                               "logi": self._logi, "logw": self._logw})
        return task_runner

    def _task_runner_helper(self, which: str) -> Tuple[DeviceMonitor, Signals]:
        device_monitor = DeviceMonitor(self._device_handles)
        # signals = SimpleNamespace()
        if which == "main":
            signals = Signals(self._running_event, self._current_aborted_event)
            # signals.paused = self._running_event
            # signals.aborted = lambda: self._current_aborted_event.is_set()
        elif which == "adhoc":
            signals = Signals(self._adhoc_running_event, self._adhoc_aborted_event)
            # signals.paused = self._adhoc_running_event
            # signals.aborted = lambda: self._adhoc_aborted_event.is_set()
        elif which == "user":
            signals = Signals(self._usefunc_running_event, self._userfunc_aborted_event)
            # signals.paused = self._userfunc_running_event
            # signals.aborted = lambda: self._userfunc_aborted_event.is_set()
        return device_monitor, signals

    def _task_runner_initialize(self, name: str, metrics: Dict[str, Dict],
                                extra_metrics: Dict[str, Dict],
                                callback: Callable) -> Tuple[bool, Optional[str]]:
        """Initializes a task_runner. The quintessential task runner is the epoch
        runner. Task runner coordinates with the trainer to ensure that nothing
        gets jammed up.

        Task runner needs signals to communicate with the trainer and those are
        created by the helper as needed and recreated when another task_runner is
        initialized.

        :param name: Name :class:`str` of the task runner.
        :param metrics: :class:`dict` of metrics to be given to the task runner
        :param extra_metrics: :class:`dict` of extra metrics to be given to the task runner
        :param callback: :class:`function` callback to be called when the execution finishes
        :returns: None
        :rtype: None

        """
        if name not in self._task_runners:
            return False, self._loge(f"Unknown task runner {name}")
        else:
            device_monitor, signals = self._task_runner_helper(name)
            kwargs = {"logd": self._logd, "loge": self._loge,
                      "logi": self._logi, "logw": self._logw}
            self._task_runners[name] = Epoch({"metrics": metrics,
                                              "extra_metrics": extra_metrics},
                                             signals, device_monitor, self.extra_report,
                                             **kwargs)
            self._task_runners[name].reset()
            self._task_runners[name].logger = self.logger
            self._task_callbacks[name] = callback
            return True, None
    # END: Init Funcs

    # START: Internal Controls
    #        These functions interact with the SM.
    #        NOTE: As of now this is only the main loop
    def _finish_if_paused_or_running(self, gather=False):
        """Should not be called from `self._transition`
        Stops the current running main flow (or alternate flow?)

        :returns: None
        :rtype: None

        """
        self._transition_flags = {}
        if gather:
            self._transition_flags["run_cb"] = True
        loop, run, step = self.current_state.split("_")
        # if run == "running":
        #     status, message = self._transition(self.current_state, "_".join([loop, "paused", step]))
        # print("DONE with paused", self.current_state)
        status, message = self._transition(self.current_state, "_".join([loop, "finished", step]))

    def _pause_and_resume_main(self):
        """Should not be called from `self._transition`
        Stops the current running main flow (or alternate flow?)

        :returns: None
        :rtype: None

        """
        loop, run, step = self.current_state.split("_")
        # if run == "running":
        #     status, message = self._transition(self.current_state, "_".join([loop, "paused", step]))
        # print("done with paused", self.current_state)
        status, message = self._transition(self.current_state, "_".join([loop, "pause", step]))
        status, message = self._transition(self.current_state, self._prev_paused_state)

    def _finish_and_resume_main(self, gather=False):
        """Should not be called from `self._transition`
        Stops the current running main flow (or alternate flow?)

        :returns: None
        :rtype: None

        """
        self._transition_flags = {}
        if gather:
            self._transition_flags["run_cb"] = True
        loop, run, step = self.current_state.split("_")
        # if run == "running":
        #     status, message = self._transition(self.current_state, "_".join([loop, "paused", step]))
        # print("done with paused", self.current_state)
        if loop in {"adhoc", "user"}:
            status, message = self._transition(self.current_state, "_".join([loop, "finished", step]))
            status, message = self._transition(self.current_state, self._prev_paused_state)
        else:
            status, message = self._transition(self.current_state, "_".join([loop, "finished", step]))

    def _pause_if_running(self):
        """Should not be called from `self._transition`

        :returns: None
        :rtype: None

        """
        loop, run, step = self.current_state.split("_")
        if run == "running":
            status, message = self._transition(self.current_state, "_".join([loop, "paused", step]))

    def _start_if_not_running(self):
        """Should not be called from `self._transition`

        :returns: None
        :rtype: None

        """
        loop, run, step = self.current_state.split("_")
        if step != "none":
            return              # only if step == "none"
        else:
            step = "train"      # default
        if run == "paused":
            status, message = self._transition(self.current_state,
                                               "_".join([loop, "running", step]))

    def _run_if_paused(self):
        """Should not be called from `self._transition`

        :returns: None
        :rtype: None

        """
        loop, run, step = self.current_state.split("_")
        if loop == "main":
            self._none_to_train()
        loop, run, step = self.current_state.split("_")
        print("NOW WILL RUN")
        if run == "paused":
            status, message = self._transition(self.current_state,
                                               "_".join([loop, "running", step]))

    def _run_new_if_finished(self):
        print("RUNNING NEW aborted flag", self._current_aborted)
        loop, run, step = self.current_state.split("_")
        if loop == "main":
            self._none_to_train()
        loop, run, step = self.current_state.split("_")
        status, message = self._transition(self.current_state, "_".join([loop, "running", step]))
        # if run == "finished":
        #     if _force:
        #         self._transition(self.current_state, "_".join(["force", "running", step]))
        #     else:
        #         self._transition(self.current_state, "_".join(["normal", "running", step]))

    # FIXME: This doesn't do anything
    def _abort_on_error(self):
        self._abort_current
    # END: Internal Controls

    # START: State Machine Helpers
    #        Helper functions to enforce state machine commands
    def _ensure_thread(self, loop, target, args=None, kwargs=None):
        print("CALLING ensure_thread for", loop, target)
        if loop not in self._threads or not self._threads[loop].is_alive():
            self._logd(f"Creating thread for {loop}")
            if args is not None and kwargs is not None:
                self._threads[loop] = Thread(target=target, args=args, kwargs=kwargs)
            elif args is None and kwargs is not None:
                self._threads[loop] = Thread(target=target, kwargs=kwargs)
            elif args is not None and kwargs is None:
                self._threads[loop] = Thread(target=target, args=args)
            else:
                self._threads[loop] = Thread(target=target)
            self._threads[loop].start()

    def _ensure_paused(self, loop, step):
        "Should not be called from anywhere but `self._transition`"
        self._logd(f"Calling ensure paused with {loop}")
        runner = self._task_runners.get(loop)
        if loop == "main":
            if not self.paused:
                self._running_event.clear()  # not running
        elif loop == "adhoc":
            if not self.adhoc_paused:
                self._adhoc_running_event.clear()
        elif loop == "user":
            if hasattr(runner, "paused") and not self.userfunc_paused:
                self._userfunc_running_event.clear()
            else:
                self._logd(f"User function cannot be paused")
        # print("LOOP ENTER WHILE running waiting", runner.train_loop.running, runner.train_loop.waiting)
        if runner is not None and runner.running:
            j = 0
            while runner.running and not runner.waiting and j < 3:
                time.sleep(1)
                # print(self.current_state)
                # print(loop, self._running_event.is_set())
                # print("LOOP running waiting", runner.train_loop.running, runner.train_loop.waiting)
                self._logd(f"Waiting for {loop} runner\n" +
                           f"running, waiting, finished " +
                           f"{runner.running}, {runner.waiting}, {runner.finished}")
                j += 1
            # while runner.waiting:
            #     time.sleep(1)
            #     self._logd(f"{loop} runner should not be waiting")

    def _ensure_unpaused(self, loop):
        "Should not be called from anywhere but `self._transition`"
        self._logd(f"Loop is {loop}")
        runner = self._task_runners.get(loop)  # say epoch
        if loop == "main":
            if self.paused:
                self._running_event.set()  # not running
        elif loop == "adhoc":
            if self.adhoc_paused:
                self._adhoc_running_event.set()
        elif loop == "user":
            if hasattr(runner, "paused") and self.userfunc_paused:
                self._userfunc_running_event.set()
            else:
                self._logd(f"User function cannot be unpaused")
        if runner is not None and runner.waiting:
            self._logd(f"{loop} runner is waiting. Should not be waiting")

    def _ensure_ready(self, loop, step):
        "Only toggles runner's waiting. Should not be called from anywhere but `self._transition`"
        runner = self._task_runners.get(loop)  # say epoch
        print("calling ensure ready")
        if loop == "main":
            if self.paused:
                self._logd(f"No need to ensure ready while paused for {loop}")
        elif loop == "adhoc":
            if self.adhoc_paused:
                self._logd(f"No need to ensure ready while paused for {loop}")
        elif loop == "user":
            if hasattr(runner, "paused") and self.userfunc_paused:
                self._logd(f"No need to ensure ready while paused for {loop}")
        if runner is not None and runner.waiting:
            # runner.toggle_waiting()
            print("NOT GOING to toggle waiting")

    # CHECK: If finished or reset is called, then what should happen to the
    #        thread?  Does the epoch runner die?
    #
    #        Actually, since the epoch_runner etc. are reset after each epoch
    #        anyway, so it doesn't really matter. They can be killed and
    #        respawned if they get stuck.
    def _ensure_finished(self, loop, step, timeout=5, run_cb=False, join_thread=True):
        """`self._ensure_paused` should be called before calling _ensure_finished
        :meth:`Epoch.finish` doesn't actually reset the metrics and variables
        stored in the batch.

        CHECK: Not sure about this:
        The function in itself is wrapped in a thread and the it's the
        responsibility of the function to listen to the aborted event.

        Should not be called from anywhere but `self._transition`

        :returns: None
        :rtype: None

        """
        # This is like a guard
        self._logd(f"Calling _ensure_finished for {loop}, {step}")
        runner = self._task_runners.get(loop)
        print("RUNNER current aborted", self._current_aborted,
              "runner_aborted", runner.aborted.is_set())
        # NOTE: if self._current_aborted and not runner.aborted.is_set()
        #       it means that we've aborted but waiting for the runner to abort
        callback = self._task_callbacks.get(loop)
        if not self._current_aborted:
            self._toggle_current_aborted()
            print("TOGGLED current_aborted", runner.aborted.is_set())
        if runner is not None:
            if self.paused:
                print("WAS paused, aborted", self.paused, runner.aborted.is_set())
                self._toggle_running()  # unpause for abort
                runner.aborted.wait()
                print("NOW paused, aborted", self.paused, runner.aborted.is_set())
                self._toggle_running()
                print("NOW paused, aborted", self.paused, runner.aborted.is_set())
            else:
                runner.aborted.wait()
            if callback is not None and run_cb:
                self._logd(f"calling callback {callback} for {loop}, {step}")
                callback()
            else:
                self._logd(f"not calling callback {callback} for {loop}, {step}")
        if self._current_aborted:
            # NOTE: Reset current_aborted_event
            print("Resetting current_aborted")
            self._toggle_current_aborted()
        if join_thread and self._threads[loop].is_alive():
            # FIXME: join is called while callback is running, LOL
            # NOTE: callbacks aren't there right now
            self._threads[loop].join(2)
            if self._threads[loop].is_alive():
                self._loge(f"Could not kill task {loop}")
        else:
            self._logd(f"Finished task {loop}")

    # def _check_paused_or_finished(self, loop, step):
    #     if self._task_runners[loop].finished:
    #         self._current_state = "_".join([loop, "finished", step])
    #     # FIXME: this is checking if step can be paused, that should be a property of the step
    #     elif not self._task_runners[loop].finished and step in {"func", "extra"}:
    #         self._ensure_finished(loop, step, join_thread=False)
    #         self._current_state = "_".join([loop, "finished", step])
    #     elif self._task_runners[loop].waiting:
    #         self._current_state = "_".join([loop, "paused", step])
    #     elif not self._task_runners[loop].waiting:
    #         self._ensure_paused(loop, step)
    #         self._current_state = "_".join([loop, "paused", step])

    # def _ensure_loop_transition(self, state_a, state_b):
    #     loop_a, _, step_a = state_a.split("_")
    #     self._check_paused_or_finished(loop_a, step_a)
    #     self._transition(self.current_state, state_b)

    def _none_to_train(self):
        loop, split, step = self.current_state.split("_")
        if step == "none":
            self._transition(self.current_state, "_".join([loop, split, "train"]))
    # END: State Machine Helpers

    # START: State Machine
    def _transition(self, _from: str, _to: str) -> None:
        """Transition to the next state from the given state.

        Args:
            _from: previous state
            _to: next state

        State is a string triple joined by "_". Each state `s` is composed of
        `loop_run_step`, where:

        - `loop` can be in {"main", "adhoc", "user"}
        - `run` can be in {"running", "paused", "finished"}
        - `step` for the main loop can be any of the
          `self.trainer_params.training_steps` + "none"
          or `{"train", "val", "test", "none"}`

        For a :class:`Trainer` instance the regular flow is `train_step ->
        post_epoch_hooks` or equivalent steps. However, all those can be paused
        to run auxiliary tasks to check how trainer is doing.

        `run` is the state of the trainer in the sense that if some active task
        is going on which has reserved some of the resourcese, whether that be
        training, validation or a given user function.

        - "running" implies that some such task is present.
        - "paused" implies that task is present but not active
        - "finished" means that the task has finished

        On top of these `self._aborted` is not a :class:`list` of flags which is
        to be set after a task was finished but was aborted by the user. See
        `abort` for details.

        The valid states is managed by a separate module :class:`StateMachine`
        and the progress of "training", "validation" etc. are kept track of by
        the trainer itself.

        In general multiple loops can run in parallel if required but currently
        only one loop is run at a given time and the other one is paused. A
        function running in a loop must be completed or aborted for another
        function to start in the same loop.

        """
        a_loop, a_run, a_step = _from.split("_")
        b_loop, b_run, b_step = _to.split("_")

        def legal_state(a, b, c):
            return (a in {"main", "adhoc", "user"} and b in self._loop_states[a]
                    and c in self._transition_steps[a])

        if _from != self.current_state:
            return False, self._loge(f"from state != current state: " +
                                     f"{_from} != {self.current_state}")
        if not legal_state(a_loop, a_run, a_step):
            return False, self._loge(f"Illegal states {_from}")
        if not legal_state(b_loop, b_run, b_step):
            return False, self._loge(f"Illegal states {_to}")
        self._logd(f"Trying to transition from {_from} to {_to}")

        def to_none():
            return (b_step in {"none"})

        def main_none():
            return (a_loop == "main" == b_loop
                    and a_step == "none"
                    and b_step in {"none", "finished"})

        # NOTE: step in {"train", "val", "test"}
        def main_same_step():
            return (a_loop == "main" == b_loop
                    and a_step == b_step != "none"
                    # NOTE: in the same step but can't be none -> none
                    and ((a_run != b_run and a_run in {"running", "paused"}
                          and b_run in {"running", "paused"})  # running <-> paused
                         # NOTE: only paused -> finished for main
                         # NOTE: changed as paused is different now [2020-03-12 Thu 16:57]
                         or (a_run in {"running", "paused"} and b_run == "finished")))

        def main_different_step():
            return (a_loop == "main" == b_loop
                    and b_step != "none"
                    and ((a_step != b_step
                          and a_run == "finished"  # finished -> {running, paused}
                          and b_run in {"running", "paused"})
                         or (a_step == "none" and a_run == "paused"  # if none -> any
                             and b_run in {"running", "paused"})))

        def new_main():
            return (a_loop == "main" == b_loop
                    and b_step != "none"      # CHECK: Can this be none here?
                    and ((a_run == "finished"  # finished -> {running, paused}
                          and b_run in {"running", "paused"})
                         or (a_step == "none" and a_run == "paused"  # if none -> any
                             and b_run in {"running", "paused"})))

        def main_to_adhoc():
            return (a_loop == "main" and b_loop == "adhoc"  # main -> {adhoc, user}
                    and b_step != "none"
                    and ((a_run in {"paused", "running", "finished"}
                          and b_run in {"paused", "running"})))  # paused <-> running

        def main_to_user():
            return (a_loop == "main" != b_loop  # main -> {adhoc, user}
                    and b_step != "none"
                    and ((a_run in {"paused", "running", "finished"}
                          and b_run in {"running"})))  # paused <-> running

        def adhoc_to_main():
            return (a_loop in {"adhoc"} and b_loop == "main"
                    and b_step != "none"
                    and a_run in {"paused", "finished"}
                    and b_run in {"paused", "running"})

        def user_to_main():
            return (a_loop in {"user"} and b_loop == "main"
                    and b_step != "none"
                    and a_run in {"running", "finished"}
                    and b_run in {"paused", "running"})

        def adhoc_to_adhoc():
            return (a_loop in {"adhoc"} and a_loop == b_loop
                    and a_step == b_step
                    and b_step != "none"
                    and a_run in {"paused", "running"}
                    and b_run in {"paused", "running", "finished"})

        def user_to_user():
            return (a_loop in {"user"} and a_loop == b_loop
                    and b_step != "none"
                    and a_step == b_step
                    and a_run in {"running"} and b_run in {"finished"})

        def adhoc_to_user():
            return (a_loop in {"adhoc"} and b_loop in {"user"}
                    and b_step != "none"
                    and a_run in {"paused", "running", "finished"}
                    and b_run in {"running"})

        def user_to_adhoc():
            return (a_loop in {"user"} and b_loop in {"adhoc"}
                    and b_step != "none"
                    and a_run in {"running", "finished"}
                    and b_run in {"paused", "running"})

        target = None
        # CHECK: To force or not to force. That is the question
        if to_none():
            print("TO NONE noop")
        elif main_none():
            print("MAIN none")
            self._ensure_thread(a_loop, self.train)
        elif main_same_step():
            print("MAIN same step")
            self._ensure_thread(a_loop, self.train)
        elif main_different_step():
            print("MAIN different step")
            self._ensure_thread(b_loop, self.train)
        elif new_main():
            print("NEW MAIN")
            self._ensure_thread(b_loop, self.train)
        elif main_to_adhoc() or main_to_user():
            print("MAIN to adhoc/user")
            self._task_runners["adhoc"].reset()
            if b_loop == "adhoc":
                target = getattr(self._task_runners[b_loop], "run_" + b_step)
            elif b_loop == "user":
                target = getattr(self._task_runners[b_loop], "run_task")
            # NOTE: The step function and loader That IS TO BE specified in task_args
            args = self._task_args[b_loop]
            # if b_loop == "adhoc" and b_step in {"val", "test", "extra"}:
            #     if "gather" in self._task_flags:
            #         # NOTE: callback is only called at finish
            #         kwargs = {"callback": partial(self._finish_and_resume_main, True)}
            #     else:
            #         kwargs = {"callback": self._finish_and_resume_main}
            # else:
            #     kwargs = {}
            self._ensure_thread(b_loop, target, args)
            self._prev_paused_state = self.current_state
            self._task_args[b_loop] = []
            time.sleep(.1)
            # NOTE: should be a function which handles control back to _transition_func
        elif adhoc_to_main() or user_to_main():
            print("ADHOC/USER to MAIN")
            if self._prev_paused_state and _to != self._prev_paused_state:
                self._loge(f"to state {_to} has to be previous paused state" +
                           f"{self._prev_paused_state}")
            self._ensure_thread(b_loop, self.train)
        elif adhoc_to_adhoc():
            print("ADHOC/ADHOC", _from, _to)
        elif user_to_user():
            print("USER/USER", _from, _to)
        elif adhoc_to_user() or user_to_adhoc():
            print("ADHOC/USER to USER/ADHOC", _from, _to)
            self._ensure_thread(b_loop, target)
        else:
            return False, self._loge(f"Not allowed transition {_from} -> {_to}")

        if a_run == "running":
            self._ensure_paused(a_loop, a_step)
        if a_run == "finished":
            self._ensure_finished(a_loop, a_step)
        self._current_state = _to
        if b_run == {"paused"}:
            self._ensure_paused(b_loop, b_step)
        if b_run == "running":
            self._ensure_ready(b_loop, b_step)
            self._ensure_unpaused(b_loop)
        if b_run == "finished":
            self._ensure_finished(b_loop, b_step, **self._transition_flags)
            self._transition_flags = {}
            # import ipdb; ipdb.set_trace()
            self._threads[b_loop].join()
            if b_loop != "main" and self._prev_paused_state:
                self._transition(self.current_state, self._prev_paused_state)
                self._prev_paused_state = None

        # TODO: Other Scenarios
        #       1. main thread died?
        #       2. main thread aborted and started main thread with val?
        #       3. alt thread create?
        #       4. alt thread reset?
        # NOTE: Actual state transition
        # NOTE: The Task runner should exist already if not, then some error
        #        has occured and previous state should be resumed.
        # if a_loop == b_loop == "main" and b_step in {"train", "val", "test"}:
        #     if "main" in self._task_runners:
        #         if not self._task_runners["main"].running:
        #             if "main" not in self._threads or not self._threads["main"].is_alive():
        #                 print("starting main")
        #                 self._threads["main"] = Thread(target=self.train)
        #                 self._threads["main"].start()
        #     else:
        #         # TODO: Throw massive error or go back to force_paused_train, or
        #         #       force_paused_none I suppose force_paused_none indicates
        #         #       that some massive error indeed occurred and we're stuck here.
        #         # NOTE: For now assume that it doesn't die
        #         pass
        # elif a_loop == "main" and b_loop in {"adhoc", "user"}:
        #     # CHECK: Pause or run on CPU or something?
        #     if "adhoc" in self._task_runners:
        #         if not self._task_runners["adhoc"].running:
        #             if "adhoc" not in self._threads or not self._threads["adhoc"].is_alive():
        #                 self._threads["adhoc"] = Thread(target=self._adhoc_func,
        #                                                 args=[self._adhoc_func_params])
        #                 self._threads["adhoc"].start()
        #     else:
        #         # Resume previous state
        #         pass
        # elif a_loop in {"adhoc", "user"} and b_loop == "main":
        #     # CHECK: What can a user func do?
        #     if "user" in self._task_runners:
        #         if not self._task_runners["user"].running:
        #             if "user" not in self._threads or not self._threads["main"].is_alive():
        #                 self._threads["user"] = Thread(target=self.train)
        #                 self._threads["user"].start()
        #     else:
        #         # Resume previous state
        #         pass
        # elif a_loop in {"adhoc", "user"} and b_loop in {"adhoc", "user"} and a_loop != b_loop:
        #     pass
        # elif a_loop in {"adhoc", "user"} and b_loop in {"adhoc", "user"} and a_loop == b_loop:
        #     pass

        # # NOTE: Post transition checks
        # if b_run == "running":
        #     self._ensure_unpaused(b_step)
        # elif b_run == "paused":
        #     self._ensure_paused(b_step)
        # elif b_run == "finished":
        #     self._ensure_finished(b_step, **self._transition_flags)
        #     self._transition_flags = {}
        #     self._threads["main"].join()
        # if b_run in {"paused", "running"}:
        #     self._ensure_ready(b_step)
        return True, self._logd(f"Transitioned from {_from} to {_to}")
    # END: State Machine

    # START: Extras
    @POST
    @extras
    def load_saves(self, weights: PathType, method: str) -> Return:
        """Load model weights or trainer state from a given filename.

        The file must be present in the :attr:`savedir`.

        Args:
            weights: The name of the weights file
            method: How to load the saves

        Returns:
            An instance of :class:`Return`

        Not sure right now, when something should be allowed to load. If it's
        paused? In the middle of current session? Should the session be
        restarted?

        """
        self._logi("Calling load saves")
        # NOTE: Proposed Checks mechanism. Actually it doesn't check at each
        #       step, so it's a bit buggy but something like this can be made.

        # checks = Checks(self._logd, self._loge)
        # checks.add("weights" in data, "Missing params \"weights\"")
        # checks.add("method" in data and data["method"] in {"resume", "load"},
        #            "Invalid or no such method")
        # checks.add(data["weights"] in self.saves, "No such file")
        # checks.check_all_true()
        # weights = data["weights"]
        # if data["method"] == "load":
        #     with checks.catch_and_log(f"Successfuly loaded weights {weights}",
        #                               f"Could not load weights {weights}") as ct:
        #         if ct:
        #             load_state = torch.load(os.path.join(self._savedir, data["weights"]))
        #             for name in self._models.names:
        #                 self._models.load_weights(name, load_state["models"][name])
        # else:
        #     with checks.catch_and_log("Resuming from file",
        #                               f"Could not resume from {weights}.") as ct:
        #         if ct:
        #             self._resume_from_path(os.path.join(self._savedir, data["weights"]))
        # return checks.status, checks.message

        # FIXME: Should not be allowed willy nilly
        # self._pause_if_running()
        # with self._paused_for_task():
        #     self.do_something()
        if method not in {"resume", "load"}:
            return make_return(False, self._logi("Invalid or no such method"))
        if weights not in os.listdir(self._savedir):
            return make_return(False, self._logi("No such file"))
        else:
            if method == "load":
                load_state = torch.load(os.path.join(self._savedir, weights))
                try:
                    for name in self.models:
                        self._models[name].load_weights(
                            {"name": name, "weights": load_state["models"][name]})
                except Exception as e:
                    return make_return(False, self._loge(f"Could not load weights {weights}.\n" +
                                                         f"Error occured {e}\n" +
                                                         traceback.format_exc()))
                return make_return(True, self._logi(f"Successfuly loaded weights {weights}"))
            else:
                try:
                    self._resume_from_path(os.path.join(self._savedir, weights))
                except Exception as e:
                    return make_return(False, self._loge(f"Could not load weights {weights}.\n" +
                                                         f"Error occured {e}\n" +
                                                         traceback.format_exc()))
                return make_return(True, self._logi(f"Resumed from file"))

    # CHECK: I think it's more generic now.
    @POST
    @extras
    def call_user_func(self, data: Dict[str, Any]) ->\
            Union[Return, ReturnExtraInfo]:
        """Call an arbitrary function. For now calls any of train/val/test or given
        update_funcs with a subset of the dataset.

        This function is more generic than adhoc_eval in the sense that any
        adhoc function can be called on any attribute of the trainer (as of now).

        Later only specified variables will be exposed.

        """
        if not data:
            return make_return(False, self._loge(f"Called with null data"))
        elif len(data) != 1:
            return make_return(False, self._loge(f"Can only call one function at a time. data is: {data}"))
        self._logi(f"Calling with data: {data}")
        # NOTE: Function is not a dict right now
        # func_name = [*data.keys()][0]
        func_name = data[0]
        if func_name not in self.user_funcs:
            return make_info(False, self._loge(f"Unknown function {func_name} given"),
                             {"available_functions": self.user_funcs})
        elif func_name in self.user_funcs:
            func = self._user_funcs[func_name]
            if not all(getattr(self, x, None) for x in inspect.signature(func).parameters):
                return make_return(False, self._loge(f"Some of the parameters for {func_name}: " +
                                                     f"{inspect.signature(func).parameters}" +
                                                     " are not available"))
            else:
                params = {x: getattr(self, x) for x in inspect.signature(func).parameters}
            self._logi(f"Running the given user func {func_name}")
            # FIXME: Switch to task runner
            pool = ThreadPool(processes=1)
            async_result = pool.apply_async(func, kwds=params)
            while not async_result.ready():
                time.sleep(1)
            try:
                output, callback = async_result.get()
            except Exception as e:
                return make_return(False, f"Unexpected output format for function {func_name}," +\
                                   f" Error occured {e}" + f"\n{traceback.format_exc()}")
            if callback not in self.user_funcs:
                return make_info(False, self._loge(f"Unknown function {callback} given"),
                                 {"available_functions": self.user_funcs})
            elif callback in self.user_funcs:
                callback_func = self._user_funcs[callback]
                param_names = [x for x in inspect.signature(callback_func).parameters]
                flag = True
                for _x in param_names:
                    if not getattr(self, _x, None) and _x != "output":
                        flag = False
                        break
                if not flag:
                    return make_return(False,
                                       self._loge(f"Some of the parameters for callback function" +
                                                  f" {callback}: " +
                                                  f" {param_names}" +
                                                  " are not available"))
                else:
                    params = {"output": output}
                    param_names.remove("output")
                    for _x in param_names:
                        params[_x] = getattr(self, _x)
                    return callback_func(**params)
            else:
                return make_return(False, "Unknown execution flow")
        else:
            return make_return(False, "Unknown execution flow")

    @POST
    @extras
    def force_run_hooks(self, data: Dict[str, Any]) -> Return:
        "Run the specified post_epoch_hooks_before the epoch is completed"
        return make_return(False, self._logi("Does nothing for now"))

    # TODO: Functions like this should return a json like form to update to the server
    #       For each such endpoint, there should be a "endpoint_params" endpoint which
    #       sends the required json_data format which is to be sent with the request
    #       Which should then be presented as a table to the user.
    #       There should be NO NESTING.

    # TODO: A curious case occurs because train, val, test are not only
    #       step_names but also dataset subsets. That may create confusion
    #
    # TODO: sphinx doctest setup
    #       >>> adhoc_eval(self, None)
    #           False, "Called with null data"
    #       should be converted to:
    #       .. doctest::
    #          adhoc_eval(self, None)
    #          # or self.adhoc_eval(None), not sure
    #          False, "Called with null data"
    # @POST
    # @extras
    # def adhoc_eval(self, data: Dict[str, Any]) -> Union[Return, ReturnExtraInfo]:
    #     """Do an arbitrary evaluation on any or combination of train/val/test data for
    #     any state in the model's stored history

    #     1. Create a new task runner
    #     2. adhoc_eval is called on either or combination of train/val/test data
    #     3. What to gather and callback can be specified beforehand
    #     4. Result is stored in _adhoc_func_result
    #     5. Callbacks can be called on the result multiple times.

    #     :param data: data
    #     :returns: status and response string
    #     :rtype: :class:`tuple`

    #     """
    #     if not data:
    #         return make_return(False, self._loge(f"Called with null data"))
    #     self._logi(f"Calling adhoc_eval with data: {data}")
    #     if not any(x in data for x in self.trainer_params.training_steps):
    #         return make_info(False, self._logi("Required Input. Given unknown dataset"),
    #                          **self.adhoc_error_dict)
    #     else:
    #         for x in data:
    #             return self.check_adhoc_eval_params(x, data[x])

    @POST
    @extras
    def adhoc_eval(self, adhoc_step: str,
                   params: AdhocEvalParams) -> Union[Return, ReturnExtraInfo]:
        """Do an arbitrary evaluation on any or combination of train/val/test data for
        any state in the model's stored history.

        Args:
            adhoc_step: Should be a valid function present in :class:`Trainer`'s
                        namespace and compatible with :class:`ModelStep`
            params: Parameters for execution. See :class:`AdhocEvalParams`

        Returns:
            An instance of :class:`Return` or :class:`ReturnExtraInfo`

        """
        # NOTE: Samples should be captured by default, model defines a sampling
        #       mechanism or else simply output is captured
        if params.epoch != "current" or self.epoch != params.epoch:
            self._logw("Param: \"epoch\" is ignored for now")
            pass
        # NOTE: From now on gather everything the step_func returns and wait for
        #       callback. "metrics" is removed
        # elif not (params["metrics"] != "all") or\
        #      not all(x in self._metrics[params["data"]] for x in params["metrics"]):
        #     self._logd(f'metrics given {params["metrics"]}')
        #     return False, {"error": self._loge("Required Input. Given unknown metrics or incorrect format"),
        #                    **self.adhoc_error_dict}
        # elif func not in (self.user_funcs + self.trainer_params.training_steps):
        #     return False, {"error": self._loge(f"Unknown function \"{params['function']}\" given"),
        #                    "available_functions": (self.user_funcs +
        #                                            self.trainer_params.training_steps)}
        # Making minimal assumptions on the function
        # elif func in self.user_funcs and\
        #      not len(inspect.signature(self._user_funcs[func]).parameters) == 1:
        #     return False, {"error": self._loge(f"Given function \"{params['function']}\"" +
        #                                        " is not suited to process data")}

        # call this in a separate thread and call the callback on the result
        # then report it.

        # NOTE: New thread should only be started from _transition
        self._adhoc_func = self.call_adhoc_eval_on_data
        self._adhoc_func_params = params
        self.call_adhoc_eval_on_data(params)
        self.pause()
        while not self.paused:
            time.sleep(10)
        t = Thread(target=self.call_adhoc_eval_on_data, args=[params])
        t.start()
        return make_return(True, self._logi("Running the given adhoc function"))

    def call_adhoc_eval_on_data(self, params: AdhocEvalParams):
        """Call adhoc evaluation on data and wait for result.

        Result is processed by the callback given by the user.

        Args:
            params: Parameters specified for the function

        """
        step = params.data
        function = self.update_functions[step]
        step_loader = getattr(self, step + "_loader")
        if params.seed:
            np.random.seed(params.seed)
        if params.num_or_fraction >= 1:
            indices = np.random.choice(len(step_loader.dataset), params.num_or_fraction)
        else:
            indices = np.random.choice(len(step_loader.dataset),
                                       int(len(step_loader.dataset) * params.num_or_fraction))
        _proxy_dataset = ProxyDataset(step_loader.dataset, indices)
        temp_params = self.dataloader_params[step].copy()
        self._logw("batch_size is set to 1 right now")
        temp_params.update({"batch_size": 1})  # stick to 1 right now

        # NOTE: MyDataLoader is to solve the problem of collation in data. So
        #       that there's a uniform interface to data. However there seem to
        #       be some problems.
        def get_temp_loader(tensorify=False):
            if hasattr(step_loader.dataset, "_get_raw"):
                _proxy_dataset._get_raw = lambda x: step_loader.dataset._get_raw(
                    _proxy_dataset._indices[x])
                if tensorify:
                    temp_params["collate_fn"] = default_tensorify
                    temp_loader = MyDataLoader(_proxy_dataset, return_raw=True,
                                               **temp_params)
                else:
                    temp_loader = MyDataLoader(_proxy_dataset, return_raw=True,
                                               **temp_params)
                self._logi(f"{step} dataset has \"_get_raw\"" +
                           " Drawing samples from temp data is available!")
            else:
                if tensorify:
                    temp_params["collate_fn"] = default_tensorify
                    temp_loader = MyDataLoader(_proxy_dataset, return_raw=False, **temp_params)
                else:
                    temp_loader = MyDataLoader(_proxy_dataset, return_raw=False, **temp_params)
                self._logw(f"{step} dataset doesn't define \"_get_raw\"" +
                           " Drawing samples from temp data will not be available.")
            return temp_loader
        temp_loader = get_temp_loader()
        x = step_loader.__iter__().__next__()
        y = temp_loader.__iter__().__next__()
        tensorify = False
        if not (isinstance(x[0], type(y[0])) and isinstance(x[1], type(y[1]))):
            print("CHECKING with tensorify", type(x[0]), type(x[1]), type(y[0]), type(y[1]))
            temp_loader = get_temp_loader(True)
            tensorify = True
            y = temp_loader.__iter__().__next__()
            # print("Y", y)
        if not (isinstance(x[0], type(y[0])) and isinstance(x[1], type(y[1]))):
            print("TYPES", type(x[0]), type(y[0]), type(x[1]), type(y[1]))
            print("NOPE returning")
            return False, f"Dataloader error"
        temp_loader = get_temp_loader(tensorify)
        if step == "train":
            raise NotImplementedError
            models = {}
            optimizers = {}
            devices = {}
            for model_name, model_params in self.model_params.items():
                models[model_name] = self.model_params[model_name].model(model_params.dict())
                optim_name = self.model_params[model_name].optimizer
                optimizers[model_name] = {"name": optim_name,
                                          "optimizer": self._optimizers
                                          [optim_name]["function"](
                                              models[model_name].parameters(),
                                              **self._optimizers[optim_name]["params"])}
                devices[model_name] = self._device
                # CHECK: This may not actually be needed
                #        May be needed if weights are updated.
                #        In that case dump/load may be a better option
                # TODO: Put extra metrics while building the step_func
                temp_models = Models(models, optimizers, devices, self.gpus, self.logger)
                # TODO: Load from checkpoint like this
                # _models.load(self._get_checkpoint(epoch)["models"])
                # TODO: Maybe let model also be altered, checkpoint of course should be
                temp_models.load(self._models.dump())  # replicate
        else:
            temp_models = self._models
        step_func = partial(function, temp_models, self.criteria)
        # CHECK: Should any of this be done manually?
        #        Shouldn't I leave all this up to user?
        metrics = [*self._metrics[step].keys()]
        if hasattr(step_loader.dataset, "_get_raw"):
            metrics.append("raw")
        metrics.append("predictions")
        metrics.append("labels")

        callback = self._user_funcs[params["callback"]]
        # NOTE: The callback should be called when the function finishes
        self._task_runner_initialize("adhoc", {step: metrics}, {}, callback)
        if hasattr(temp_loader.dataset, "_get_raw"):
            self._task_args["adhoc"] = [step_func, temp_loader, True]
        else:
            self._task_args["adhoc"] = [step_func, temp_loader]
        self._transition(self.current_state, "adhoc_running_" + step)

    @POST
    @extras
    def report_adhoc_run(self, data) -> Return:
        runner = self._task_runners["adhoc"]
        batch_vars = runner.batch_vars
        if data is None:
            return False, self._loge("Called with null data.")
        elif "report_function" not in data:
            return False, self._loge("report_function not in data.")
        elif data["report_function"] not in self._user_funcs:
            report_function = data["report_function"]
            return False, self._loge(f"Unknown report function {report_function}.")
        else:
            report_function = self._user_funcs[data["report_function"]]
        if not all(x in self._user_func_env_params
                   for x in inspect.signature(report_function).parameters):
            return False, self._loge(f"Some parameters required to run function" +
                                     f" {data['report_function']} are not available")
        if not hasattr(runner, "batch_vars"):
            return False, self._logd("Adhoc runner was never initialized")
        elif runner.running:
            return True, self._logd("Adhoc runner is still running")
        else:
            self._user_func_env(report_function, batch_vars=batch_vars)

    def _user_func_env(self, func, **kwargs):
        # TODO: check task completion
        param_names = list(inspect.signature(func).parameters.keys())
        params = {}
        # CHECK: Here. Can this be changed?
        for x in param_names:
            if x in self._user_func_env_params:
                params[x] = self._user_func_env_params[x]
            elif x in kwargs:
                params[x] = kwargs[x]
        output = func(**params)
        return True, {"success": output}

    # END: Extras

    # START: Methods
    @POST
    @methods
    def set_model(self, model_name: str) -> Return:
        """Set one of the available models as current active model"""
        return self._set_model_active(model_name)

    def _set_model_active(self, model_name: str) -> Return:
        """Set model with name `model_name` the active model.

        Model name is an abstraction and a `model` can have multiple
        :class:`torch.nn.Module` modules within it with separate criteria and
        optimizers. It is the prerogative of the update_function to interact
        with the model.

        Args:
            model_name: Name of the model

        Returns:
            An instance of :class:`Return`

        """
        if model_name not in self.models:
            return make_return(False, self._loge(f"No such model {model_name}"))
        else:
            for name in self._models:
                if name != model_name:  # free only GPU resources
                    self._models.set_device(model_name, torch.device("cpu"))
            for x in self.update_functions:
                self._training_steps[x]._model_name = model_name
            return make_return(True, self._logd(f"Model {model_name} is now the current active model."))

    # TODO: This should be a given input and not an image
    @POST
    @methods
    def fetch_predictions_for_data(self, img_path: PathType) -> ReturnExtraInfo:
        """Fetch the prediction for a given data instance

        Args:
            img_path: The path to the image

        Return:
            A :class:`dict` of type {"beam_preds": beam_preds, "greedy_preds": greedy_preds}

        """
        if True:              # img_path in self._temp_runner._processed_images:
            if not hasattr(self._temp_runner, "running") or self._temp_runner.running:
                return make_info(False, f"The function is still running")
            else:
                import ipdb; ipdb.set_trace()
                temp_list = [(x[1], [_x[0] for _x in x[-1]]) for x in self._temp_runner.batch_vars
                             if x[2] == "raw"]
                indx = None
                batch_preds = None
                batch_targets = None
                batch_lengths = None
                img_indx = None
                for x in temp_list:
                    if img_path in x[1]:
                        indx = x[0]
                        break
                if indx is not None:
                    for x in self._temp_runner.batch_vars:
                        if x[1] == indx and x[2] == "predictions":
                            batch_preds = x[-1]
                        if x[1] == indx and x[2] in {"targets", "labels"}:
                            batch_targets = x[-1]
                        if x[1] == indx and x[2] == "lengths":
                            batch_lengths = x[-1].tolist()
                        if x[1] == indx and x[2] == "raw":
                            img_indx = [_x[0] for _x in x[-1]].index(img_path)
                else:
                    import ipdb; ipdb.set_trace()
                    return make_info(False, f"Img seems to not have been processed. Check.")
                if all(x is not None for x in [batch_preds, batch_lengths,
                                               img_indx, batch_targets]):
                    probs, indices = torch.topk(torch.nn.functional.softmax(batch_preds, 1), 5)
                    probs = probs.cpu().numpy()
                    indices = indices.cpu().numpy()
                    vocab = getattr(self.report_function, "keywords")["vocab"]
                    preds = [vocab.idx2word[int(x[0])] for x in indices]
                    topk = [[vocab.idx2word[int(x)] for x in y] for y in indices]
                    targets = [vocab.idx2word[int(x)] for x in batch_targets]
                    # _raw = [x for x in self._temp_runner.batch_vars if x[2] == "raw"]
                    return make_info(True, "Got predictions",
                                     {"preds_targets": [preds, targets],
                                      "topk_words": topk, "probs": probs})
                else:
                    return make_info(False, f"Img seems to not have been processed. Check.")
        elif not os.path.exists(img_path):
            return False, f"Image {img_path} doesn't exist"
        else:
            return False, f"Evaluation of single image is not implemented yet"
            # try:
            #     Image.open(img_path)
            # except Exception as e:
            #     return False, f"Error occurred while reading file {e}"
            # Assuming predictions exist already somewhere in the report

    # TODO: These functions for images can be generalized for arbitrary binary data
    @GET
    @methods
    def fetch_image(self, img_path: PathType) ->\
            Union[Return, ReturnBinary]:
        """Fetch the image from a given path.
        """
        img_path = os.path.join(self.image_root, img_path)
        if not os.path.exists(img_path):
            return make_return(False, self._logd(f"Image {img_path} doesn't exist"))
        else:
            try:
                Image.open(img_path)
            except Exception as e:
                return make_return(False, self._loge(f"Error occurred while reading file {e}" +
                                                     f"\n{traceback.format_exc()}"))
            with open(img_path, "rb") as img_file:
                return return_image(True, base64.b64encode(img_file.read()).decode("utf-8"))

    @POST
    @methods
    def load_image(self, request) -> Return:
        """Load an image from the given data and call given functions on it.
        """
        try:
            img_file = request.files["file"].read()
            test = self._check_file_magic(img_file, "image")
        except Exception as e:
            return make_return(False, self._loge(f"Error reading file {e}" +
                                                 f"\n{traceback.format_exc()}"))
        if test:
            import skimage
            self._logd("Detected image file")
            img = skimage.io.imread(io.BytesIO(img_file))
            # At this point, just run
            func_names = json.loads(request.form["callbacks"])
            funcs = [self._user_funcs[f] for f in func_names]
            model = self._models[self.active_model]
            self._logd(f"Calling functions {func_names}")
            ix_to_word = self.train_loader.loader._ix_to_word
            funcs[0](model, img, ix_to_word, funcs[1])
            # FIXME: It calls the callback but return message should be better
            return make_return(True, "meh")
        else:
            return make_return(False, self._loge("Data is not image"))

    @POST
    @methods
    def load_weights(self, model_names: List[str], weights: bytes) -> Return:
        """Load weights for given model names.

        The weights file must be sent in the request along with the model names
        and must be loadable from :meth:`torch.load` with all the model names
        present in the keys for the file.

        Args:
            model_names: A list of model names
            weights_file: File with all the model names present in the keys for the file.

        Returns:
            An instance of :class:`Return`

        """
        try:
            weights = torch.load(io.BytesIO(weights), map_location="cpu")
        except Exception as e:
            return make_return(False,
                               self._loge(f"Error occured while reading data {e}" +
                                          f"\n{traceback.format_exc()}"))
        if not all(x in weights for x in model_names):  # type: ignore
            return make_return(False,
                               self._logd(f"Check save file! Not all {model_names} " +
                                          f"in given weights {weights.keys()}"))  # type: ignore
        if not all(x in self.models for x in model_names):
            return make_return(False, self._logd(f"Some models currently not in scope"))
        try:
            for model_name in model_names:
                status, err = self._models[model_name].load_weights(
                    {"name": model_name, "weights": weights[model_name]})  # type: ignore
                if err:
                    return make_return(False, self._loge(f"Error while updating component {err}"))
            return make_return(True, self._logd(f"Updated Models {model_names}"))
        except Exception as e:
            return make_return(False, self._loge(f"Error occured while loading models {e}" +
                                                 f"\n{traceback.format_exc()}"))

    @POST
    @methods
    def hack_param(self, data: Dict[str, Dict[str, str]]) -> Return:
        """Update a param as a hack.

        Data is assumed to be a pair of `{key, [type, value]}` dictionary. Only
        [str, int, float, bool] (simple) types are accepted.

        """
        statuses = []
        type_dict = {"str": str, "int": int, "float": float, "bool": bool}
        for k, v in data.items():
            if not hasattr(self, k):
                self._logw(f"{k} not an attribute of {self}. Will add.")
                if v["type"] in type_dict.keys():
                    _v = type_dict[v["type"]](v["value"])
                    setattr(self, k, _v)
                    statuses.append(True)
                else:
                    self._loge(f"Not a recognized type for {k} {v['type']}")
                    statuses.append(False)
            else:
                try:
                    if v["type"] in type_dict.keys():
                        _v = type_dict[v["type"]](v["value"])
                        if k not in self.__class__.__dict__:
                            setattr(self, k, _v)
                            self._logi(f"Set param {k} to {_v} successfully!")
                        else:
                            self._loge(f"Cannot modify class attr {k}")
                            statuses.append(False)
                        statuses.append(True)
                    else:
                        self._loge(f"Not a recognized type for {k} {v['type']}")
                        statuses.append(False)
                except Exception as e:
                    statuses.append(False)
                    self._loge(f"could not set value for {k}: {v['value']}. Error {e}")
        if all(statuses):
            return make_return(True, "All values updated!")
        elif any(statuses):
            return make_return(True, "Some values could not be updated.")
        else:
            return make_return(False, "None of the values could be updated.")

    @POST
    @methods
    def add_user_funcs(self, request: flask.Request) -> Return:
        """Adds user given functions.
        Delegates to :meth:`~trainer.mods.Modules.add_user_funcs`

        """
        return self._modules.add_user_funcs(request, self._user_funcs)

    @POST
    @methods
    def add_model(self, model_str: Union[str, bytes]) -> Return:
        """Add a model from a given python or module as a zip file.  Delegates the
        request to :meth:`add_module`

        For this case `module_exports` has to include models and
        optimizers. Optimizers can be a string and if so, if optimizer_params
        are given then it is initialized with that, else the params are checked
        in instance scope. If both aren't present it's initialized with default
        params from :mod:`torch.optim`

        Args:
            request: http request forwarded from the daemon

        Example: `file_which_is_sent.py`::

            class SomeModel:
                # code

            class AnotherModel:
                # code

            class YourOptimizer:
                # code

            module_exports = {"model_params": {"model_1": {"model": SomeModel, "optimizer": "Adam",
                                                           "param_1": 123, "param_2": "bleh"},
                                               "model_2": {"model": AnotherModel,
                                                           "optimizer": "YourOptimizer",
                                                           "param_1": True, "param_2": False}}
                              "optimizers": {"YourOptimizer": {"function": YourOptimizer,
                                                               "params": {"lr": 0.01, "theta": 0.9}}}}

        In the above example`Adam`'s parameteres are omitted here for `model_1`
        so first they'll be searched in existing optimizers with trainer, else
        no params will be given to the function that retuns the `Adam` instance.

        In case `Adam` isn't present in the optimizers at all, it will be
        searched in :mod:`torch.optim` and in case even there it's not found, an
        error will be raised.

        Returns:
            A tuple of status and message

        """
        status, response = self.add_module(model_str, None, [])
        if status:
            module_exports = response
            if "model_params" not in module_exports:
                return make_return(False, self._logw(f"Model params not sent in data"))
            elif "optimizers" not in module_exports:
                return make_return(False, self._logw(f"Optimizers not sent in data"))
            else:
                return self._add_new_models_helper(module_exports["model_params"],
                                                   module_exports["optimizers"])
        else:
            return make_return(status, response)

    # TODO: Any change in state of trainer vars should have a rollback mechanism
    #       E.g., if some of the params change here and then an error is raised.
    def _add_new_models_helper(self, model_params: Dict[str, Dict[str, Union[str, Dict, Callable]]],
                               optimizers: Dict[str, config.Optimizer]):
        """Extract `models` from the `model_names`, `model_defs` and `model_params`

        Args:
            model_params: Model parameters
            optimizers: Optimizers

        Returns:
            A :class:`tuple` of `status`, `response` where if
            `status` is successful the response is model else an error string

        """
        try:
            model_names = [*model_params.keys()]
            status = {k: "" for k in model_names}
            for model in model_names:
                model_func = model_params[model]["model"]
                _params = model_params[model]["params"]
                if "__inherit" in _params:  # HACK
                    if "__add" in _params:  # FIXME: add params from self, stupid HACK
                        add_params = _params["__add"]
                    else:
                        add_params = []
                    inherit_name = _params["__inherit"]
                    sig = inspect.signature(model_func)
                    model_args = {}
                    for x in sig.parameters:
                        if x not in add_params:
                            model_args[x] = self.model_params[inherit_name][x]
                        else:
                            model_args[x] = getattr(self, x)  # FIXME: Bad HACK
                else:
                    model_args = _params
                if model in self.model_params:
                    self._logw(f"Will overwrite model, optimizer params and defs for {model}")
                    self._backup_model_params[model] = copy.deepcopy(self.model_params[model])
                self.model_params[model]["params"] = model_args.copy()
                self.model_params[model]["model"] = model_func
                self.model_params[model]["optimizer"] = model_params["optimizer"]
                self.model_params[model]["gpus"] = model_params["gpus"]
                self._logd(f"Updated model_params and model_def for {model}")
                self.allocate_devices(model)
                self._models[model] = self._model_init_helper(model)
                if model_params["load"]:
                    _status, response = self._models[model].load_into_memory()
                    if _status:
                        status[model] = True
                    else:
                        if model in self._backup_model_params[model]:
                            self.model_params[model] = copy.deepcopy(self._backup_model_params[model])
                            status[model] = False, "Reverted"
                        else:
                            status[model] = False, response
                else:
                    status[model] = True
                # models["models"][model] = model_func(**model_args)
                # if isinstance(model_defs[model]["optimizer"], str):
                #     optim_name = model_defs[model]["optimizer"]
                #     self._model_defs[model]["optimizer"] = optim_name
                #     self._logd(f"Updated optimizer params for {model}")
                #     models["optim_names"][model] = optim_name
                #     if "optimizer_params" in model_defs and hasattr(torch.optim, optim_name):
                #         models["optimizers"][model] = getattr(torch.optim, optim_name)(
                #             **model_defs[model]["optimizer_params"])
                #         self._logd(f"Initialized optimizer for {model} in add_model with given params")
                #     elif optim_name in self._optimizers:
                #         models["optimizers"][model] = self._optimizers[optim_name]["function"]\
                #             (models["models"][model].parameters(),
                #              **self._optimizers[optim_name]["params"])
                #         self._logd(f"Initialized optimizer for {model} in add_model with self params")
                #     else:
                #         models["optimizers"][model] = getattr(torch.optim, optim_name)()
                #         self._logw(f"Initialized optimizer for {model} in add_model with default params")
                # else:
                #     False, self._logd(f"Unrecognized optimizer for model {model}")
            return make_return(True, status)
        except Exception as e:
            return make_return(False, f"{e}")

    def exec_some_string(self, string: str):
        exec(string)

    @POST
    @methods
    def add_module(self, request: flask.Request,
                   checks: Iterable[Callable[[str], bool]] = []) -> Return:
        """File must be present in request and is read as :code:`request.files["file"]`

        Args:
            request: A :mod:`flask` request
            checks: An iterable of predicates. The input to the functions will be the
                    file path to which the module is written.
        """
        return make_return(*self._modules.add_module(request, checks))
    # END: Methods

    # START: Objects
    @objects                    # type: ignore
    @property
    def epoch_runner(self):
        return self._epoch_runner

    @POST
    @objects                    # type: ignore
    def task_runners(self):
        return self._task_runners
    # END: Objects

    # START: Save, Load, Resume
    @internals
    def _dump_state(self) -> Return:
        "Dump everything except weights"
        try:
            dump_path = os.path.join(self._data_dir, "session_state")
            state = self._get_state(True)
            with open(dump_path, "w") as f:
                f.write(state.json())
            return make_return(True, "Dumped")
        except Exception as e:
            return make_return(False, self._loge(f"{e}" + f"\n{traceback.format_exc()}"))

    def _get_state(self, lite=False) -> TrainerState:
        """Return the trainer state.

        State is a difficult thing to serialize, retrieve and
        restore. Essentially, all of the trainer should resume given:
            a. initial config
            b. current state

        We keep track of the state vars with the tag :attr:`state_vars`.
        Additionally, the state is defined in TrainerState for required
        variables, while some extra variables in state can be defined by the
        user.

        """
        state = {}
        for k in self.state_vars:
            if k == "models":
                if lite:
                    state["models"] = {x: self._models[x].dict() for x in self._models}
                else:
                    state["models"] = {x: self._models[x].dump() for x in self._models}
            else:
                state[k] = getattr(self, k)
        # state["epoch"] = self.epoch
        # state["given_name"] = getattr(self, "given_name", "")
        # state["iterations"] = self.iterations
        # if lite:
        #     state["models"] = {x: self._models[x].dict() for x in self._models}
        # else:
        #     state["models"] = {x: self._models[x].dump() for x in self._models}
        # state["model_params"] = copy.deepcopy(self.model_params)
        # state["criteria_params"] = copy.deepcopy(self.criteria_params)
        # state["data"] = self.data_params
        # state["dataloader_params"] = {}
        # for k, v in self.dataloader_params.items():
        #     if v is None:
        #         state["dataloader_params"][k] = None
        #     else:
        #         state["dataloader_params"][k] = {}
        #         for a, b in v.items():
        #             if a == "collate_fn":
        #                 self._logw(f"collate_fn in dataloader {k} params will not be saved")
        #                 state["dataloader_params"][k][a] = "callable_" + type(b).__qualname__
        #             else:
        #                 value = self.dataloader_params[k][a]
        #                 if isinstance(value, dict):
        #                     state["dataloader_params"][k][a] = {}
        #                     for x, y in value.items():
        #                         if callable(y):
        #                             self._logw(f"callable {type(y).__qualname__} in dataloader" +
        #                                        f" {k} params {a, x} will not be saved")
        #                             state["dataloader_params"][k][a][x] = "callable_" +\
        #                                 type(y).__qualname__
        #                         else:
        #                             state["dataloader_params"][k][a][x] = y
        #                 else:
        #                     if callable(value):
        #                         self._logw(f"callable {value} in dataloader {k}" +
        #                                    f" params {a} will not be saved")
        #                         state["dataloader_params"][k][a] = "callable_" +\
        #                             type(value).__qualname__
        #                     else:
        #                         state["dataloader_params"][k][a] = value
        # state["trainer_params"] = {}
        # for k, v in self.trainer_params.items():
        #     if callable(v):
        #         self._logw(f"callable {type(v).__qualname__}" +
        #                    f" for trainer_params {k} will not be saved")
        #         state["trainer_params"][k] = "callable_" + type(v).__qualname__
        #     else:
        #         state["trainer_params"][k] = copy.deepcopy(v)
        # state["metrics"] = self._metrics
        # NOTE: Extra items for tracking required by daemon
        state["max_epochs"] = self.max_epochs
        state["max_iterations"] = self.max_iterations
        # CHECK: If there's a given_name
        state["data"] = self.data_params.name
        state["given_name"] = getattr(self, "given_name", "")
        state["mode"] = "lite" if lite else "full"  # type: ignore
        return TrainerState.parse_obj(state)

    def _save(self, save_path=None, best=False):
        if not save_path:
            save_path = self._save_path_with_epoch
        if best:
            if not save_path.endswith(".pth"):
                save_path += "_best.pth"
            else:
                save_path = save_path.replace(".pth", "") + "_best.pth"
        elif not save_path.endswith(".pth"):
            save_path += ".pth"
        self._logd(f"Trying to save to {save_path}")
        save_state = self._get_state()
        self._logi(f"Saving to {save_path}")
        torch.save(save_state, save_path)
        # # FIXME: If some thing is not saved, it cannot be resumed also
        # # NOTE: As I've removed callable saving from dataloader params,
        # #       this should save now
        # def try_save():
        #     try:
        #         torch.save(save_state, save_path)
        #         not_saved = False
        #     except Exception:
        #         not_saved = True
        #     return not_saved
        # not_saved = try_save()
        # while not_saved:
        #     self.fix_state(save_state, save_path)
        #     not_saved = try_save()

    # def fix_state(self, save_state: Dict, save_path: str):
    #     """Fix serialization errors in trainer state.

    #     Used to save with :func:`torch.save`.

    #     """
    #     tmp_path = save_path + "_tmp"

    #     def find_error_key(state_dict):
    #         for x in state_dict.keys():
    #             try:
    #                 torch.save(state_dict[x], tmp_path)
    #             except Exception:
    #                 return x
    #         return None
    #     state_dict = save_state
    #     keys = []
    #     while True:
    #         error_key = find_error_key(state_dict)
    #         if error_key is None:
    #             break
    #         else:
    #             keys.append(error_key)
    #             state_dict = state_dict[error_key]
    #     x = save_state
    #     if keys:
    #         for key in keys[:-1]:
    #             x = x[key]
    #         x[keys[-1]] = type(x[keys[-1]]).__qualname__
    #         self._logw(f"Value with keychain {keys} could not be saved." +
    #                    f" Replaced with name = {x[keys[-1]]}")
    #         if "not_saved" not in save_state:
    #             save_state["not_saved"] = [keys]
    #         else:
    #             save_state["not_saved"].append(keys)
    #         os.remove(tmp_path)
    #     else:
    #         self._logw(f"Nothing to fix in state")

    # TODO: Right now, the list of saves and resume_path etc are given as full paths while
    #       they should be relative paths to .savedir/"_".join(model_names)
    def _resume_from_path(self, resume_path: PathType):
        saved_state = torch.load(resume_path)
        self._resume_from_state(saved_state)

    def _resume_from_state(self, saved_state: TrainerState) -> Return:
        """Resume the trainer from a given state.

        Trainer can only be resumed if the models and dataloaders are
        identical. If either of them differs, it's essentially a new training
        session and should be run as such. Everything else (mostly parameters)
        can be changed in the resume state.

        """
        if self._have_resumed:
            make_return(False, "Already resumed")
        self._backup_state = self._get_state()
        self._have_resumed = True
        if isinstance(saved_state, dict):
            saved_keys = saved_state.keys()
        elif isinstance(saved_state, TrainerState):
            saved_keys = saved_state.dict().keys()
        else:
            return reterr(self._loge("Unknown data for saved state. Could not load."))
        diff = diff_as_sets(self._backup_state.dict().keys(), saved_keys)
        if diff:
            return reterr(self._loge(f"Could not load saved_state. Missing keys {diff}"))
        if isinstance(saved_state, dict):
            data_name = saved_state["data"]
        else:
            data_name = saved_state.data
        if data_name != self.data_params.name:
            return reterr(self._loge("Cannot load saved_state, Different datasets: " +
                                     f"{data_name}, {self.data_params}"))
        if isinstance(saved_state, dict):
            saved_models = saved_state["models"].keys()
        else:
            saved_models = saved_state.models.keys()     # type: ignore
        # TODO: Should allow extra models
        diff = diff_as_sets(self.models, saved_models)  # type: ignore
        if diff:  # type: ignore
            return reterr(self._loge("Could not load saved_state. " +
                                     f"Some required models not in saved state {diff}"))
        if isinstance(saved_state, dict):
            saved_dict = saved_state
        else:
            saved_dict = saved_state.dict()
        try:
            self._init_all()
            for k, v in saved_dict.items():
                if k == "models":
                    for model, model_state in v.items():
                        if model in self._models:
                            self._models[model].load(v)
                        # TODO: This has to be initialized manually as optimizer
                        #       definition in case would be required.
                        #
                        # elif "model_def" in model_state:  # pickled state is error prone
                        #     self._models[model] = Model.from_dump(model_state)
                        else:
                            self._logw(f"Model {model} has no definition. Will not add.")
                elif k == "saves" and isinstance(v, dict):
                    # write the files
                    pass
                elif k not in {"loaded_models", "active_model", "hooks", "devices",
                               "saves", "metrics", "allocated_devices", "extra_metrics",
                               "extra_items"}:
                    # NOTE: state_vars
                    setattr(self, k, v)
            if saved_dict["extra_items"]:
                for k, v in saved_dict["extra_items"].items():
                    setattr(self, k, v)
            # NOTE: restore metrics
            diff = diff_as_sets(self._metrics.keys(), saved_dict["metrics"].keys())
            if diff:
                self._logw(f"Some metric _steps_ aren't there in saved state {diff}")
            for k in self._metrics.keys():
                diff = diff_as_sets(self._metrics.keys(), saved_dict["metrics"][k].keys())
                if diff:
                    self._logw(f"Some metrics {diff} in {k} aren't there in saved state")
            self._logd("Restoring metrics")
            self._metrics = copy.deepcopy(saved_dict["metrics"])
            status, message = True, self._logi("Resumed successfully")
            self._dump_state()
        except Exception as e:
            status, message = False, f"Error occurred. {e}" + "\n" + traceback.format_exc()
            self._rollback_resume()
        return make_return(status, message)

    def _rollback_resume(self):
        self._init_all()
        self._resume_from_state(self._backup_state)

    def check_and_save(self):
        if self._check_func is not None:
            assert ("when" in self._check_func.requires and
                    self._check_func.requires["when"]
                    in ["train", "val", "test"]), "Not sure when to save"
            when = self._check_func.requires["when"]
            assert all(x in self._metrics[when] for x in self._check_func.requires["metrics"]),\
                "self._check_func requirements not fulfilled"
            if self._check_func(self._metrics[when]):
                self._logi("Save check returned True.")
                self._save(None, True)
            else:
                self._logi("Save check returned False. Not saving")
        else:
            self._logi("Check func is None. Not saving")
    # END: Save, Load and Resume

    # START: Controls
    # # FIXME: resume_best can only be done if an index is kept which keeps
    # #        track of what's best.
    # # TODO: So basically save all the metrics outside in a separate file
    # # TODO: It may be some arbitrary predicate
    # def resume_best(self):
    #     """Resumes from the last best saved checkpoint. By default checks for lowest
    #     `val_acc`
    #     :returns: None
    #     :rtype: None
    #     """
    #     self._logd("Trying to resume last best checkpoint %s" % self.best_save)
    #     if self.best_save:
    #         self._resume_path = self.best_save

    # CHECK if this thing works correctly. There might be a few things I may have missed
    # TODO: For any worker loop which returns an error, the next
    #       one should pause or halt or something.
    @control
    def reset_session(self) -> Return:
        """Reset should dump all the variables and data to a backup state with the
        option to save, restore or delete later and reset all the state of the
        session to init_state.
        """
        try:
            self.stop()
            self.save()
            message = self._logi("Resetting the current session")
            self._init_all()
        except Exception as e:
            return make_return(False, f"{e}")
        return make_return(True, message)

    @control
    def pause(self) -> str:
        self._pause_if_running()
        return self._logi("Pausing")

    @control
    def resume(self) -> str:
        self._run_if_paused()
        return self._logi("Resuming")

    @control
    def start(self) -> str:
        self._start_if_not_running()
        return self._logi("Starting")

    # # CHECK: Can we resume after stop?
    # @control
    # def abort_current(self):
    #     """Stops training entirely. Does not reset the data and the state.
    #     Unlike self.abort`, `self.stop` will force stop the training, gather the
    #     metrics and terminate the running thread.
    #     :returns: None
    #     :rtype: None
    #     """
    #     self._abort_current()
    #     return True, self._logi("Forced stopped")
    #     # listen for commands

    @control
    def save(self) -> Return:
        self._logi("Saving")
        self._pause_if_running()
        self._logd("Trying force save")
        try:
            self._save(self._save_path_with_epoch + "_force")
            status = True
            message = f"Saved to {self._save_path_with_epoch}" + "_force"
        except Exception as e:
            status = False
            message = f"Could not save to {self._save_path_with_epoch}" + "_force" +\
                f" error {e}" + f"\n{traceback.format_exc()}"
        self._run_if_paused()
        return make_return(status, message)

    @control
    def force_eval(self) -> Return:
        """Do a full evaluation run on the adhoc loop"""
        if not self.val_loader:
            return make_return(False, "No val loader")
        self._pause_if_running()
        if self._task_runners["adhoc"] is None:
            device_monitor, signals = self._task_runner_helper("adhoc")
            self._task_runners["adhoc"] = self._init_default_task_runner(device_monitor, signals)
        self._task_args["adhoc"] = [self.val_step_func, self.val_loader]
        self._transition(self.current_state, "adhoc_running_val")
        # if self._prev_paused_state is not None:
        #     self._transition(self.current_state, self._prev_paused_state)
        # self._prev_paused_state = None

    @control
    def force_test(self) -> Return:
        if not self.test_loader:
            return make_return(False, "No test loader")
        self._pause_if_running()
        if self._task_runners["adhoc"] is None:
            device_monitor, signals = self._task_runner_helper("adhoc")
            self._task_runners["adhoc"] = self._init_default_task_runner(device_monitor, signals)
        self._task_args["adhoc"] = [self.test_step_func, self.test_loader]
        self._transition(self.current_state, "adhoc_running_test")
        # if self._prev_paused_state is not None:
        #     self._transition(self.current_state, self._prev_paused_state)
        # self._prev_paused_state = None

    @control
    def abort_session(self) -> Return:
        """`abort_session` finishes the session and switches to "finished" state with
        aborted flag set to true. Saves the current session with aborted suffix.

        :returns: None
        :rtype: None

        """
        try:
            self._abort_current("user")
            self.save()
            self._abort_session("user")
        except Exception as e:
            return make_return(False,
                               self._logi(f"Could not abort {self.current_state}. Error {e}" +
                                          f"\n{traceback.format_exc()}"))
        return make_return(True,
                           self._logi(f"Aborted state {self.current_state} and current session"))

    @control
    def abort_loop(self) -> Return:
        """`abort_loop` aborts only the current loop stops with the aborted flag. Useful
        for changing the parameters and starting again. Saves the current
        metrics gathered.

        """
        try:
            self._abort_current("user")
        except Exception as e:
            return make_return(False,
                               self._logi(f"Could not abort {self.current_state}. Error {e}" +
                                          f"\n{traceback.format_exc()}"))
        return make_return(True, self._logi(f"Aborted {self.current_state}"))

    @control
    def abort_loop_with_callback(self) -> Return:
        """`abort_loop` aborts only the current loop stops with the aborted flag. Useful
        for changing the parameters and starting again. Saves the current
        metrics gathered.

        :returns: None
        :rtype: None

        """
        try:
            self._abort_current_run_cb("user")
        except Exception as e:
            return make_return(False,
                               self._logi(f"Could not abort {self.current_state}. Error {e}" +
                                          f"\n{traceback.format_exc()}"))
        return make_return(True,
                           self._logi(f"Aborted {self.current_state}"))
    # END: Controls

    # START: Flags
    def _toggle_running(self):
        loop = self.current_state.split("_")[0]
        if loop == "main":
            if self._running_event.is_set():
                self._running_event.clear()
            else:
                self._running_event.set()
        elif loop == "adhoc":
            if self._adhoc_running_event.is_set():
                self._adhoc_running_event.clear()
            else:
                self._adhoc_running_event.set()
        elif loop == "user":
            if self._userfunc_running_event.is_set():
                self._userfunc_running_event.clear()
            else:
                self._userfunc_running_event.set()

    def _toggle_current_aborted(self):
        loop = self.current_state.split("_")[0]
        if loop == "main":
            if self._current_aborted_event.is_set():
                self._current_aborted_event.clear()
            else:
                self._current_aborted_event.set()
        elif loop == "adhoc":
            if self._adhoc_aborted_event.is_set():
                self._adhoc_aborted_event.clear()
            else:
                self._adhoc_aborted_event.set()
        elif loop == "user":
            if self._userfunc_aborted_event.is_set():
                self._userfunc_aborted_event.clear()
            else:
                self._userfunc_aborted_event.set()

    def _toggle_session_aborted(self):
        if self._session_aborted_event.is_set():
            self._session_aborted_event.clear()
        else:
            self._session_aborted_event.set()

    def _abort_session(self, cause: str):
        # any -> force_finished_none with aborted_session True
        self._toggle_session_aborted()
        self._transition(self.current_state, "force_finshed_none")

    def _abort_current(self, cause: str):
        if not self._current_aborted:
            self._toggle_current_aborted()
        self._finish_if_paused_or_running()
        # self._toggle_current_aborted() is called in _ensure_finished
        print("ABORTED flag", self._current_aborted)
        self._aborted.append([self.current_state.split("_")[-1], cause])

    def _abort_current_run_cb(self, cause: str):
        if not self._current_aborted:
            self._toggle_current_aborted()
        self._finish_if_paused_or_running(True)
        # self._toggle_current_aborted() is called in _ensure_finished
        print("ABORTED flag", self._current_aborted)
        self._aborted.append([self.current_state.split("_")[-1], cause])
    # END: Flags

    # START: Internal Controls Other
    def _force_validate(self):
        # Pauses main loop
        pass

    def _force_validate_parallel(self):
        # Runs in alternate loop
        pass

    def _force_test_parallel(self):
        # Runs in alternate loop
        pass

    def _force_test(self):
        # Pauses main loop
        pass

    def _run_user_func(self, user_func_name: str):
        # pauses main loop, shouldn't update weights
        pass

    def _run_user_func_parallel(self, user_func_name: str):
        # runs in alternate loop, if model is used will make a copy of model
        pass
    # END: Internal Controls Other

    # START: Properties
    @prop                       # type: ignore
    @property
    def version(self) -> str:
        "Version of the server"
        return self.__version__

    # @prop
    # @property
    # def have_cuda(self):
    #     "Do we have gpus? And cuda?"
    #     bleh = torch.cuda.is_available()
    #     return bleh

    @prop                       # type: ignore
    @property
    def current_state(self) -> Dict[str, Any]:
        "Current global state of the state machine"
        return self._current_state

    @prop                       # type: ignore
    @property
    def loop_type(self) -> str:
        """Loop type determines if the training monitor number of epochs or number of batches.
        """
        return self.trainer_params.training_type.value

    @property
    def logger(self):
        "Current :class:`logging.Logger` instance"
        return self._logger

    @prop                       # type: ignore
    @property
    def logfile(self) -> str:
        "Logfile contents"
        with open(self._logfile) as f:
            return f.read()

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def saves(self) -> List[str]:
        "A list of all the files in the Saves directory"
        return os.listdir(self._savedir)

    @prop                       # type: ignore
    @property
    def gpus(self) -> List[int]:
        """List of GPUs requested by the config."""
        return self.config.trainer_params.gpus

    @gpus.setter
    def gpus(self, x: Union[List[int], int, str]):
        try:
            self.config.trainer_params.gpus = x
        except Exception as e:
            self._loge(f"Could not set gpus {e}")

    @prop                       # type: ignore
    @property
    def system_info(self) -> Dict[str, Optional[Dict]]:
        "System Info: cpu_util, mem_util, gpu_util"
        return {"gpu_util": gpu_util(self._device_handles) if self.gpus[0] != -1 else None,
                "cpu_info": cpu_info(),
                "memory": memory_info()}

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def devices(self) -> Dict[str, List[int]]:
        """Used to keep track of which device(s) is(are) allocated to which model(s).

        Depending on the trainer and model parameters, the model(s) can be
        parallelized over multiple devices or reside on a single device. Device
        info is pulled from the models where it's stored separately for the
        models. Multiple models can reside on one device if it has enough VRAM
        and compute capacity.

        """
        return self._devices

    # @devices.setter
    # def devices(self, x: Union[int, List[int]]):
    #     if isinstance(x, int):
    #         self._devices

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def allocated_devices(self) -> List[int]:
        devices: List[int] = []
        for x in self._devices.values():
            devices.extend(x)
        return devices

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def models(self) -> List[str]:
        "Return the names of the models available with the server"
        return [*self._models.keys()]

    @prop                       # type: ignore
    @property
    def optimizers(self) -> Dict[str, config.Optimizer]:
        return self.config.optimizers

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def active_model(self) -> str:
        "Active model is both get and set by setting the _update_function"
        # NOTE: Was self.update_functions[self.trainer_params.training_steps[0]]._model_name
        #       "train" is assumed to be present as a step
        if self.update_functions.train is None:
            return ""
        else:
            return ", ".join(self.update_functions.train.models.keys())

    @prop                       # type: ignore
    @property
    def props(self) -> List[str]:
        """Return all properties of the instance including `extras` and `methods`
        except hidden properties
        """
        return [x for x, y in prop.members.items()
                if isinstance(y, property) and
                x != "props" and
                (x in {"extras", "methods"} or not x.startswith("_"))]

    @property
    def state_vars(self) -> List[str]:
        """Return all properties of the instance including `extras` and `methods`
        except hidden properties
        """
        return [*state_var.members.keys()]

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def epoch(self) -> int:
        "Current Epoch while training, if training type is epoch. See `loop_type`"
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @prop                       # type: ignore
    @property
    def max_epochs(self) -> int:
        "Max Epochs to train, if training type is epoch. See `loop_type`"
        return self.config.trainer_params.max_epochs

    @max_epochs.setter
    def max_epochs(self, x: int):
        self.config.trainer_params.max_epochs = x

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def iterations(self) -> int:
        "Current iteration if training type is iterations. See `loop_type`"
        return self._iterations

    @iterations.setter
    def iterations(self, x):
        self._iterations = x

    @prop                       # type: ignore
    @property
    def max_iterations(self) -> int:
        "Max iterations to run if training type is iterations. See `loop_type`"
        return self.trainer_params.max_iterations or 0

    @max_iterations.setter
    def max_iterations(self, x: int):
        self.config.trainer_params.max_iterations = x

    @prop                       # type: ignore
    @property
    def controls(self) -> Dict[str, Callable[[], str]]:
        """Controls are primary functions through which training is controlled. They
        affect the main training loop and other functions can be run either in
        parallel or while the main loop is paused.

        :rtype: `dict`
        """
        return control.members

    @prop                       # type: ignore
    @property
    def methods(self) -> Dict[str, Callable]:
        """Trainer methods are additional methods besides the controls to manage and
        modify the training session.  Containing such functions which upload
        weights to the server, load previous saves, add/set a model or view
        predictions after pausing the trainer.

        """
        return methods.members

    @prop                       # type: ignore
    @property
    def all_props(self) -> Dict[str, property]:
        return prop.members

    @prop                       # type: ignore
    @property
    def _internals(self) -> Dict[str, Union[Callable, property]]:
        """Internals are diagnostic tools to examine the state of the
        training. Currently only has one function: `_dump_state`."""
        return internals.members

    @prop                       # type: ignore
    @property
    def extras(self) -> Dict[str, Callable]:
        """Extras are experimental methods whose execution is more complicated and
        should be used with caution. They include calling an pausing and
        evaluating any combination of train/val/test data on current or any
        previous state, running an arbitrary user given function on the data, etc.

        """
        return extras.members

    # START: State props
    @prop                       # type: ignore
    @property
    def running(self) -> bool:
        "Is the current loop running?"
        return self._running_event.is_set()

    @prop                       # type: ignore
    @property
    @deprecated
    def current_run(self) -> str:
        "Which loop is the current loop? Can be one of [adhoc, user, main]"
        if "_epoch_runner" not in self.__dict__:
            return "None"
        else:
            return self._epoch_runner.current_loop

    @prop                       # type: ignore
    @property
    def paused(self) -> bool:
        "Is the current loop paused?"
        loop = self.current_state.split("_")[0]
        if loop == "main":
            return not self._running_event.is_set()
        elif loop == "adhoc":
            return not self._adhoc_running_event.is_set()
        elif loop == "user":
            return not self._userfunc_running_event.is_set()

    @prop                       # type: ignore
    @property
    def _current_aborted(self) -> bool:
        loop = self.current_state.split("_")[0]
        if loop == "main":
            return self._current_aborted_event.is_set()
        elif loop == "adhoc":
            return self._adhoc_aborted_event.is_set()
        elif loop == "user":
            return self._userfunc_aborted_event.is_set()

    @prop                       # type: ignore
    @property
    def _session_aborted(self) -> bool:
        return self._session_aborted_event.is_set()

    @prop                       # type: ignore
    @property
    def adhoc_paused(self) -> bool:
        "Is the adhoc loop paused?"
        return not self._adhoc_running_event.is_set()

    # Are adhoc_aborted and userfunc_aborted needed?
    # If they can all be run together then there's no concept of a
    # "current_loop". current_loop therefore only applies to [train, val, test]
    @prop                       # type: ignore
    @property
    def adhoc_aborted(self) -> bool:
        "Was the adhoc loop aborted?"
        return self._adhoc_aborted_event.is_set()

    @prop                       # type: ignore
    @property
    def userfunc_paused(self) -> bool:
        "Is a given userfunc paused?"
        return not self._userfunc_running_event.is_set()

    @prop                       # type: ignore
    @property
    def userfunc_aborted(self) -> bool:
        "Was a given userfunc aborted?"
        return self._userfunc_aborted_event.is_set()
    # END: State props

    # START: Path props
    @prop                       # type: ignore
    @property
    def best_save(self) -> Optional[pathlib.Path]:
        """The path of the best save recorded. Probably should not be available to the client"""
        if not os.path.exists(self._savedir) or not os.listdir(self._savedir):
            return None
        else:
            save_files = os.listdir(self._savedir)
            if "checkpoint.pth" in save_files:
                save_files.remove("checkpoint.pth")
            if save_files:
                results = []
                for f in save_files:
                    result = re.search("val_acc_......", f)
                    if result:
                        results.append((f, result.group()))
                if results:
                    results.sort(key=lambda x: x[1])
                    return os.path.join(self._savedir, results[-1][0])
                else:
                    return None
            else:
                return None

    @property
    def _save_path_with_epoch(self):
        given_name = getattr(self, "given_name", "")
        if "iterations" in self.trainer_params.training_steps:
            update_key = self.iterations / self._hooks_run_iter_frequency
        else:
            update_key = self.epoch
        model_names = (given_name and (given_name + "_")) + "_".join(self.models)
        save_name = os.path.join(self._savedir, "_".join([model_names,
                                                          "{:03}".format(update_key)]))
        return save_name

    @property
    def _save_path_without_epoch(self):
        given_name = getattr(self, "given_name", "")
        model_names = (given_name and (given_name + "_")) + "_".join(self.models)
        save_name = os.path.join(self._savedir, model_names)
        return save_name

    @property
    def _checkpoint_path(self):
        # model_names = "_".join(self._models.names)
        return os.path.join(self._save_path_without_epoch + "_checkpoint" + ".pth")
    # END: Path props

    # START: Params
    @prop                       # type: ignore
    @property
    def trainer_params(self) -> config.TrainerParams:
        """Trainer params govern the behaviour of the trainer.

        Trainer Params must be of type :class:`~config.TrainerParams`

        - `training_steps` can only be `['train', 'val', 'test', 'iterations']` as
          of now.
        - if `iterations` is present in `training_steps` then the trainer will
          not progress via epochs but via number of batches. `epoch` and
          `max_epochs` in that case are 0. In this case `max_iterations` must
          be provided and > 0
        - Otherwise `epoch` and `max_epoch` must be given.
        - A `condition` can also be added instead of `max_epochs` or
          `max_iterations` but isn't implemented right now.
        - Both `cuda` and `gpus` have to be given. To understand how they
          operate see :meth:`_init_device`.

        """
        return self.config.trainer_params

    @prop                       # type: ignore
    @property
    def log_levels(self) -> config.LogLevelParams:
        "File and stream Log levels for the trainer"
        return self.config.log_levels

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def metrics(self) -> Dict[str, Dict]:
        "All the metrics used for evaluation"
        return self._metrics

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def extra_metrics(self) -> Dict[str, Metric]:
        "Extra metrics used for evaluation"
        return self.config.extra_metrics

    @prop                       # type: ignore
    @property
    def optimizer_params(self) -> Dict[str, config.Optimizer]:
        return self.config.optimizer_params

    @prop                       # type: ignore
    @property
    def model_params(self) -> Dict[str, config.ModelParams]:
        """Parameters with which to initialize the models.

        They must be a :class:`dict` mapping model names and :class:`config.ModelParams`

        These parameters are specific to each model and can be changed during
        the training procedure. Depending on the parameter changes, the model
        may or may not be able to resume trainming from that point. For example,
        a parameter which changes the number of layers or the dimension of one
        of the layers would render the model incompatible with the previously
        trained model. The new model in that case will have to be patched by
        hand, and the weights copied from the previous model.

        In addition they can contain a `devices` parameter also which is a
        directive to the trainer assigning the specified devices to the
        models. An `auto` field for the devices will find the optimal
        allocation w.r.t training speed.

        Multiple models can be specified and not all of which need to be loaded
        on to the devices. Ones which are to be loaded should be specified with
        {"loaded": True} in their parameters.

        """
        return self.config.model_params

    @prop                       # type: ignore
    @property
    def criteria_params(self) -> Dict[str, config.Criterion]:
        """Names of the criteria and their paramters

        Can be used to initialize the criteria if required.

        """
        return self.config.criteria

    @prop                       # type: ignore
    @property
    def dataloader_params(self) -> config.DataLoaderParams:
        """Parameters passed to :class:`~torch.utils.data.Dataloader`.
        They must be acceptable to :class:`~torch.utils.data.Dataloader`

        For :class:`~config.CustomDataLoader` the parameters are defined in the
        `~config.CustomDataLoader` and they are unrestricted.

        """
        return self.config.dataloader_params

    @prop                       # type: ignore
    @property
    def data_params(self) -> config.DataParams:
        return self.config.data_params

    @prop                       # type: ignore
    @property
    def update_functions(self) -> config.UpdateFunctions:
        return self.config.update_functions

    # FIXME: NEW Now a whole bunch of things can be sent anyway. Updatable
    #        params should be dynamicall determined
    # TODO: Allow extra_metrics, update_funcs and any other params to be updated
    @prop                       # type: ignore
    @property
    def updatable_params(self) -> Dict[str, Dict]:
        """All the parameters which can be updated by the user midway.

        We segregate them into ``[model_params, trainer_params,
        dataloader_params]``

        """
        params = {}
        params["model_params"] = self.model_params
        params["trainer_params"] = self.trainer_params
        params["dataloader_params"] = self.dataloader_params
        return params

    @prop                       # type: ignore
    @property
    def all_params(self) -> Dict[str, Any]:
        """All params is the entire config which can be serialized as JSON. It contains
        [epoch, iterations, model_params, criteria_params, dataloader_params,
        trainer_params, metrics]"""
        save_state = {}
        save_state["epoch"] = self.epoch
        save_state["iterations"] = self._iterations
        # save_state["models"] = dict((k, v.state_dict()) for k, v in self._models.items())
        # save_state["optimizers"] = dict((k, v.state_dict()) for k, v in self.optimizers.items())
        save_state["model_params"] = self.model_params
        save_state["criteria_params"] = self.criteria_params
        save_state["dataloader_params"] = self.dataloader_params
        save_state["trainer_params"] = self.trainer_params
        save_state["metrics"] = self._metrics
        # return _dump(save_state)
        return save_state
    # END: Params

    # @prop
    # @property
    # def all_attrs(self):
    #     """Full __dict__ serialized as json. The parts which can't be serialized are
    #     left out."""
    #     return self.__dict__

    # TODO: What about other losses
    #       `prop` can help
    @prop                       # type: ignore
    @property
    def train_losses(self) -> Dict[str, Any]:
        """History of training losses"""
        return dict((k, v) for k, v in self._metrics["train"].items()
                    if k[0] == "loss")

    @prop                       # type: ignore
    @property
    def progress(self) -> Dict[str, int]:
        """Progress returns the number of steps and max steps of training (whether
        epochs or iterations) and also the round of current batch being processed"""
        predicate = "iterations" in self.trainer_params.training_steps
        cur_step = self.iterations / self._hooks_run_iter_frequency\
            if predicate else self.epoch
        max_step = self.max_iterations / self._hooks_run_iter_frequency\
            if predicate else self.max_epochs
        cur_round = self._epoch_runner.info["batch_nums"]["train"]
        max_round = self._hooks_run_iter_frequency if predicate else len(self.train_loader)
        return {"cur_step": cur_step, "max_step": max_step,
                "cur_round": cur_round, "max_round": max_round}

    # FIXME: self.user_funcs MAY create problems
    @prop                       # type: ignore
    @property
    def user_funcs(self) -> List[str]:
        """Names of all the functions uploaded by the user"""
        return [x for x in self._user_funcs]

    # @prop                       # type: ignore
    @property
    def _current_user_func(self):
        """User function currently active. This function will be run if `run_user_func`
        is called."""
        if self._current_user_func_name and\
           self._current_user_func_params:
            return partial(self._user_funcs[self._current_user_func_name],
                           kwargs=self._current_user_func_params)
        else:
            return lambda: None

    # TODO: Define what is a sample correctly
    @prop                       # type: ignore
    @property
    def val_samples(self) -> Dict[str, Dict]:
        """Metric for model output if run on a small (possibly random) subset of
        validation data."""
        return dict((k, v) for k, v in self._metrics["val"].items()
                    if k[0] == "sample")

    # NOTE: Not sure if I want to use dir(self)
    @prop                       # type: ignore
    @property
    def all_post_epoch_hooks(self) -> Dict[str, Any]:
        """All the post epoch hooks present. All of them may not be called. See
        `post_epoch_hooks_before`."""
        dict_a = dict((x, y) for (x, y) in self.__class__.__dict__.items()
                      if x.endswith("post_epoch_hook") and
                      callable(y) and
                      x != "add_post_epoch_hook" and
                      x != "remove_post_epoch_hook")
        dict_b = dict((x, y) for (x, y) in self.__dict__.items()
                      if x.endswith("post_epoch_hook") and
                      callable(y) and
                      x != "add_post_epoch_hook" and
                      x != "remove_post_epoch_hook")
        return {**dict_a, **dict_b}

    @prop                       # type: ignore
    @property
    def post_epoch_hooks_to_run(self) -> Dict[str, Callable]:
        """All the post epoch hooks which will be called at the end of an epoch."""
        return self._post_epoch_hooks_to_run

    @prop                       # type: ignore
    @property
    def items_to_log_dict(self) -> Dict[str, Any]:
        """Which of the items will be logged."""
        return self._items_to_log_dict

    # CHECK: Should this be a property?
    def docs(self) -> Dict[str, Dict[str, str]]:
        """Return all doc strings which are available for any functionality in the
        server. Currently only returns for `props`.

        """
        return {"props": {x: self.__class__.__dict__[x].__doc__ for x in self.props}}
    # END: Properties

    # START: Broken funcs
    # FIXME: THIS IS totally broken :-(
    # TODO: params should only be predefined names. As such, the required python
    #       objects must already be available to the wrapper. The search
    #       protocol can be developed later.
    # TODO: Implies reset
    def try_update(self, params):
        """Update the trainer w.r.t to any of the possible variables.  `params` must be
        a dict where the keys must be the same as the keys returned by
        updatable_params.

        As of now non-serializable updates are not supported, so the both
        key/value pairs must be json-encodeable.

        :param params: :class:`dict`
        :returns: None
        :rtype: None

        """
        self._logi("Trying to update")
        if not all(k in self.updatable_params for k in params):
            return False
        valid_updates = []
        invalid_updates = []
        for param in params:
            for k in param:
                if k in self.updatable_params[param]:
                    if type(self.updatable_params[param][k]) in [str, int, float]:
                        pass
                else:
                    pass
        assert all(k in ["model_params", "model_defs", "criteria", "optimizer",
                         "update_funcs", "dataloader_params", "trainer_params",
                         "extra_metrics"] for k in params)
        raise NotImplementedError
        self.lr = params.lr
        self.momentum = params.momentum
        self.batch_size = params.batch_size
        self._optimizer_name = params.optimizer
        self._criterion_name = params.criterion
        self.max_epochs = params._max_epochs
        self.init_weights = params.init_weights
        if not os.path.exists(self._savedir):
            os.mkdir(self._savedir)
        # FIXME: BAD
        # self._gpus = trainer_params.gpus
        # self._cuda = trainer_params.cuda
        self._init_device()
        # if torch.cuda.is_available():
        #     self._device = torch.device("cuda:%d" % params.gpu)
        self.save_on = params.save_on
        self.save_var = params.save_var

    # # FIXME: Optimizer is initialized here or before it?
    # #        It can possibly be based on name or a custom func
    # # TODO: This thing doesn't do anything now
    # def _get_optimizer(self, name):
    #     if name.lower() == 'sgd':
    #         return torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
    #                                momentum=self.args.momentum)
    #     elif name.lower() == 'adam':
    #         return torch.optim.Adam(self.model.parameters())
    # END: Broken funcs

    # START: Traninig Steps
    # Train validate and stuff are relatively fine
    # TODO: custom reportables
    # TODO: Things should get updated in a shared queue after each batch
    # NOTE: Maybe not really required, as only that thread writes to those
    #       variablesthingies.
    # TODO: What if run has to be aborted in the middle?
    #       Ensure that run returns
    # NOTE: train ONLY runs in main loop
    def train(self):
        """If `iterations` exists in self.trainer_params, then we do iterations only
        training and loop_type is set to iterations, else we do standard epoch
        wise training

        if `self._abort_current` flag is set then current main loop is
        aborted. The same flag is also watched for in `self._epoch_runner`.

        post_epoch_hooks are NOT run after an abort.

        :returns: None
        :rtype: None

        """
        loop_type = self.loop_type
        self._logd(f"Beginning training. Loop type is {loop_type}.")
        if loop_type == "iterations":
            self._logd(f"Total number of iterations is {self.max_iterations}")
            self._logd(f"Will run hooks after {self._hooks_run_iter_frequency} iterations")
            while self.iterations < self.max_iterations:
                self._epoch_runner.reset()
                # NOTE: run for self._hooks_run_iter_frequency
                self._epoch_runner.run_train(self._training_steps["train"][0], self.train_loader,
                                             loop_type, self._hooks_run_iter_frequency)
                if not self._epoch_runner.status[0]:
                    self._loge(f"Error in train loop {self._epoch_runner.status[1]}")
                    # TODO: the thread should join on crash
                    # import ipdb; ipdb.set_trace()
                    self._abort_current(self._epoch_runner.status[1])
                    # print("THIS SHOULD be set", self._current_aborted)
                if self._current_aborted:
                    self._logd("Aborting training")
                    # import ipdb; ipdb.set_trace()
                    return
                self._run_post_epoch_hooks()
                self._iterations += self._hooks_run_iter_frequency
        else:
            self._logd(f"Total number of batches is {len(self.train_loader)}")
            while self.epoch < self.max_epochs:
                self._epoch_runner.reset()
                self._epoch_runner.run_train(self._training_steps["train"][0], self.train_loader,
                                             loop_type)
                if not self._epoch_runner.status[0]:
                    self._loge(f"Error in train loop {self._epoch_runner.status[1]}")
                    # TODO: the thread should join on crash
                    # import ipdb; ipdb.set_trace()
                    self._abort_current(self._epoch_runner.status[1])
                    # print("THIS SHOULD be set", self._current_aborted)
                if self._current_aborted:
                    self._logd("Aborting training")
                    return
                self._run_post_epoch_hooks()
                self.epoch += 1
        self._logi('finished training')

    def validate(self, runner):
        self._logd(f"Validating with {runner.name}")
        try:
            runner.run_val(self._training_steps["val"][0], self.val_loader)
            if not self._epoch_runner.status[0]:
                self._loge(f"Error in val loop {self._epoch_runner.status[1]}")
                self._abort_current(self._epoch_runner.status[1])
        except Exception as e:
            self._loge(f"Some weird error occured {e}\n{traceback.format_exc()}")

    def test(self, runner):
        self._logd(f"Testing with {runner.name}")
        try:
            runner.run_test(self._training_steps["test"][0], self.test_loader)
            if not self._epoch_runner.status[0]:
                self._loge(f"Error in val loop {self._epoch_runner.status[1]}")
                self._abort_current(self._epoch_runner.status[1])
        except Exception as e:
            self._loge(f"Some weird error occured {e}\n{traceback.format_exc()}")

    # END: Training Steps
    def add_post_epoch_hook(self, hook: Callable[[], None],
                            name: str, position: Union[int, str],
                            overwrite: bool = False):
        if not hasattr(self, "_post_epoch_hooks_to_run"):
            return False, "Cannot add hook without initializing"
        if position not in {"first", "last"} and not isinstance(position, int):
            return False, "Invalid position"
        if hasattr(self, name):
            if overwrite:
                setattr(self, name, hook)
            else:
                return False, "Hook already exists. Use 'overwrite=True' to overwrite"
        else:
            setattr(self, name, hook)
        hook_name = name.replace("_post_epoch_hook", "")
        if hook_name in self.post_epoch_hooks_to_run:
            self.post_epoch_hooks_to_run.remove(hook_name)
        if position == "first":
            self._post_epoch_hooks_to_run.insert_top(name.replace("_post_epoch_hook", ""))
        elif position == "last":
            self._post_epoch_hooks_to_run.append(name.replace("_post_epoch_hook", ""))
        else:
            self._post_epoch_hooks_to_run.insert(position, name.replace("_post_epoch_hook", ""))

    def _run_post_epoch_hooks(self):
        self._logd("Running post epoch hooks")
        all_hooks = self.all_post_epoch_hooks
        hook_prefixes = self.post_epoch_hooks_to_run
        for hook in hook_prefixes:
            all_hooks["_".join([hook, "post_epoch_hook"])](self)

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def hooks(self) -> List[str]:
        return [*self._hooks.keys()]

    @prop                       # type: ignore
    @state_var                  # type: ignore
    @property
    def hooks_with_args(self) -> List[str]:
        return [*self._hooks_with_args.keys()]

    def run_hook(self, hook_name) -> Union[Return, ReturnExtraInfo]:
        if hook_name in self.hooks:
            retval = self._hooks[hook_name](self)
            return make_info(True, f"Ran hook {hook_name}", retval)
        else:
            return make_return(False, f"Hook {hook_name} not found")

    def run_hook_with_args(self, hook_name, *args, **kwargs) -> Union[Return, ReturnExtraInfo]:
        if hook_name in self.hooks_with_args:
            retval = self._hooks_with_args[hook_name](self, *args, **kwargs)
            return make_info(True, f"Ran hook {hook_name}", retval)
        else:
            return make_return(False, f"Hook {hook_name} not found")
    # END: Stateless Functions
