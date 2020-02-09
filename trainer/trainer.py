import re
import io
import os
import base64
import copy
import time
import json
import torch
import inspect
from functools import partial
from threading import Thread, Event
from PIL import Image
import numpy as np
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader

from .device import init_nvml, gpu_util, cpu_info, memory_info, DeviceMonitor
from .util import get_backup_num, gen_file_and_stream_logger, deprecated
from .epoch import Epoch
from .mods import Modules as Modules
from .state_machine import StateMachine
from .overrides import MyDataLoader
from .components import Models
from .functions import _log_metrics_for_step
from ._log import Log
from .checks import Checks
from ._checks import (_check_model_params, _check_trainer_params, _check_data_params,
                      _check_resume_or_init_weights)
from .helpers import (control, prop, extras, helpers, ProxyDataset, get_proxy_dataloader,
                      PropertyProxy, HookDict, HookList, GET, POST, Exposes)
from .version import __version__


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

# TODO: autoprop decorator, expand name to different properties
#       based on their name, remove leading "_"
#       e.g., autoprop(self._training_step) becomes
#       @property
#       def training_step(self):
#           return self._training_step


class Trainer:
    """The :class:`Trainer` class is envisioned as an interface to any training
    procedure. As of now it's not model agnostic and assumes :mod:`torch` is the
    underlying backend. But the principles on which it's designed are universal
    and any backend should be feasible.

    """
    __version__ = __version__

    def __init__(self, uid, model_params, criteria, optimizer, model_defs, update_functions,
                 extra_metrics, trainer_params, data, dataloader_params):
        """Initializes the :class:`Trainer` object. This is supposed to be a catch all
        trainer which is robust and easy to train and can generate graphs
        automatically etc.

        `model_params`, `criteria`, `trainer_params`, `dataloader_params` are
        stateless parameters.

        `model_defs`, `optimizers`, `update_functions` contain callables and as
        such aren't part of config but model and training definitions.

        `uid` simply identifies the trainer and should remain the same
        throughout the trainer's life.

        :param model: model which is a :class:`torch.nn.Module`
        :param model_params: model params where (k, v) are (:class:`str` model_name,
        `list` of model params) :class:`dict`
        :param criteria: `dict` where (k, v) are (`str`, :class:`torch.nn.Module`)
        :param optimizer: `dict` where (k, v) are (`str`, :class:`torch.optim.Optimizer`)
        :param model_init: `dict` where (k, v) are (`str` model_name, :function: returns the initialized model)
        :param train_step_func: :function: which is called for running each batch forward iteration
        :param trainer_params: TODO
        :param train_loader: a train data loader usually :class:`torch.utils.data.Dataloader`
        :param val_loader: a validation data loader usually :class:`torch.utils.data.Dataloader`
        :param test_loader: a test data loader usually :class:`torch.utils.data.Dataloader`
        :param args: `types.SimpleNamespace` which contains the rest of the arguments

        """
        # DONE: model, train_loader, val_loader should be resettable from the interface
        #       Say, trainer.reset() is called, then the interface should place a hook there
        #       that automatically resets the trainloader and the valloader
        #       Mostly Done.
        # Basic assign parameters

        # __props is initialized early and anything that's to be exposed has to be
        # appended to the list. _init_property_vars should check if everything in __props
        # is a property or not
        self.__props = set()
        self._unique_id = uid
        self.__props.add("unique_id")
        self._model_params = model_params
        self._model_defs = model_defs
        self._criteria_params = criteria
        self._optimizer_params = optimizer
        self._data = data
        self._dataloader_params = dataloader_params
        self._trainer_params = trainer_params
        self._extra_metrics = extra_metrics
        self._update_functions = update_functions
        # static attributes
        self._savedir = ".savedir"
        self._logdir = ".logs"
        if not os.path.exists(self._savedir):
            os.mkdir(self._savedir)
        if not os.path.exists(self._logdir):
            os.mkdir(self._logdir)
        self._logfile, self._logger = gen_file_and_stream_logger(
            self._logdir, "_".join(["trainer", self._unique_id]), "debug", "debug")
        log = Log(self._logger)
        self._logd = log._logd
        self._loge = log._loge
        self._logi = log._logi
        self._logw = log._logw
        self._logi(f"Initialized logger in {os.path.abspath(self._logdir)}")
        self._logi(f"Savedir is {os.path.abspath(self._savedir)}")
        # check all params here
        self._have_resumed = False
        self._sanity_check()
        self._init_static_vars()
        self._init_property_vars()
        # self._init_external_vars()
        self._check_exports()
        if trainer_params["resume"] or "init_weights" in trainer_params:
            self._init_models()
            _check_resume_or_init_weights(self)

    def init(self, force=False):
        if self._have_resumed and not force:
            self._logw("\"init\" cannot be called after resume. Use \"force\"")
            self._init_all()
        elif self._have_resumed and force:
            self._logw("forcing \"init\" call after resume")
            self._init_all()
        else:
            self._init_all()

    def _init_all(self):
        self._logi("Initializing trainer")
        self._init_models()
        self._init_nvml()
        self._init_dataloaders()
        # self._init_criteria_optimizers()
        self._init_metrics()
        self._init_update_funcs()
        self._init_state_vars()
        self._init_epoch_runner()
        self._init_modules()
        # self._init_extra_controls()

    # NOTE: Shouldn't export this to Checks as name will be mangled
    def _check_exports(self):
        """Checks the API as exported endpoints.

        All the properties not beginning with _ are exported except _extras and
        _helpers.

        Controls and other export checks are to be added.

        :returns: None
        :rtype: None

        """
        attrs = [*self.__class__.__dict__.keys()]
        assert all(x in attrs for x in self.__props), "Some properties not correctly exported"

    # START: Checks
    def _sanity_check(self):
        """Checks are imported from the file _checks.py

        :returns: None
        :rtype: None

        """
        self._logi("Performing Sanity Check")
        _check_model_params(self)   # checks model params and defs both
        _check_trainer_params(self)  # checks optimizer and stuff also
        _check_data_params(self)     # checks data and dataloaders
    # END: Checks

    # START: Init Funcs
    def _init_device(self):
        gpus_str = [x for x in self._trainer_params["gpus"].split(",") if x]
        if gpus_str:
            self._gpus = list(map(int, gpus_str))
        else:
            self._gpus = [-1]
        has_cuda = torch.cuda.is_available()
        gpus_given = self._gpus and (not self._gpus == [-1])
        cuda_given = self._trainer_params["cuda"]
        if not gpus_given:
            self._logd("No gpus given. Will run on cpu")
            self._device = torch.device("cpu")
            self._gpus = [-1]
        if cuda_given and not has_cuda:
            self._logw("cuda specified but not available. Will run on cpu")
            self._device = torch.device("cpu")
            self._gpus = [-1]
        elif gpus_given and not cuda_given:
            self._logw("cuda not specified but gpus given. Will run on cpu")
            self._device = torch.device("cpu")
            self._gpus = [-1]
        elif cuda_given and has_cuda and len(self._gpus) == 1:
            self._logi(f"GPU {self._gpus[0]} detected and specified")
            self._device = torch.device(f"cuda:{self._gpus[0]}")
        elif cuda_given and has_cuda and len(self._gpus) > 1:
            self._logi(f"Data parallel specified with gpus {self._gpus}")
            if torch.cuda.device_count() >= len(self._gpus):
                self._logi(f"{torch.cuda.device_count()} gpus are available")
                if "parallel" in self._trainer_params:
                    # I always get confused by this statement
                    # It's somewhhat mirthful and one has to see the next line
                    # to make sense of it.
                    self._logi(f"Parallel call be functional {self._trainer_params['parallel']}")
                    self._device = self._trainer_params["parallel"]
                else:
                    self._logi("Parallel call be Module dataparallel")
                    self._device = "dataparallel"
            else:
                self._loge(f"{torch.cuda.device_count()} gpus are not available")
                raise AttributeError
        else:
            self._logi("cuda not specified. Using cpu")
            self._device = torch.device("cpu")
            self._gpus = [-1]
        torch.cuda.manual_seed(self._trainer_params["seed"])
        for t, v in self._trainer_params.items():
            if t in self.__class__.__dict__:
                self._logw(f"Tried overwriting attribute {t}! Denied.")
            elif t != "gpus":
                self.__dict__[t] = v

    def _init_static_vars(self):
        self.adhoc_error_dict = {"required_oneof_[function]": ["train", "val", "test", "user_func_name"],
                                 "required_for_[function]": {"epoch": "[int|string]_which_epoch",
                                                             "data": "[string]_train_val_or_test",
                                                             "num_or_fraction":
                                                             "[int|float]_number_of_points_or_fraction_of_dataset",
                                                             "callback": "[string]_name_of_callback_function"}}

    def _init_state_vars(self):
        """Initialize default state variables.
        `epoch` always remains 0 if training only with iterations and
        `self._iterations` increase.
        
        post_epoch_hooks are run after a specified number of iterations which is
        `self._hooks_run_iter_frequency`

        
        :returns: None
        :rtype: None

        """
        # params and state properties
        self.__props.add("saves")
        self.__props.add("gpus")
        self.__props.add("system_info")
        self.__props.add("device")
        self.__props.add("models")
        self.__props.add("active_model")
        self.__props.add("epoch")
        self.__props.add("max_epochs")
        self.__props.add("iterations")
        self.__props.add("max_iterations")
        self.__props.add("updatable_params")
        self.__props.add("all_attrs")
        self.__props.add("all_params")
        self.__props.add("metrics")
        self.__props.add("post_epoch_hooks_to_run")
        self.__props.add("all_post_epoch_hooks")
        self.__props.add("items_to_log_dict")  # CHECK: WTF is this?

        # running status
        # self.__props.add("aborted")
        self.__props.add("current_run")
        self.__props.add("paused")
        self.__props.add("best_save")  # FIXME: Really?

        # Other exposed API
        self.__props.add("props")
        self.__props.add("controls")
        self.__props.add("_helpers")
        self.__props.add("_extras")

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
        self._post_epoch_hooks_to_run = HookList(["validate", "test", "update_metrics"])

        # FIXME: validate, test, update_metrics is mandatory for now,
        #        unless val_loader and test_loader are none of course
        self._post_epoch_hooks_to_run.append("save_history")
        self._post_epoch_hooks_to_run.append("save_best")
        self._post_epoch_hooks_to_run.append("save_checkpoint")
        self._post_epoch_hooks_to_run.append("log")

        # NOTE: _log_metrics is a function so "metrics" defines a way to log it
        #       rather than just copying the values.
        self._items_to_log_dict = {"metrics": self._log_metrics}
        self._init_device()
        self._epoch = 0
        self._iterations = 0
        self._init_nvml()
        self._temp_runner = SimpleNamespace()
        steps = self._trainer_params["training_steps"]
        if "iterations" in steps:
            # NOTE: one loop but three possible things can be running
            #       adhoc has two loops right now
            self._transition_steps = {"main": {"train", "val", "test"},
                                      "adhoc": {"val", "test"},
                                      "user": None}
        else:
            self._transition_steps = {"main": set(steps).union({"none"}),
                                      "adhoc": {"val", "test"},
                                      "user": None}

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
        if "extra_report" not in self._trainer_params:
            self._logd("No Extra Reportables")
            self.extra_report = {}
        self._user_funcs = {}
        self._current_user_func_name = None
        self._current_user_func_params = None
        self.__props.add("user_funcs")
        self.load_weights.__dict__["content_type"] = "form"
        self.add_model.__dict__["content_type"] = "form"
        self.add_user_funcs.__dict__["content_type"] = "form"
        self.load_image.__dict__["content_type"] = "form"

    def _init_modules(self):
        self._mods = Modules("trainer_modules", self._logd, self._loge, self._logi, self._logw)
        self._sm = StateMachine(3, self._transition_steps, self._forced_states, logd=self._logd,
                                loge=self._loge, logi=self._logi, logw=self._logw)

    @deprecated
    def _init_external_vars(self):
        """Initialize some variables which will be attached to it later. Right now a
        hack but it could be more principled later.

        :returns: None
        :rtype: None

        """
        if not hasattr(self, "report_function"):
            self.report_function = None

    def _init_nvml(self):
        """Initializes the Nvidia monitoring library. It's called by _init_state_vars so
        needn't be called again.

        :returns: None
        :rtype: None

        """
        self._logi("Initializing nvml")
        # CHECK: I don't remember how the order is printed.
        # Assumes torch.cuda devices are of the same order as PCI BUS for
        # getting correct info with pynvml
        if self._gpus[0] != -1:
            self._device_handles = init_nvml(self._gpus)
        else:
            self._device_handles = None

    def _init_models(self):
        self._logi("Initializing Models, Optimizers and Criteria ")
        self.criteria = {}
        for k, v in self._criteria_params.items():
            self.criteria[k] = v["function"](**v["params"])
        models = {}
        optimizers = {}
        devices = {}
        if (not hasattr(self, "devices") or not hasattr(self, "_devices"))\
           and hasattr(self, "_device"):
            devices = {m: self._device for m in self._model_params}
        else:
            # TODO: Model parallel and sharding
            self._init_device()
            devices = {m: self._device for m in self._model_params}
        for model_name, model_params in self._model_params.items():
            models[model_name] = self._model_defs[model_name]["model"](**model_params)
            optim_name = self._model_defs[model_name]["optimizer"]
            optimizers[model_name] = {"name": optim_name,
                                      "optimizer": self._optimizer_params[optim_name]["function"](
                                          models[model_name].parameters(),
                                          **self._optimizer_params[optim_name]["params"])}
        self._models = Models(models, optimizers, devices, self.gpus, self.logger)

    def _init_dataloaders(self):
        """Dataloaders can be initialized from a {step, data, params} in which the
        corresponding torch dataloader is initialized with the given data. Or
        from a function like `get_dataloader` with certain parameters. In case
        `train data` is available, then train_step has to be
        available. Arbitrary custom named steps aren't supported as of now.

        :returns: None
        :rtype: None

        """

        self._logi("Initializing Dataloaders")

        # FIXME: Remove this and streamline data and loaders
        #        Maybe initialize dataloaders based on update_funcs? Not sure
        def _check_raw(loader, name):
            if loader and not hasattr(loader, "dataset"):
                self._logw(name + " loader doesn't have a dataset")
            elif loader and not hasattr(loader.dataset, "_get_raw"):
                self._logw(name + " dataset doesn't define \"_get_raw\"" +
                           " Drawing samples from validation data will not be available.")
            elif loader and hasattr(loader.dataset, "_get_raw"):
                self._logw(name + " dataset has \"_get_raw\"" +
                           " Drawing samples from validation data is available!")

        for loader, params in self._dataloader_params.items():
            if loader == "train":
                if self._data is None:
                    self.train_loader = params["function"](**params["function_args"])
                else:
                    self.train_loader = DataLoader(self._data["train"], **params)
                _check_raw(self.train_loader, "Train")
            elif loader == "val":
                if params:
                    if self._data is None:
                        self.val_loader = params["function"](**params["function_args"])
                    else:
                        self.val_loader = DataLoader(self._data["val"], **params)
                else:
                    self._logi("No Val loader. Will not do validation")
                    self.val_loader = None
                _check_raw(self.val_loader, "Val")
            elif loader == "test":
                if params:
                    if self._data is None:
                        self.test_loader = params["function"](**params["function_args"])
                    else:
                        self.test_loader = DataLoader(self._data["test"], **params)
                else:
                    self._logi("No Test loader. Will not do testing")
                    self.test_loader = None
                _check_raw(self.test_loader, "Test")

    # CHECK: Should I use namedtuple instead?
    def _init_metrics(self):
        """Intializes and checks the metrics.

        Anything returned by the `step` having the first element `metric` is a
        default metric and is logged.

        Other metrics are specified in `extra_metrics` have to conform to the format
        {\"step\": step_name, \"function\": `callable`,
        \"inputs\": input_variables_to_function, \"when\": batch_or_epoch}

        :returns: None
        :rtype: None

        """
        self._logi("Initializing Metrics")
        self._metrics = {}
        for x in self._update_functions:
            if self._dataloader_params[x] is not None:
                self._metrics[x] = dict((l[1], {}) for l in self._update_functions[x].returns
                                        if l[0] == "metric")
                self._metrics[x]["num_datapoints"] = {}
                if self._extra_metrics is None:
                    self._extra_metrics = {}
                if x in self._extra_metrics:
                    for k in self._extra_metrics[x].keys():
                        self._metrics[x][k] = {}
                        if self._extra_metrics[x][k]["when"] == "batch":
                            retvals = [_[1] for _ in self._update_functions[x].returns]
                            assert all(_ in retvals for _ in self._extra_metrics[x][k]["inputs"]),\
                                f"failed on batch {x}, {k}"
                        elif self._extra_metrics[x][k]["when"] == "epoch":
                            # NOTE: validation with inputs
                            vals = [*self.__dict__.keys(),
                                    *[_[1] for _ in self._update_functions[x].returns], "epoch"]
                            assert all(s in vals for s in self._extra_metrics[x][k]["inputs"]
                                       if isinstance(s, str)), f"failed on epoch {x}, {k}"
                            # FIXME: Only "models" as tuple allowed
                            for _x in self._extra_metrics[x][k]["inputs"]:
                                if isinstance(_x, tuple):
                                    assert _x[0] == "models" and\
                                        all(_ in self._models.names for _ in _x[1]),\
                                        "Required model not in self._models"
                            # assert all(all(_d in self.__dict__[d[0]].keys() for _d in d[1])
                            #            for d in self._extra_metrics[x][k]["inputs"]
                            #            if isinstance(d, tuple)), "failed on tuple %s, %s" % (x, k)
                else:
                    self._extra_metrics[x] = {}

    def _init_update_funcs(self):
        self._logi("Initializing Update Functions")
        for k, v in self._update_functions.items():
            if k == "train":
                self._train_step_func = self._update_functions["train"]
            elif k == "val":
                self._val_step_func = self._update_functions["val"]
            elif k == "test":
                self._test_step_func = self._update_functions["test"]

    def _task_runner_helper(self, which):
        device_monitor = DeviceMonitor(self._device_handles)
        signals = SimpleNamespace()
        if which == "main":
            signals.paused = self._running_event
            signals.aborted = lambda: self._current_aborted_event.is_set()
        elif which == "adhoc":
            signals.paused = self._adhoc_running_event
            signals.aborted = lambda: self._adhoc_aborted_event.is_set()
        elif which == "user":
            signals.paused = self._userfunc_running_event
            signals.aborted = lambda: self._userfunc_aborted_event.is_set()
        return device_monitor, signals

    def _init_epoch_runner(self):
        """The task_runners and threads are a triple
        As of now there are three kinds of task runners There can be:
          1. epoch_runner
          2. adhoc_runner
          3. user_func_runner

        Corresponding to each task runner there are threads and with the same
        names and the references are further stored in `Trainer._task_thread_keys`,
        `_tasks_callbacks`

        :returns: None
        :rtype: None

        """
        device_monitor, signals = self._task_runner_helper("main")
        self._logi("Initializing Epoch Runner")
        self._epoch_runner = Epoch({"metrics": self._metrics, "extra_metrics": self._extra_metrics},
                                   signals, device_monitor, self.extra_report)
        self._epoch_runner.name = "epoch_runner"
        self._task_runners = {"epoch": self._epoch_runner,
                              "train": self._epoch_runner,
                              "adhoc": None,
                              "user": None}
        self._task_thread_keys = {"epoch": "main",
                                  "train": "main",
                                  "adhoc": "adhoc",
                                  "user": "user"}
        # FIXME: For val and test maybe update in separate variables
        self._tasks_callbacks = {"epoch": self._run_post_epoch_hooks,
                                 "train": self._run_post_epoch_hooks,
                                 "adhoc": None,
                                 "user": None}

    def _task_runner_initialize(self, name, metrics, extra_metrics, callback):
        """Initializes a task_runner. The quintessential task runner is the epoch
        runner. Task runner coordinates with the trainer to ensure that nothing
        gets jammed up.

        Task runner needs signals to communicate with the trainer and those are
        created by the helper as needed and recreated when another task_runner is
        initialized.

        :param name: Name :class:`str` of the task runner.
        :param metrics: 
        :param extra_metrics: 
        :param callback: 
        :returns: 
        :rtype:

        """
        assert name in self._task_runners
        device_monitor, signals = self._task_runner_helper(name)
        self._task_runners[name] = Epoch({"metrics": metrics,
                                          "extra_metrics": extra_metrics},
                                         signals, device_monitor, self.extra_report)
        self._task_runners[name].reset()
        self._task_runners[name].logger = self.logger
        self._tasks_callbacks[name] = callback
    # END: Init Funcs

    # START: Internal Controls
    #        These functions interact with the SM.
    #        NOTE: As of now this is only the main loop
    def _finish_if_paused_or_running(self, _force=False, gather=False):
        """Should not be called from `self._transition`
        Stops the current running main flow (or alternate flow?)

        :returns: None
        :rtype: None

        """
        self._transition_flags = {}
        if gather:
            self._transition_flags["run_cb"] = True
        force, run, step = self.current_state.split("_")
        if _force:
            force = "force"
        if run == "running":
            self._transition(self.current_state, "_".join([force, "paused", step]))
        self._transition(self.current_state, "_".join([force, "finished", step]))

    def _pause_if_running(self, _force=False):
        """Should not be called from `self._transition`

        :returns: None
        :rtype: None

        """
        force, run, step = self.current_state.split("_")
        if _force:
            force = "force"
        if run == "running":
            self._transition(self.current_state, "_".join([force, "paused", step]))

    def _start_if_not_running(self, _force=False):
        """Should not be called from `self._transition`

        :returns: None
        :rtype: None

        """
        force, run, step = self.current_state.split("_")
        if step != "none":
            return              # only if step == "none"
        else:
            step = "train"      # default
        if _force:
            force = "force"
        if run == "paused":
            self._transition(self.current_state, "_".join([force, "running", step]))

    def _run_if_paused(self, _force=False):
        """Should not be called from `self._transition`

        :returns: None
        :rtype: None

        """
        force, run, step = self.current_state.split("_")
        if _force:
            force = "force"
        if run == "paused":
            self._transition(self.current_state, "_".join([force, "running", step]))

    def _run_new_if_finished(self, _force=False):
        force, run, step = self.current_state.split("_")
        # if _force:
        #     force = "force"
        if run == "finished":
            if _force:
                self._transition(self.current_state, "_".join(["force", "running", step]))
            else:
                self._transition(self.current_state, "_".join(["normal", "running", step]))
    # END: Internal Controls

    # START: State Machine Helpers
    #        Helper functions to enforce state machine commands
    def _ensure_paused(self, task):
        "Should not be called from anywhere but `self._transition`"
        self._logd(f"Calling ensure paused with {task}")
        if task == "main":
            if not self.paused:
                self._running_event.clear()  # not running
        elif task == "adhoc":
            # CHECK: Is paused supported? If not then go back or wait for abort
            # CHECK: How to force abort?
            if not self.adhoc_paused:
                self._adhoc_running_event.clear()
        elif task == "user":
            if not self.userfunc_paused:
                self._userfunc_running_event.clear()
        runner = self._task_runners.get(task)  # say epoch
        if runner is not None and runner.running:
            while not runner.waiting:
                time.sleep(1)
                self._logd(f"Waiting for {task} runner")

    def _ensure_unpaused(self, task):
        "Should not be called from anywhere but `self._transition`"
        if self.paused:
            self._running_event.set()
        runner = self._task_runners.get(task)  # say epoch
        if runner is not None and runner.waiting:
            self._logd(f"{task} runner is waiting. Should not be waiting")

    def _ensure_ready(self, task):
        "Should not be called from anywhere but `self._transition`"
        print("calling ensure ready")
        if self.paused:
            self._logd(f"No need to ensure ready while paused for {task}")
        else:
            runner = self._task_runners.get(task)  # say epoch
            if runner is not None and runner.waiting:
                runner.toggle_waiting()

    # CHECK: If finished or reset is called, then what should happen to the
    #        thread?  Does the epoch runner die?
    #
    #        Actually, since the epoch_runner etc. are reset after each epoch
    #        anyway, so it doesn't really matter. They can be killed and
    #        respawned if they get stuck.
    def _ensure_finished(self, task, timeout=5, run_cb=False):
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
        self._logd("Calling _ensure_finished")
        if not self._current_aborted:
            self._toggle_current_aborted()
        runner = self._task_runners.get(task)
        callback = self._tasks_callbacks.get(task)
        if runner is not None:
            if self.paused:
                self._toggle_running()  # unpause for abort
                runner.aborted.wait()
                self._toggle_running()
            else:
                runner.aborted.wait()
            if callback is not None and run_cb:
                self._logd("calling callback")
                callback()
            else:
                self._logd("not calling callback")
        if self._threads[self._task_thread_keys[task]].is_alive():
            self._loge(f"Could not kill task {task}")
        else:
            self._logd(f"Finished task {task}")
        self._toggle_current_aborted()
    # END: State Machine Helpers

    # START: State Machine
    def _allowed_transition(self, a, b):
        # NOTE: The states in the SM are for the main loop. For alternate loops
        #       it should be simpler.
        return self._sm.allowed_transition(a, b)

    def _transition(self, _from, _to):
        """Transitions to the next state from the given state.

        State is a string triple joined by "_". Each state `s` is composed of
        `force_run_step`, where:
                `force` can be in {"normal", "force"}
                `run` can be in {"running", "paused", "finished"}
                `step` can be any of the `self._trainer_params["training_steps"]
                    or `{"train", "val", "test", "none"}

        For a :class:`Trainer` instance the regular flow is `train_step ->
        post_epoch_hooks` or equivalent steps. However, all those can be paused
        to run auxiliary tasks to check how trainer is doing. As such, anything
        that breaks the regular flow is a forced state.

        `run` is the state of the trainer in the sense that if some active task
        is going on which has reserved some of the resourcese, whether that be
        training, validation or a given user function.
            "running" implies that some such task is present.
            "paused" implies that task is present but not active
            "finished" means that the task has finished

        On top of these `self._aborted` is not a :class:`list` of flags which is
        to be set after a task was finished but was aborted by the user. See
        `abort` for details.

        The "step" refers one of the possible steps: train, val, test, user,
        adhoc. {train, val, test} are run under a single thread and {user},
        {adhoc} under different ones. In all three threads.

        The valid states is managed by a separate module :class:`StateMachine`
        and the progress of "training", "validation" etc. are kept track of by
        the trainer itself.

        :param _from: From state
        :param _to: To state
        :returns: None
        :rtype: None

        """
        if _from != self.current_state:
            return False, self._loge(f"from state != current state: " +
                                     f"{_from} != {self.current_state}")
        if _from == "force_finished_train":
            if not self._allowed_transition("force_finished_stop", _to):
                return False, self._loge(f"State transition {_from} -> {_to} is not allowed")
        elif _to == "force_finished_train":
            if not self._allowed_transition(_from, "force_finished_stop"):
                return False, self._loge(f"State transition {_from} -> {_to} is not allowed")
        elif not self._allowed_transition(_from, _to):
            return False, self._loge(f"State transition {_from} -> {_to} is not allowed")
        self._logd(f"Trying to transition from {_from} to {_to}")

        a_force, a_run, a_step = _from.split("_")
        b_force, b_run, b_step = _to.split("_")

        # FIXME: The design needs to split here. The fact that I have to modify
        #        this component means that there's something wrong, but how do I
        #        design a component like this even?

        # NOTE: Pre transition checks
        # NOTE: Only for main loop
        if a_force == "normal" == b_force:
            self._sm.current_state = _to
        elif a_force == "normal" and b_force == "force":
            # will have to ensure first is paused?
            self._ensure_paused(a_step)
            self._prev_normal_state = self.current_state
            self._sm.current_state = _to
        elif a_force == "force" and b_force == "normal" and not a_run == "finished":
            assert _to == self._prev_normal_state
            self._sm.current_state = _to
        elif a_force == "force" and b_force == "normal" and a_run == "finished":
            self._sm.current_state = _to
        elif a_force == "force" == b_force:
            self._sm.current_state = _to

        # NOTE: Actual state transition
        # NOTE: The Task runner should exist already if not, then some error
        #        has occured and previous state should be resumed.
        if b_step in {"train", "val", "test"} and b_force == "normal":
            # NOTE: Check only for main
            if "main" in self._task_runners:
                # epoch_runner exists but "train" isn't running
                if not self._task_runners["main"].running:
                    if "main" not in self._threads or not self._threads["main"].is_alive():
                        self._threads["main"] = Thread(target=self.train)
                        self._threads["main"].start()
            else:
                # TODO: Throw massive error or go back to force_paused_train, or
                #       force_paused_none I suppose force_paused_none indicates
                #       that some massive error indeed occurred and we're stuck here.
                # NOTE: For now assume that it doesn't die
                pass
        # NOTE: b_force == normal for this case shouldn't be allowed. It could
        #       be allowed but that would mean launching it on a separate device
        #       etc. with separate models so that perhaps training is also
        #       allowed, so that it doesn't really interfere with the main thread.
        #       That is not really supported right now.
        elif b_step == "adhoc" and b_force == "force":
            # TODO: Scenarios
            #       1. main thread died?
            #       2. main thread aborted and started main thread with val?
            #       3. alt thread create?
            #       4. alt thread reset?
            # CHECK: Pause or run on CPU or something?
            if "adhoc" in self._task_runners:
                if not self._task_runners["adhoc"].running:
                    if "adhoc" not in self._threads or not self._threads["adhoc"].is_alive():
                        self._threads["adhoc"] = Thread(target=self._adhoc_func,
                                                        args=[self._adhoc_func_params])
                        self._threads["adhoc"].start()
            else:
                # Resume previous state
                pass
        elif b_step == "user":
            # CHECK: What can a user func do?
            if "user" in self._task_runners:
                if not self._task_runners["user"].running:
                    if "user" not in self._threads or not self._threads["main"].is_alive():
                        self._threads["user"] = Thread(target=self.train)
                        self._threads["user"].start()
            else:
                # Resume previous state
                pass

        # NOTE: Post transition checks
        if b_run == "running":
            self._ensure_unpaused(b_step)
        elif b_run == "paused":
            self._ensure_paused(b_step)
        elif b_run == "finished":
            self._ensure_finished(b_step, **self._transition_flags)
            self._transition_flags = {}
            self._threads["main"].join()
        if b_run in {"paused", "running"}:
            self._ensure_ready(b_step)
        return True, self._logd(f"Transitioned from {_from} to {_to}")
    # END: State Machine

    # START: Extras
    @POST
    @extras
    def load_saves(self, data):
        """Loads model weights or trainer state from a given filename. The file must be
        present in the `savedir`.

        Not sure right now, when something should be allowed to load. If it's
        paused? In the middle of current session? Should the session be
        restarted?

        :param data: 
        :returns: 
        :rtype:

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
        if "weights" not in data:
            return False, self._logi("Missing params \"weights\"")
        else:
            weights = data["weights"]
        if "method" not in data or data["method"] not in {"resume", "load"}:
            return False, self._logi("Invalid or no such method")
        else:
            method = data["method"]
        self._logd(f"Data given was {data}")
        if weights not in os.listdir(self._savedir):
            return False, self._logi("No such file")
        else:
            if method == "load":
                load_state = torch.load(os.path.join(self._savedir, weights))
                try:
                    for name in self._models.names:
                        self._models.load_weights(name, load_state["models"][name])
                except Exception as e:
                    return False,\
                        self._logi(f"Could not load weights {weights}. Error occured {e}")
                return True, self._logi(f"Successfuly loaded weights {weights}")
            else:
                try:
                    self._resume_from_path(os.path.join(self._savedir, weights))
                except Exception as e:
                    return False, self._logi(f"Could not resume from {weights}. Error occured {e}")
                return True, self._logi(f"Resumed from file")

    # CHECK: I think it's more generic now.
    @POST
    @extras
    def call_user_func(self, data):
        """Call an arbitrary function. For now calls any of train/val/test or given
        update_funcs with a subset of the dataset.

        This function is more generic than adhoc_eval in the sense that any
        adhoc function can be called on any attribute of the trainer (as of now).

        Later only specified variables will be exposed.

        """
        if not data:
            return False, self._loge(f"Called with null data")
        elif len(data) != 1:
            return False, self._loge(f"Can only call one function at a time. data is: {data}")
        self._logi(f"Calling with data: {data}")
        # NOTE: Function is not a dict right now
        # func_name = [*data.keys()][0]
        func_name = data[0]
        if func_name not in self.user_funcs:
            return False, {"error": self._loge(f"Unknown function {func_name} given"),
                           "available_functions": self.user_funcs}
        elif func_name in self.user_funcs:
            func = self._user_funcs[func_name]
            if not all(getattr(self, x, None) for x in inspect.signature(func).parameters):
                return False, {"error": self._loge(f"Some of the parameters for {func_name}: " +
                                                   f"{inspect.signature(func).parameters}" +
                                                   " are not available")}
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
                return False, f"Unexpected output format for function {func_name}," +\
                    f" Error occured {e}"
            if callback not in self.user_funcs:
                return False, {"error": self._loge(f"Unknown function {callback} given"),
                               "available_functions": self.user_funcs}
            elif callback in self.user_funcs:
                callback_func = self._user_funcs[callback]
                param_names = [x for x in inspect.signature(callback_func).parameters]
                flag = True
                for _x in param_names:
                    if not getattr(self, _x, None) and _x != "output":
                        flag = False
                        break
                if not flag:
                    return False, {"error": self._loge(f"Some of the parameters for callback function" +
                                                       f" {callback}: " +
                                                       f" {param_names}" +
                                                       " are not available")}
                else:
                    params = {"output": output}
                    param_names.remove("output")
                    for _x in param_names:
                        params[_x] = getattr(self, _x)
                    return callback_func(**params)

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
    @POST
    @extras
    def adhoc_eval(self, data):
        """Do an arbitrary evaluation on any or combination of train/val/test data for
        any state in the model's stored history

        1. Create a new task runner
        2. adhoc_eval is called on either or combination of train/val/test data
        3. What to gather and callback can be specified beforehand
        4. Result is stored in _adhoc_func_result
        5. Callbacks can be called on the result multiple times.

        :param data: data
        :returns: status and response string
        :rtype: :class:`tuple`

        """
        if not data:
            return False, self._loge(f"Called with null data")
        self._logi(f"Calling adhoc_eval with data: {data}")
        if not any(x in data for x in self._trainer_params["training_steps"]):
            return False, {"error": self._logi("Required Input. Given unknown dataset"),
                           **self.adhoc_error_dict}
        else:
            for x in data:
                return self.check_adhoc_eval_params(x, data[x])

    def check_adhoc_eval_params(self, func, params):
        """Call :meth:`call_adhoc_func_on_data` with params. Perhaps it can be lifted
        above though

        :param func: Function name. Should be present in :meth:`Trainer.user_funcs`
                     or one of {"train", "val", "test"}
        :param params: `params` is a :class:`dict` of type
                        {"epoch": :class:`int` num or "current",
                         "num_or_fraction": 0 < x,
                         "device", "gpu" or "cpu",  # allocated automatically
                         "parallel", True or False,
                         "data": "train_val_or_test",
                         "callback": "name_of_callback_function"}

        """
        # NOTE: Samples should be captured by default, model defines a sampling
        #       mechanism or else simply output is captured
        self._logw("Ignoring \"epoch\" for now")
        if params["epoch"] != "current" or self.epoch != params["epoch"]:
            # Load the model if present in history
            pass
        try:
            iter(params)
        except TypeError:
            return False, {"error": self._logi("Required Input. Incorrent format"),
                           **self.adhoc_error_dict}
        if not all(x in params for x in ["epoch", "num_or_fraction", "data", "device",
                                         "callback"]):
            return False, {"error": self._logi("Required Input. Incorrent parameters"),
                           **self.adhoc_error_dict}
        # NOTE: From now on gather everything the step_func returns and wait for
        #       callback. "metrics" is removed
        # elif not (params["metrics"] != "all") or\
        #      not all(x in self._metrics[params["data"]] for x in params["metrics"]):
        #     self._logd(f'metrics given {params["metrics"]}')
        #     return False, {"error": self._loge("Required Input. Given unknown metrics or incorrect format"),
        #                    **self.adhoc_error_dict}
        # elif func not in (self.user_funcs + self._trainer_params["training_steps"]):
        #     return False, {"error": self._loge(f"Unknown function \"{params['function']}\" given"),
        #                    "available_functions": (self.user_funcs +
        #                                            self._trainer_params["training_steps"])}
        # Making minimal assumptions on the function
        # elif func in self.user_funcs and\
        #      not len(inspect.signature(self._user_funcs[func]).parameters) == 1:
        #     return False, {"error": self._loge(f"Given function \"{params['function']}\"" +
        #                                        " is not suited to process data")}
        elif params["num_or_fraction"] <= 0:
            return False, self._loge(f"Incorrect fraction or number of points" +
                                     " {params['num_or_fraction']}")
        elif params["device"] not in {"gpu", "cpu"}:
            return False, self._loge(f"Incorrect device given {params['device']}")
        else:
            # NOTE: All this should be rewritten with guards

            # call this in a separate thread and call the callback on the result
            # then report it.
            # NOTE: New thread should only be started from _transition
            self._adhoc_func = self.call_adhoc_eval_on_data
            self._adhoc_func_params = params
            self._transition(self.current_state, "force_running_adhoc")
            # self.pause()
            # while not self.paused:
            #     time.sleep(10)
            # params["function_name"] = func
            # t = Thread(target=self.call_adhoc_func_on_data, args=[params])
            # if not self._flag_adhoc_func_running:
            #     self._flag_adhoc_func_running = True
            #     t.start()
            #     return True, {"success": self._logi("Running the given adhoc function")}
            # else:
            #     return False, {"error": self._logi("Another adhoc function is still running")}

    def call_adhoc_eval_on_data(self, params):
        """Call adhoc evaluation on data and wait for result. Result is processed by the
        callback given by the user.

        :param params: Parameters specified for the function
        :returns: None
        :rtype: None

        """
        step = params["data"]
        function = self._update_functions[step]
        step_loader = getattr(self, step + "_loader")
        if "seed" in params:
            np.random.seed(params["seed"])
        if params["num_or_fraction"] > 1:
            indices = np.random.choice(len(step_loader.dataset), params["num_or_fraction"])
        else:
            indices = np.random.choice(len(step_loader.dataset),
                                       int(len(step_loader.dataset) * params["num_or_fraction"]))
        _proxy_dataset = ProxyDataset(step_loader.dataset, indices)
        temp_params = self._dataloader_params[step].copy()
        temp_params.update({"batch_size": 1})  # stick to 1 right now
        # NOTE: MyDataLoader is to solve the problem of collation in data. So
        #       that there's a uniform interface to data. However there seem to
        #       be some problems.
        if hasattr(step_loader.dataset, "_get_raw"):
            _proxy_dataset._get_raw = lambda x: step_loader.dataset._get_raw(
                _proxy_dataset._indices[x])
            temp_loader = MyDataLoader(_proxy_dataset, return_raw=True,
                                       **temp_params)
            self._logi(f"{step} dataset has \"_get_raw\"" +
                       "Drawing samples from temp data is available!")
        else:
            temp_loader = MyDataLoader(_proxy_dataset, **temp_params)
            self._logw(f"{step} dataset doesn't define \"_get_raw\"" +
                       "Drawing samples from temp data will not be available.")
        if step == "train":
            raise NotImplementedError
            models = {}
            optimizers = {}
            devices = {}
            for model_name, model_params in self._model_params.items():
                models[model_name] = self._model_defs[model_name]["model"](**model_params)
                optim_name = self._model_defs[model_name]["optimizer"]
                optimizers[model_name] = {"name": optim_name,
                                          "optimizer": self._optimizer_params
                                          [optim_name]["function"](models[model_name].parameters(),
                                              **self._optimizer_params[optim_name]["params"])}
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
        metrics = self._trainer_params["metrics"]
        if hasattr(step_loader.dataset, "_get_raw"):
            metrics.append("raw")
        metrics.append("predictions")
        metrics.append("labels")

        callback = self._user_funcs[params["callback"]]
        self._task_runner_initialize("adhoc", {step: metrics}, {}, callback)
        self._logd(f"starting temp_runner for {step} step")
        # TODO: It should be temp_runner.run_temp instead of run_ + step
        # TODO: Threads should only be handled by the transition function
        if hasattr(temp_loader.dataset, "_get_raw"):
            t = Thread(target=getattr(temp_runner, "run_" + step),
                       args=[step_func, temp_loader, True])
        else:
            t = Thread(target=getattr(temp_runner, "run_" + step),
                       args=[step_func, temp_loader])
        # report function only takes in targets and predictions
        t.start()
        Thread(target=self._check_adhoc_run).start()

    def _check_adhoc_run(self):
        while self._temp_runner.running:
            time.sleep(1)
        self._flag_adhoc_func_running = False
        self.resume()

    # NOTE: Actually report_function should be user defined and should be
    #       uploaded as a python module so that along with the adhoc_func, it
    #       can be processed and reported according to how the _get_raw is
    #       implemented and how adhoc_function actually operates.
    @POST
    @extras
    @Exposes("ix_to_word", "batch_vars")
    def report_adhoc_run(self, data):
        # TODO: These generic checks should be in the POST or GET pre_call
        #+FROM_HERE
        # ix_to_word may be used by user_func later
        ix_to_word = self.train_loader.loader._ix_to_word
        batch_vars = self._temp_runner.batch_vars
        if data is None:
            return False, self._loge("Called with null data.")
        elif "report_function" not in data:
            return False, self._loge("report_function not in data.")
        elif data["report_function"] not in self._user_funcs:
            report_function = data["report_function"]
            return False, self._loge(f"Unknown report function {report_function}.")
        else:
            report_function = self._user_funcs[data["report_function"]]
        # NOTE: Should I just call the report func all the time? and it should
        #       be the report func's responsibility to check the data and
        #       respond? Make the report func a generic class?  Although the
        #       whole point was to avoid boilerplate.
        #       - One way could be to wrap the function in a class and replace the
        #         call method with the function. But then the function will never
        #         access "self" and in any case has no idea what the instance may
        #         have as its remote and opaque.
        #       - And of course, the whole point of "Exposes" is to make sure that
        #         the function's attributes are declared alongside the function to
        #         enhance readability. I can declare it somewhere in the class but...
        #       - I can make a self._exposes attribute which contains the func_name
        #         list

        # TODO: "check" would be better.  In fact these checks should be done
        #       before the func is called
        if not all(x in self.report_adhoc_run.exposes
                   for x in inspect.signature(report_function).parameters):
            return False, f"Given function {data['report_function']}" +\
                " is not compatible with report_adhoc_run"
        #+TO_HERE
        if not hasattr(self._temp_runner, "running"):
            return False, self._logd("Adhoc function was never initialized")
        elif self._temp_runner.running:
            return True, self._logd("Adhoc function is still running")
        else:
            param_names = list(inspect.signature(report_function).parameters.keys())
            params = {}
            # CHECK: Here. Can this be changed?
            for x in param_names:
                params[x] = locals()[x]
            output = report_function(**params)
            return True, {"success": output}
    # END: Extras

    # START: Helpers
    @POST
    @helpers
    def set_model(self, model_name):
        return self._set_model_active(model_name)

    def _set_model_active(self, model_name):
        """Model name is an abstraction and a `model` can have multiple
        :class:`torch.nn.Module` modules within it with separate criteria and
        optimizers. It is the prerogative of the update_function to interact
        with the model.

        :param model_name: :class:`str` model_name
        :returns: None
        :rtype: None

        """
        if model_name not in self._models.names:
            return False, self._loge(f"No such model {model_name}")
        else:
            for name in self._models:
                if name != model_name:  # free only GPU resources
                    self._models.set_device(model_name, torch.device("cpu"))
            for x in self._update_functions:
                self._update_functions[x]._model_name = model_name
            return True, self._logd(f"Model {model_name} is now the current active model.")

    @POST
    @helpers
    def fetch_preds(self, img_path):
        """Fetch the prediction for a given image. Returns predictions

        :param img_path: Image Path
        :returns: preds: {"beam_preds": beam_preds, "greedy_preds": greedy_preds}
        :rtype: :class:`dict`

        # Test would be something like 
        >>> response = requests.request("GET", server_url)
        >>> 

        """
        if True:              # img_path in self._temp_runner._processed_images:
            if not hasattr(self._temp_runner, "running") or self._temp_runner.running:
                return False, f"The function is still running"
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
                    return False, f"Img seems to not have been processed. Check."
                if all(x is not None for x in [batch_preds, batch_lengths,
                                               img_indx, batch_targets]):
                    probs, indices = torch.topk(torch.nn.functional.softmax(batch_preds, 1), 5)
                    probs = probs.cpu().numpy()
                    indices = indices.cpu().numpy()
                    vocab = self.report_function.__getattribute__("keywords")["vocab"]
                    preds = [vocab.idx2word[int(x[0])] for x in indices]
                    topk = [[vocab.idx2word[int(x)] for x in y] for y in indices]
                    targets = [vocab.idx2word[int(x)] for x in batch_targets]
                    # _raw = [x for x in self._temp_runner.batch_vars if x[2] == "raw"]
                    return True, {"preds_targets": [preds, targets],
                                  "topk_words": topk, "probs": probs}
                else:
                    return False, f"Img seems to not have been processed. Check."
        elif not os.path.exists(img_path):
            return False, f"Image {img_path} doesn't exist"
        else:
            return False, f"Evaluation of single image is not implemented yet"
            # try:
            #     Image.open(img_path)
            # except Exception as e:
            #     return False, f"Error occurred while reading file {e}"
            # Assuming predictions exist already somewhere in the report

    @POST
    @helpers
    def fetch_image(self, img_path):
        """Fetch the image from a given path.
        """
        img_path = os.path.join(self._trainer_params["image_root"], img_path)
        if not os.path.exists(img_path):
            return False, self._logd(f"Image {img_path} doesn't exist")
        else:
            try:
                Image.open(img_path)
            except Exception as e:
                return False, self._loge(f"Error occurred while reading file {e}")
            with open(img_path, "rb") as img_file:
                return True, base64.b64encode(img_file.read()).decode("utf-8")

    @POST
    @helpers
    def load_image(self, request):
        """Load an image from the given data.
        """
        try:
            img_file = request.files["file"].read()
            test = self._check_file_magic(img_file, "image")
        except Exception as e:
            return False, self._loge(f"Error reading file {e}")
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
            return True, "meh"
        else:
            return False, self._loge("Data is not image")

    @POST
    @helpers
    def load_weights(self, request):
        if "model_names" not in request.form:
            return False, self._logd(f"Model name not sent in data")
        model_names = json.loads(request.form["model_names"])
        try:
            weights = torch.load(request.files["file"], map_location="cpu")
        except Exception as e:
            return False, self._loge(f"Error occured while reading data {e}")
        if not all(x in weights for x in model_names):
            return False, self._logd(f"Check save file! " +
                                     f"Not all {model_names} in given weights {weights.keys()}")
        if not all(x in self._models.names for x in model_names):
            return False, self._logd(f"Some models currently not in scope")
        try:
            for model_name in model_names:
                status, err = self._models.load_weights(model_name, weights[model_name])
                if err:
                    return False, self._loge(f"Error while updating component {err}")
            return True, self._logd(f"Updated Models {model_names}")
        except Exception as e:
            return False, self._loge(f"Error occured while loading models {e}")

    @POST
    @helpers
    def hack_param(self, data):
        """Update a param as a hack. Data is assumed to be a pair of `{key, [type, value]}`
        dictionary."""
        statuses = []
        for k, v in data.items():
            if not hasattr(self, k):
                self._logw(f"{k} not an attribute of {self}")
            else:
                try:
                    if v["type"] in {"str", "int", "float"}:
                        _v = {"str": str, "int": int, "float": float}[v["type"]](v["value"])
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
            return True, "All values updated!"
        elif any(statuses):
            return True, "Some values could not be updated."
        else:
            return False, "None of the values could be updated."

    @POST
    @helpers
    def add_user_funcs(self, request):
        return self._mods.add_user_funcs(request, self._user_funcs)

    @POST
    @helpers
    def add_model(self, request):
        """Add a model from a given python or module as a zip file.
        Delegates the request to :meth:`Trainer.add_module`

        For this case ``module_exports`` has to include at models and
        optimizers. Optimizers can be a string and if so, if optimizer_params
        are given then it is initialized with that, else the params are checked
        in instance scope. If both aren't present it's initialized with default
        params from :mod:`torch.optim`

        :param flask.request request: request is the http request
        :returns: :class:`bool` status, :class:`str` message
        :rtype: :class:`tuple`

        """
        checks = []
        status, response = self.add_module(request, checks)
        if status:
            module_exports = response
            if "model_names" not in module_exports:
                return False, self._logw(f"Model name not sent in data")
            else:
                status, models = self._get_new_models(module_exports["model_names"],
                                                      module_exports["model_defs"],
                                                      module_exports["model_params"])
                if not status:
                    return status, models
                else:
                    # CHECK: How to force destruction of those models and resources and
                    #        initialize new ones? Does thread work?
                    try:
                        for name in models["models"]:
                            model = models["models"][name]
                            params = {"name": name,
                                      "optimizer": models["optimizers"][name],
                                      "optimizer_name": models["optim_names"][name],
                                      "device": self.device}
                            self._models.add(model, params)
                    except Exception as e:
                        return False, self._loge(f"Some weird error occured {e}")
                    return status, self._logd(f"Added model {name} successfully")
        else:
            return status, response

    # TODO: Any change in state of trainer vars should have a rollback mechanism
    #       E.g., if some of the params change here and then an error is raised.
    #
    # TODO: model initialization is repetitive here and should be delegated to a
    #       subroutine of _init_models. Fix.
    def _get_new_models(self, model_names, model_defs, model_params):
        """Extracts ``models`` from the ``model_names``, ``model_defs`` and ``model_params``

        :param list model_names:
        :param dict model_defs:
        :param dict model_params:

        :returns: A :class:`tuple` of ``status``, ``response`` where if
        ``status`` is successful the response is model else an error string

        :rtype: :class:`tuple`

        """

        if not all(x in model_params and x in model_defs for x in model_names):
            return False, self._logd(f"Some of the model_names not in given module")
        models = {"models": {}, "optimizers": {}, "optim_names": {}}
        for model in model_names:
            _def = model_defs[model]["model"]
            _params = model_params[model]
            if "__inherit" in _params:
                if "__add" in _params:  # NOTE: add params from self, stupid hack
                    add_params = _params["__add"]
                else:
                    add_params = []
                inherit_name = _params["__inherit"]
                sig = inspect.signature(_def)
                model_args = {}
                for x in sig.parameters:
                    if x not in add_params:
                        model_args[x] = self._model_params[inherit_name][x]
                    else:
                        model_args[x] = getattr(self, x)  # NOTE: Bad hack
            else:
                model_args = _params
            if model not in self._model_defs:
                self._model_defs[model] = {}
            else:
                self._logw(f"Will overwrite model, optimizer params and defs for {model}")
            self._model_defs[model]["model"] = _def
            self._model_params[model] = model_args.copy()
            self._logd(f"Updated model_params and model_def for {model}")
            models["models"][model] = _def(**model_args)
            if isinstance(model_defs[model]["optimizer"], str):
                optim_name = model_defs[model]["optimizer"]
                self._model_defs[model]["optimizer"] = optim_name
                self._logd(f"Updated optimizer params for {model}")
                models["optim_names"][model] = optim_name
                if "optimizer_params" in model_defs and hasattr(torch.optim, optim_name):
                    models["optimizers"][model] = getattr(torch.optim, optim_name)(
                        **model_defs[model]["optimizer_params"])
                    self._logd(f"Initialized optimizer for {model} in add_model with given params")
                elif optim_name in self._optimizer_params:
                    models["optimizers"][model] = self._optimizer_params[optim_name]["function"](
                        models["models"][model].parameters(),
                        **self._optimizer_params[optim_name]["params"])
                    self._logd(f"Initialized optimizer for {model} in add_model with self params")
                else:
                    models["optimizers"][model] = getattr(torch.optim, optim_name)()
                    self._logw(f"Initialized optimizer for {model} in add_model with default params")
            else:
                False, self._logd(f"Unrecognized optimizer for model {model}")
        return True, models

    @POST
    @helpers
    def add_module(self, request, checks):
        return self._mods.add_module(request, checks)
    # END: Helpers

    # START: Save, Load, Resume
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
        save_state = {}
        save_state["epoch"] = self.epoch
        save_state["iterations"] = self.iterations
        save_state["models"] = self._models.dump()
        save_state["model_params"] = copy.deepcopy(self._model_params)
        save_state["criteria_params"] = copy.deepcopy(self._criteria_params)
        save_state["dataloader_params"] = {}
        for k, v in self._dataloader_params.items():
            if v is None:
                save_state["dataloader_params"][k] = None
            else:
                save_state["dataloader_params"][k] = {}
                for a, b in v.items():
                    if a == "collate_fn":
                        self._logw(f"collate_fn in dataloader {k} params will not be saved")
                        save_state["dataloader_params"][k][a] = "callable_" + type(b).__qualname__
                    else:
                        value = self._dataloader_params[k][a]
                        if isinstance(value, dict):
                            save_state["dataloader_params"][k][a] = {}
                            for x, y in value.items():
                                if callable(y):
                                    self._logw(f"callable {type(y).__qualname__} in dataloader" +
                                               f" {k} params {a, x} will not be saved")
                                    save_state["dataloader_params"][k][a][x] = "callable_" +\
                                        type(y).__qualname__
                                else:
                                    save_state["dataloader_params"][k][a][x] = y
                        else:
                            if callable(value):
                                self._logw(f"callable {value} in dataloader {k}" +
                                           f" params {a} will not be saved")
                                save_state["dataloader_params"][k][a] = "callable_" +\
                                    type(value).__qualname__
                            else:
                                save_state["dataloader_params"][k][a] = value
        save_state["trainer_params"] = {}
        for k, v in self._trainer_params.items():
            if callable(v):
                self._logw(f"callable {type(v).__qualname__}" +
                           f" for trainer_params {k} will not be saved")
                save_state["trainer_params"][k] = "callable_" + type(v).__qualname__
            else:
                save_state["trainer_params"][k] = copy.deepcopy(v)
        save_state["metrics"] = self._metrics
        self._logi(f"Saving to {save_path}")

        # FIXME: If some thing is not saved, it cannot be resumed also
        # NOTE: As I've removed callable saving from dataloader params,
        #       this should save now
        def try_save():
            try:
                torch.save(save_state, save_path)
                not_saved = False
            except Exception:
                not_saved = True
            return not_saved
        not_saved = try_save()
        while not_saved:
            self.fix_state(save_state, save_path)
            not_saved = try_save()

    def fix_state(self, save_state, save_path):
        tmp_path = save_path + "_tmp"

        def find_error_key(state_dict):
            for x in state_dict.keys():
                try:
                    torch.save(state_dict[x], tmp_path)
                except Exception:
                    return x
            return None
        state_dict = save_state
        keys = []
        while True:
            if isinstance(state_dict, dict):
                error_key = find_error_key(state_dict)
            else:
                break
            if error_key:
                keys.append(error_key)
                state_dict = state_dict[error_key]
        x = save_state
        for key in keys[:-1]:
            x = x[key]
        x[keys[-1]] = type(x[keys[-1]]).__qualname__
        self._logw(f"Value with keychain {keys} could not be saved." +
                   f" Replaced with name = {x[keys[-1]]}")
        if "not_saved" not in save_state:
            save_state["not_saved"] = [keys]
        else:
            save_state["not_saved"].append(keys)
        os.remove(tmp_path)

    # DONE: Unique Id check
    #       - We only resume with same id
    # DONE: Check if {models, metrics, dataloaders, update_funcs} are resumed correctly as
    #       there may be callables in the saved_state. trainer shouldn't allow callables
    #       - Callables are not resumed
    # TODO: Right now, the list of saves and resume_path etc are given as full paths while
    #       they should be relative paths to .savedir/unique_id/"_".join(model_names)
    def _resume_from_path(self, resume_path):
        self._have_resumed = True
        saved_state = torch.load(resume_path)
        # not_saved = saved_state["not_saved"]
        self.epoch = saved_state["epoch"]
        self.iterations = saved_state["iterations"]
        self._model_params = saved_state["model_params"]
        self._criteria_params = saved_state["criteria_params"]

        # NOTE: restore dataloader_params
        self._logd("Restoring dataloader_params")
        if any([("collate_fn" in y or callable(y))
                for x, y in saved_state["dataloader_params"].items()]):
            self._logw("collate_fn will not be restored")
        for k, v in saved_state["dataloader_params"].items():
            for a, b in v.items():
                if a != "collate_fn":
                    value = saved_state["dataloader_params"][k][a]
                    if isinstance(value, dict):
                        for x, y in value.items():
                            if isinstance(y, str) and not y.startswith("callable_"):
                                self._dataloader_params[k][a][x] = y
                    else:
                        if isinstance(value, str) and not value.startswith("callable_"):
                            self._dataloader_params[k][a] = value

        # NOTE: restore trainer_params
        self._logd("Restoring trainer_params")
        for k, v in saved_state["trainer_params"].items():
            if isinstance(v, str) and v.startswith("callable_"):
                self._logw(f"callable {k} not restored in trainer_params")
            else:
                self._trainer_params[k] = saved_state["trainer_params"][k]

        # NOTE: sanity check after param updates
        self._sanity_check()
        # FIXME: Init models again only if model or model parameters have changed
        self._init_models()
        self._init_nvml()
        self._init_dataloaders()

        # Only if criteria and/or optimizer have changed.  In fact, there might
        # be a mismatch if criteria change suddenly as the model has changed,
        # but resume_weights should not really be concerned about that, at
        # least.
        # self._init_criteria_optimizers()
        # Only if new metrics are added and even then only update metrics
        # NOTE: Other inits
        self._init_metrics()
        # Only if update_funcs are changed.
        # In fact, this is not in saved state
        self._init_update_funcs()
        self._init_state_vars()
        self._init_epoch_runner()
        self._init_modules()

        # NOTE: The model and optimizer checks are in Models
        # NOTE: restore model
        self._logd("Restoring models and optimizers")
        default = [*self._optimizer_params.keys()][0]
        for k in saved_state["models"].keys():
            if "optimizer" in saved_state["models"][k]:
                self._logw(f"Optimizer shouldn't be in saved_state for model {k}")
                x = saved_state.pop("optimizer")
                saved_state["models"][k]["optimizer_name"] = x
            elif "optimizer_name" not in saved_state["models"][k]:
                optim_name = default
                self._logw(f"No optimizer_name in saved_state for model {k}. Using {default}")
            else:
                optim_name = saved_state["models"][k]["optimizer_name"]
                if optim_name not in self._optimizer_params:
                    self._logw(f"{optim_name} not a known optimizer for model {k}. Using {default}")
                    optim_name = default
            optim = self._optimizer_params[optim_name]
            saved_state["models"][k]["optimizer_name"] = optim_name
            saved_state["models"][k]["optimizer"] = optim["function"](self._models[k].parameters(),
                                                                      **optim["params"])
        self._models.load(saved_state["models"])
        diff = set(self._metrics.keys()).difference(saved_state["metrics"].keys())

        # NOTE: setup metrics
        if diff:
            self._logw(f"Some metric _steps_ aren't there in saved state {diff}")
        for k in self._metrics.keys():
            diff = set(self._metrics[k].keys()).difference(saved_state["metrics"][k].keys())
            if diff:
                self._logw(f"Some metrics {diff} in {k} aren't there in saved state")
        self._logd("Restoring metrics")
        self._metrics = copy.deepcopy(saved_state["metrics"])
        self._logi("Resumed successfully")

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
    def reset_session(self):
        """Reset should dump all the variables and data to a backup state with the
        option to save, restore or delete later and reset all the state of the
        session to init_state.
        """
        self.stop()
        self.save()
        self._logi("Resetting the current session")
        self._init_all()

    @control
    def does_nothing(self):
        pass

    @control
    def pause(self):
        self._pause_if_running()
        return self._logi("Pausing")

    @control
    def resume(self):
        self._run_if_paused()
        return self._logi("Resuming")

    @control
    def start(self):
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

    # Actually a pause, "force_save" and then resume
    @control
    def save(self):
        self._logi("Saving")
        self._pause_if_running()
        self._transition(self.current_state, "force_running_save")
        self._logw("Trying force save")
        try:
            self._save(self._save_path_with_epoch + "_force")
            status = True
            message = f"Saved to {self._save_path_with_epoch}" + "_force"
        except Exception as e:
            status = False
            message = f"Could not save to {self._save_path_with_epoch}" + "_force" +\
                f" error {e}"
        self._transition(self.current_state, self._prev_normal_state)
        return status, message

    @control
    def force_eval(self):
        self._pause_if_running()
        self._transition(self.current_state, "force_running_eval")
        # with a _temp_epoch_runner
        # self.validate()
        # Save everything to a temp_run_metrics
        self._transition(self.current_state, self._prev_normal_state)

    @control
    def force_test(self):
        self._pause_if_running()
        self._transition(self.current_state, "force_running_test")
        # with a _temp_epoch_runner
        # self.test()
        self._transition(self.current_state, self._prev_normal_state)

    @control
    def abort_session(self):
        """``abort_session`` finishes the session and switches to "finished" state with
        aborted flag set to true. Saves the current session with aborted suffix.

        :returns: None
        :rtype: None

        """
        try:
            self._abort_current()
            self.save()
            self._abort_session()
        except Exception as e:
            return False, self._logi(f"Could not abort {self.current_state}. Error {e}")
        return True, self._logi(f"Aborted state {self.current_state} and current session")

    @control
    def abort_loop(self):
        """`abort_loop` aborts only the current loop stops with the aborted flag. Useful
        for changing the parameters and starting again. Saves the current
        metrics gathered.

        :returns: None
        :rtype: None

        """
        try:
            self._abort_current()
        except Exception as e:
            return False, self._logi(f"Could not abort {self.current_state}. Error {e}")
        return True, self._logi(f"Aborted {self.current_state}")

    @control
    def abort_loop_with_callback(self):
        """`abort_loop` aborts only the current loop stops with the aborted flag. Useful
        for changing the parameters and starting again. Saves the current
        metrics gathered.

        :returns: None
        :rtype: None

        """
        try:
            self._abort_current_run_cb()
        except Exception as e:
            return False, self._logi(f"Could not abort {self.current_state}. Error {e}")
        return True, self._logi(f"Aborted {self.current_state}")
    # END: Controls

    # START: Flags
    def _toggle_running(self):
        if self._running_event.is_set():
            self._running_event.clear()
        else:
            self._running_event.set()

    def _toggle_current_aborted(self):
        if self._current_aborted_event.is_set():
            self._current_aborted_event.clear()
        else:
            self._current_aborted_event.set()

    def _toggle_session_aborted(self):
        if self._session_aborted_event.is_set():
            self._session_aborted_event.clear()
        else:
            self._session_aborted_event.set()

    def _abort_session(self):
        # any -> force_finished_none with aborted_session True
        self._session_aborted_event.set()
        self._transition(self.current_state, "force_finshed_none")

    def _abort_current(self):
        self._current_aborted_event.set()
        self._finish_if_paused_or_running(True)
        self._aborted.append(self.current_state.split("_")[-1])

    def _abort_current_run_cb(self):
        self._current_aborted_event.set()
        self._finish_if_paused_or_running(True, True)
        self._aborted.append(self.current_state.split("_")[-1])
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

    def _run_user_func(self, user_func_name):
        # pauses main loop, shouldn't update weights
        pass

    def _run_user_func_parallel(self, user_func_name):
        # runs in alternate loop, if model is used will make a copy of model
        pass
    # END: Internal Controls Other

    # START: Properties
    @property
    def version(self):
        return self.__version__

    @property
    def unique_id(self):
        return self._unique_id

    @property
    def current_state(self):
        return self._sm.current_state

    @property
    def loop_type(self):
        if "iterations" in self._trainer_params["training_steps"]:
            return "iterations"
        else:
            return "epoch"

    @property
    def logger(self):
        return self._logger

    @property
    def logfile(self):
        return open(self._logfile).read()

    @property
    def saves(self):
        return os.listdir(self._savedir)

    @property
    def gpus(self):
        return self._gpus

    @property
    def system_info(self):
        return {"gpu_util": gpu_util(self._device_handles) if self._gpus[0] != -1 else None,
                "cpu_info": cpu_info(),
                "memory": memory_info()}

    @property
    def device(self):
        return self._device

    # FIXME: self.models creates problems
    @property
    def models(self):
        return self._models.names

    @property
    def train_step_func(self):
        return partial(self._train_step_func, self._models, self.criteria)

    @property
    def val_step_func(self):
        return partial(self._val_step_func, self._models, self.criteria)

    @property
    def test_step_func(self):
        return partial(self._val_step_func, self._models, self.criteria)

    @property
    def active_model(self):
        "Active model is both get and set by setting the _update_function"
        # NOTE: Was self._update_functions[self._trainer_params["training_steps"][0]]._model_name
        #       "train" is assumed to be present as a step
        return self._update_functions["train"]._model_name

    # exclude properties beginning with _
    @property
    def props(self):
        return [x for x, y in self.__class__.__dict__.items()
                if isinstance(y, property) and
                x != "props" and
                (x in {"_extras", "_helpers"} or not x.startswith("_"))]

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, epoch):
        self._epoch = epoch

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, x):
        self._iterations = x

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def controls(self):
        """Which of the functions can be accessed via the API

        :returns: API exports
        :rtype: list

        """
        # return dict((x, self.__getattribute__(x)) for x in self._controls)
        return dict((x.__name__, x) for x in control.members)

    @property
    def _helpers(self):
        return dict((x.__name__, x) for x in helpers.members)

    @property
    def _extras(self):
        return dict((x.__name__, x) for x in extras.members)

    # START: State props
    @property
    def running(self):
        return self._running_event.is_set()

    @property
    def current_run(self):
        if "_epoch_runner" not in self.__dict__:
            return "None"
        else:
            return self._epoch_runner.current_loop

    @property
    def paused(self):
        return not self._running_event.is_set()

    @property
    def _current_aborted(self):
        return self._current_aborted_event.is_set()

    @property
    def _session_aborted(self):
        return self._session_aborted_event.is_set()

    @property
    def adhoc_paused(self):
        return not self._adhoc_aborted_event.is_set()

    # Are adhoc_aborted and userfunc_aborted needed?
    # If they can all be run together then there's no concept of a
    # "current_loop". current_loop therefore only applies to [train, val, test]
    @property
    def adhoc_aborted(self):
        return self._adhoc_aborted_event.is_set()

    @property
    def userfunc_paused(self):
        return not self._userfunc_aborted_event.is_set()

    @property
    def userfunc_aborted(self):
        return self._userfunc_aborted_event.is_set()
    # END: State props

    @property
    def best_save(self):
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
                        results.append(f, result.group())
                if results:
                    results.sort(key=lambda x: x[1])
                    return os.path.join(self._savedir, results[-1][0])
                else:
                    return None
            else:
                return None

    # Internal property. Will not be exposed outside
    @property
    def _save_path_with_epoch(self):
        if "iterations" in self._trainer_params["training_steps"]:
            update_key = self.iterations / self._hooks_run_iter_frequency
        else:
            update_key = self.epoch
        model_names = "_".join(self._models.names)
        save_name = os.path.join(self._savedir, "_".join([str(self._unique_id),
                                                          model_names,
                                                          "{:03}".format(update_key)]))
        return save_name

    @property
    def _save_path_without_epoch(self):
        model_names = "_".join(self._models.names)
        save_name = os.path.join(self._savedir, "_".join([str(self._unique_id),
                                                          model_names]))
        return save_name

    @property
    def _checkpoint_path(self):
        # model_names = "_".join(self._models.names)
        # save_name = "_".join([str(self._unique_id), model_names, "checkpoint"])
        return os.path.join(self._save_path_without_epoch + "_checkpoint" + ".pth")

    # TODO: Allow extra_metrics, update_funcs and any other params to be updated
    @property
    def updatable_params(self):
        params = {}
        params["model_params"] = self._model_params
        params["trainer_params"] = self._trainer_params
        params["dataloader_params"] = self._dataloader_params
        return params

    @property
    def all_params(self):
        save_state = {}
        save_state["epoch"] = self.epoch
        save_state["iterations"] = self._iterations
        # save_state["models"] = dict((k, v.state_dict()) for k, v in self._models.items())
        # save_state["optimizers"] = dict((k, v.state_dict()) for k, v in self.optimizers.items())
        save_state["model_params"] = self._model_params
        save_state["criteria_params"] = self._criteria_params
        save_state["dataloader_params"] = self._dataloader_params
        save_state["trainer_params"] = self._trainer_params
        save_state["metrics"] = self._metrics
        # return _dump(save_state)
        return save_state

    @property
    def all_attrs(self):
        return self.__dict__

    # TODO: What about other losses
    #       `prop` can help
    @property
    def train_losses(self):
        return dict((k, v) for k, v in self._metrics["train"].items()
                    if k[0] == "loss")

    @property
    def progress(self):
        predicate = "iterations" in self._trainer_params["training_steps"]
        cur_step = self.iterations / self._hooks_run_iter_frequency\
            if predicate else self.epoch
        max_step = self.max_iterations / self._hooks_run_iter_frequency\
            if predicate else self.max_epochs
        cur_round = self._epoch_runner.info["batch_nums"]["train"]
        max_round = self._hooks_run_iter_frequency if predicate else len(self.train_loader)
        return {"cur_step": cur_step, "max_step": max_step,
                "cur_round": cur_round, "max_round": max_round}

    # FIXME: self.user_funcs MAY create problems
    @property
    def user_funcs(self):
        return [x for x in self._user_funcs]

    @property
    def _current_user_func(self):
        if self._current_user_func_name and\
           self._current_user_func_params:
            return partial(self._user_funcs[self._current_user_func_name],
                           kwargs=self._current_user_func_params)
        else:
            return lambda: None

    @property
    def metrics(self):
        return self._metrics

    # TODO: Define what is a sample correctly
    @property
    def val_samples(self):
        return dict((k, v) for k, v in self._metrics["val"].items()
                    if k[0] == "sample")

    # NOTE: Not sure if I want to use dir(self)
    @property
    def all_post_epoch_hooks(self):
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

    @property
    def post_epoch_hooks_to_run(self):
        return self._post_epoch_hooks_to_run

    @property
    def items_to_log_dict(self):
        """Which of the items will be logged."""
        return self._items_to_log_dict
    # END: Properties

    # START: Broken funcs
    # where are the hooks run?
    # @post_epoch_hooks_to_run.setter
    # def post_epoch_hooks_to_run(self, x):
    #     assert any(_x in x for _x in ["train", "val", "test"])
    #     assert all(all(__x in self.all_post_batch_hooks for __x in _x) for _x in x.values())
    #     for _x in x:
    #         self._post_batch_hooks_to_run[_x] = x[_x]

    # control_validation, e.g., can't call validate if it's already running
    # Or what can be called in which state
    # TODO: Define state machine
    # def _define_controls(self):
    #     self._controls = ["train", "validate", "test", "reset",
    #                       "anneal_lr", "set_params", "pause", "abort_current_loop",
    #                       "resume", "start", "stop", "destroy"]
    #     assert all(x in self.__class__.__dict__ for x in self._controls)
    #     assert all(callable(x) for x in self.controls.values())

    # TODO: A lot of these controls and methods which depend on params will
    #       have to be rewritten.
    # TODO: multiplier can be a trainer_param
    # FIXME: Annealing may depend on extra_metrics
    # TODO: Annealing can be an external function like CheckFunc
    def anneal_lr(self, multiplier=.9):
        self._logi("Annealing Learning Rate")
        check_losses = [l[2] for l in self.losses if l[0] == self.save_on]
        if len(check_losses) >= 2:
            delta = check_losses[-2] - check_losses[-1]
            if delta < .01 * check_losses[-2]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= multiplier
                self._logi("Annealing...")

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
        self._max_epochs = params._max_epochs
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
        """If `iterations` exists in self._trainer_params, then we do iterations only
        training and loop_type is set to iterations, else we do standard epoch
        wise training

        if `self._abort_current` flag is set then current main loop is
        aborted. The same flag is also watched for in `self._epoch_runner`.

        post_epoch_hooks are NOT run after an abort.

        :returns: None
        :rtype: None

        """
        if "iterations" in self._trainer_params["training_steps"]:
            loop_type = "iterations"
        else:
            loop_type = "epoch"
        self._logd(f"Beginning training. Loop type is {loop_type}.")
        if loop_type == "iterations":
            self._logd(f"Total number of iterations is {self._max_iterations}")
            self._logd(f"Will run hooks after {self._hooks_run_iter_frequency} iterations")
            while self.iterations < self.max_iterations:
                self._epoch_runner.reset()
                # NOTE: run for self._hooks_run_iter_frequency
                self._epoch_runner.run_train(self.train_step_func, self.train_loader,
                                             loop_type, self._hooks_run_iter_frequency)
                if self._current_aborted:
                    self._logd("Aborting training")
                    return
                self._run_post_epoch_hooks()
                self._iterations += self._hooks_run_iter_frequency
        else:
            self._logd(f"Total number of batches is {len(self.train_loader)}")
            while self.epoch < self._max_epochs:
                self._epoch_runner.reset()
                self._epoch_runner.run_train(self.train_step_func, self.train_loader,
                                             loop_type)
                if self._current_aborted:
                    self._logd("Aborting training")
                    return
                self._run_post_epoch_hooks()
                self.epoch += 1
        self._logi('finished training')

    def validate(self, runner):
        self._logd(f"Validating with {runner.name}")
        runner.run_val(self.val_step_func, self.val_loader)

    def test(self, runner):
        self._logd(f"Testing with {runner.name}")
        runner.run_test(self.test_step_func, self.test_loader)
    # END: Training Steps

    # START: Stateless Functions
    # DONE: There should be a separate definition of "steps" there where it
    #       could be {train, val, test} or simply iterations
    #       NOTE: Now iterations are also handled.
    def _log_metrics(self):
        if "iterations" in self._trainer_params["training_steps"]:
            update_key = self.iterations / self._hooks_run_iter_frequency
            key_name = "iterations chunk"
        else:
            update_key = self.epoch
            key_name = "epoch"
        log_func = self._logd
        for step in self._metrics:
            if getattr(self, step + "_loader"):
                _log_metrics_for_step(step, key_name, getattr(self, step + "_loader"),
                                      self._metrics[step], update_key, log_func)
            else:
                self._logd(f"No dataloader for {step}")

    # FIXME: TRAINING_STEPS
    # NOTE: For this a sample function has to be defined
    def _log_samples(self, fraction=0.01):
        """For a few randomly selected datapoints, log the datapoint_name and
        corresponding model output
        """
        if "iterations" in self._trainer_params["training_steps"]:
            raise NotImplementedError
        for step in self._trainer_params["training_steps"]:
            dataset = getattr(self, step + "_loader").dataset
            loader = get_proxy_dataloader(dataset,
                                          self._dataloader_params[step],
                                          10,  # seems like a good number
                                          self.logger)
            step_func = getattr(self, step + "_step_func")
            # reset, launch each in a separate thread, wait for finish
            # CHECK: Is this a good idea? Maybe separate runner from epoch
            getattr(self._epoch_runner, "run_" + step)(step_func, loader, True)

    def update_metrics_post_epoch_hook(self):
        """Update the metrics being recorded
        :returns: None
        :rtype: None
        """
        self._logd("Updating the metrics")
        if "iterations" in self._trainer_params["training_steps"]:
            update_key = self.iterations / self._hooks_run_iter_frequency
        else:
            update_key = self.epoch
        for step in self._metrics:
            metric_names = self._metrics[step]
            self._metrics[step]["num_datapoints"][update_key] =\
                self._epoch_runner.total_samples[step]
            for m in metric_names:
                all_vals = [x[3] for x in self._epoch_runner.batch_vars
                            if x[0] == step and x[2] == m]
                if len(all_vals):
                    self._metrics[step][m][update_key] = np.mean(all_vals)

    # TODO: I should log some image names and output text also
    #       That should be there in _log_samples
    def log_post_epoch_hook(self):
        """Summarizes and log the metrics/losses etc post epoch
        items_to_log_dict can be accessed and modified by the user

        :returns: None
        :rtype: None

        """
        self._logi("Running post epoch log hook")
        # But these are certain transformations I'm doing to metrics
        for k, v in self._items_to_log_dict.items():
            getattr(self, "_log_" + k)()

    def val_post_epoch_hook(self):
        self.validate_post_epoch_hook(self)

    def validate_post_epoch_hook(self):
        self._logd("Running post epoch validate hook")
        if self.val_loader is not None:
            self.validate(self._epoch_runner)
        else:
            self._logi("No val loader. Skipping")

    def test_post_epoch_hook(self):
        self._logd("Running post epoch test hook")
        if (self.epoch+1) % self.test_frequency == 0:
            if self.test_loader is not None:
                self.test(self._epoch_runner)
            else:
                self._logi("No test loader. Skipping")

    def save_history_post_epoch_hook(self):
        self._logd("Running save history state post epoch hook")
        self._save(self._save_path_with_epoch)

    def save_best_post_epoch_hook(self):
        self._logd("Running save best post epoch hook")
        self.check_and_save()

    def save_checkpoint_post_epoch_hook(self):
        self._logd("Running post epoch save hook")
        self._save(self._checkpoint_path)
        self.check_and_save()

    def add_post_epoch_hook(self, hook, name, position, overwrite=False):
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
    # END: Stateless Functions
