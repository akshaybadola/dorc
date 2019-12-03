import ipdb
import re
import os
import copy
import time
import torch
from functools import partial
from threading import Thread
import numpy as np
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader

from .device import init_nvml, gpu_util, cpu_info, memory_info, DeviceMonitor
from .util import get_backup_num, gen_file_and_stream_logger
from .epoch import Epoch
from .components import Models
from .overrides import MyDataLoader
from .helpers import control, prop, extras, ProxyDataset, PropertyProxy


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

# TODO: THE REAL BIG ONE
#       Perhaps, not sure, but I would like to have remote code execution, perhaps
#       from right there in the browser, or via a hook from any text editor, like
#       emacs. I can simply hook my editor or open a simple text editor in the browser,
#       paste the code and send for integration. Issues:
#       1. Where is the patch done and how to convey?
#          - diff can work, but that's not arbitrary exec.
#          - simply add and delete object attributes by name
#            But then any watchers will have to update themselves.
#       2. Effective saves, checks etc. after the code is sent to remote.
#       3. Effective data and reports retrieval after the updates.
#       BUT OF COURSE:
#       It should not allow `pip install`, though even that can be done via exec,
#       but the module reload might not be that easy after pip install. Though, in
#       python, there probably is a way.
#       ONE WAY, of doing that could be, allowing the `trainer` object to stay in memory
#       in a python shell and that it never goes out of scope. After that objects can
#       be attached and detached from the `trainer` asynchronously, though deletion should
#       be avoided.
#       IDEALLY, one should only do patching on model and user provided functions and not
#       really core trainer objects. I think that is doable.

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


class Trainer:
    """The :class: `Trainer` class is envisioned as
    an interface to any training procedure.
    """
    def __init__(self, model_params, criteria, optimizer, model_defs, update_functions,
                 extra_metrics, trainer_params, data, dataloader_params):
        """Initializes the :class: `Trainer` object. This is supposed to be a catch all
        trainer which is robust and easy to train and can generate graphs automatically etc.

        :param model: model which is a :class: `torch.nn.Module`
        :param model_params: model params where (k, v) are (:class: `str` model_name, `list` of model params) :class: `dict`
        :param criteria: `dict` where (k, v) are (`str`, :class: `torch.nn.Module`)
        :param optimizer: `dict` where (k, v) are (`str`, :class: `torch.optim.Optimizer`)
        :param model_init: `dict` where (k, v) are (`str` model_name, :function: returns the initialized model)
        :param train_step: :function: which is called for running each batch forward iteration
        :param trainer_params: TODO
        :param train_loader: a train data loader usually :class: `torch.utils.data.Dataloader`
        :param val_loader: a validation data loader usually :class: `torch.utils.data.Dataloader`
        :param test_loader: a test data loader usually :class: `torch.utils.data.Dataloader`
        :param args: `types.SimpleNamespace` which contains the rest of the arguments

        """
        # DONE: model, train_loader, val_loader should be resettable from the interface
        #       Say, trainer.reset() is called, then the interface should place a hook there
        #       that automatically resets the trainloader and the valloader
        #       Mostly Done.
        # Basic assign parameters
        self._unique_id = "bleh"
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
        self.logger.info("Initialized logger in %s", os.path.abspath(self._logdir))
        self.logger.info("Savedir is %s", os.path.abspath(self._savedir))
        # check all params here
        self._have_resumed = False
        self._sanity_check()
        self._init_state_vars()
        if trainer_params["resume"] or "init_weights" in trainer_params:
            self._init_models()
            self._check_resume_or_init_weights()

    # TODO: Check certain variables after everything is initialized, like
    #       `update_funcs`.
    # TODO: Or in fact any thing that is run fater everything is initilialzed,
    #       it's probably not checked right now.
    #       1. controls, properties
    #       2. update_funcs, datasets, dataloaders
    #          - Let's not be too pedantic. Datasets can have basic checks like
    #            if a tensor is returned or not
    #          - update_funcs should run with a given batch
    #          - I should be able to sample a couple of batches from a dataloader
    #       3. optimizers, extra_metrics
    #          - optimizers and extra_metrics should exist and should be callables
    #          - optimizers
    #       4. dataloader_params conflicts
    #       5. check_func
    def _init_all(self):
        if self._have_resumed:
            self.logger.warn("\"_init_all\" being called after resume!")
        self.logger.info("Initializing trainer")
        self._init_models()
        self._init_dataloaders()
        # self._init_criteria_optimizers()
        self._init_metrics()
        self._init_update_funcs()
        self._init_epoch_runner()
        # self._init_extra_controls()

    def _sanity_check(self):
        self.logger.info("Performing Sanity Check")
        self._check_model_params()  # checks model params and defs both
        self._check_trainer_params()  # checks optimizer and stuff also
        self._check_data_params()     # checks data and dataloaders

    def _check_model_params(self):
        assert isinstance(self._model_params, dict), "_model_params has to be a dict"
        assert len(self._model_params) > 0, "_model_params can't be empty"
        assert all(isinstance(x, dict) for x in self._model_params.values()),\
            "all the _model_params should be dict"
        assert isinstance(self._model_defs, dict),\
            "_model_defs has to be a dict"
        assert len(self._model_defs) > 0, "_model_defs can't be empty"
        assert all(callable(x["model"]) for x in self._model_defs.values()),\
            "_model_defs values have to be callabes"

    # TODO: Standardize the device nomenclature especially for dataparallel later
    def _check_trainer_params(self):
        """Checks trainer params

        :returns: None
        :rtype: None

        """
        # optimizer: function now has to be of type Optimizer. Also criterion
        # has to have attribute forward.
        assert all(isinstance(x, dict)
                   for x in [self._trainer_params, self._criteria_params,
                             self._optimizer_params])
        assert all(len(x) > 0 for x in [self._trainer_params, self._criteria_params,
                                        self._optimizer_params])
        assert all(isinstance(x, dict) and callable(x["function"])
                   and hasattr(x["function"], "forward")
                   for x in self._criteria_params.values())
        assert all(isinstance(x, dict) and issubclass(x["function"], torch.optim.Optimizer)
                   for x in self._optimizer_params.values())
        # TODO: This is no longer relevant
        if "anneal" in self._trainer_params:
            assert all(x in self._trainer_params
                       for x in ["anneal_lr_after", "anneal_lr_factor", "anneal_lr_on"])
        if "test_frequency" not in self._trainer_params:
            self.test_frequency = 5
        assert "gpus" in self._trainer_params
        assert "cuda" in self._trainer_params
        assert "max_epochs" in self._trainer_params
        assert "check_func" in self._trainer_params
        self._max_epochs = self._trainer_params["max_epochs"]
        self._check_func = self._trainer_params["check_func"]
        if not self._have_resumed:
            self.logger.debug("Ignoring resume_params in while resuming")
            assert "init_weights" in self._trainer_params
            assert "resume_weights" in self._trainer_params
        
    # assert anneal_lr_on in some metric
    # check metric decease or increase?

    def _check_resume_or_init_weights(self):
        if ("init_weights" in self._trainer_params and self._trainer_params["init_weights"]):
            assert (not self._trainer_params["resume_best"] and
                    not self._trainer_params["resume_weights"]),\
                    "Cannot initialize from weights and resume from save data"
        if self._trainer_params["init_weights"]:
            self.logger.warn("Warning! Loading weights directly to model")
            load_state = torch.load(self._trainer_params["init_weights"])
            for name in self.models.names:
                self.models.load_model(name, load_state["models"][name])
        elif self._trainer_params["resume"]:  # implies resume from somewhere
            if self._trainer_params["resume_best"]:
                # try to find and resume best weights
                self.logger.error("Resume from best is not yet implemented")
                self._resume_path = None
            elif self._trainer_params["resume_weights"]:
                if os.path.exists(self._trainer_params["resume_weights"]):
                    self._resume_path = self._trainer_params["resume_weights"]
                else:
                    self.logger.warn("Given resume weights do not exist")
                    self._resume_path = None  # set appropriate path
            else:
                if os.path.exists(self._checkpoint_path):
                    self.logger.info("Checkpoint exists. Will resume from there")
                    self._resume_path = self._checkpoint_path
                else:
                    self.logger.info("No checkpoint found. Will train from beginninng")
                    self._resume_path = None
        else:
            # Don't resume
            self._resume_path = None
        if self._trainer_params["resume"] and self._resume_path:
            self.logger.info("Resuming from %s" % self._resume_path)
            self._resume_from_path(self._resume_path)

    # TODO: What if there are other keys besides train/val/test
    def _check_data_params(self):
        """If self._data is None, then data is extracted from the dataloader later.

        :returns: None
        :rtype: None

        """
        assert all([x in self._dataloader_params for x in ["train", "val", "test"]])
        assert self._dataloader_params["train"] is not None
        if self._data is None:
            for x in ["train", "val", "test"]:
                if self._dataloader_params[x] is not None:
                    assert "function" in self._dataloader_params[x],\
                        "dataloader_params for data subset cannot be None if data is None"
        else:
            assert all([x in self._data for x in ["train", "val", "test"]])
            assert self._data["train"] is not None, "Training data cannot be None"

    def _set_device(self):
        self._gpus = list(map(int, self._trainer_params["gpus"].split(",")))
        if self._trainer_params["cuda"] and not torch.cuda.is_available():
            self.logger.error("cuda specified but not available. Will run on cpu")
            self._device = torch.device("cpu")
            self._gpus = [-1]
        elif len(self._gpus) == 1 and torch.cuda.is_available():
            self.logger.info("GPU %d detected and specified" % self._gpus[0])
            self._device = torch.device("cuda:%d" % self._gpus[0])
        elif len(self._gpus) > 1 and torch.cuda.is_available():
            self.logger.info("Data parallel specified with gpus %s" % str(self._gpus))
            if torch.cuda.device_count() >= len(self._gpus):
                self.logger.info("%d gpus are available" % torch.cuda.device_count())
                self._device = "parallel"
            else:
                self.logger.error("%d gpus are not available" % torch.cuda.device_count())
                raise AttributeError
        else:
            self.logger.info("cuda not specified. Using cpu")
            self._device = torch.device("cpu")
            self._gpus = [-1]
        torch.cuda.manual_seed(self._trainer_params["seed"])
        for t, v in self._trainer_params.items():
            if t in self.__class__.__dict__:
                self.logger.warn(f"Tried overwriting attribute {t}! Denied.")
            elif t != "gpus":
                self.__dict__[t] = v

    # def to_(self, x):
    #     if self.device == "parallel":
    #         return x.cuda()
    #     else:
    #         return x.to(self.device)

    def _init_state_vars(self):
        self.logger.info("Initializing State Variables")
        self._paused = True
        self._abort = False
        self._post_epoch_hooks_to_run = ["validate", "test", "save", "log"]
        self._set_device()
        if "extra_report" not in self._trainer_params:
            self.logger.debug("No Extra Reportables")
            self.extra_report = {}
        self._epoch = 0
        self._init_nvml()
        self._temp_runner = SimpleNamespace()
        self._flag_adhoc_func_running = False

    def _init_nvml(self):
        """Initializes the Nvidia monitoring library. It's called by _init_state_vars so
        needn't be called again.

        :returns: None
        :rtype: None

        """
        self.logger.info("Initializing nvml")
        # CHECK: I don't remember how the order is printed.
        # Assumes torch.cuda devices are of the same order as PCI BUS for
        # getting correct info with pynvml
        if self._gpus[0] != -1:
            self._device_handles = init_nvml(self._gpus)
        else:
            self._device_handles = None

    def _init_models(self):
        self.logger.info("Initializing Models, Optimizers and Criteria ")
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
            self._set_device()
            devices = {m: self._device for m in self._model_params}
        for model_name, model_params in self._model_params.items():
            models[model_name] = self._model_defs[model_name]["model"](**model_params)
            optim_name = self._model_defs[model_name]["optimizer"]
            optimizers[model_name] = {"name": optim_name,
                                      "optimizer": self._optimizer_params[optim_name]["function"](
                                          models[model_name].parameters(),
                                          **self._optimizer_params[optim_name]["params"])}
        self.models = Models(models, optimizers, devices, self.gpus, self.logger)
        # NOTE: Old optimizer initialization code
        # for k, v in self._optimizer_params.items():
        #     model_name = [x for x, y in self._model_defs.items()
        #                   if y["optimizer"] == k][0]
        #     self.optimizers[k] = v["function"](self.models[model_name].parameters(),
        #                                        **v["params"])

    # TODO: Check by sampling a few instances from the dataset.
    def _init_dataloaders(self):
        self.logger.info("Initializing Dataloaders")
        for loader, params in self._dataloader_params.items():
            if loader == "train":
                if self._data is None:
                    self.train_loader = params["function"](**params["function_args"])
                else:
                    self.train_loader = DataLoader(self._data["train"], **params)
                if not hasattr(self.train_loader.dataset, "_get_raw"):
                    self.logger.warn("Train dataset doesn't define \"_get_raw\".\
                    Drawing samples from training data will not be available.")
            elif loader == "val":
                if params:
                    if self._data is None:
                        self.val_loader = params["function"](**params["function_args"])
                    else:
                        self.val_loader = DataLoader(self._data["val"], **params)
                else:
                    self.logger.info("No Val loader. Will not do validation")
                    self.val_loader = None
                if self.val_loader and not hasattr(self.val_loader.dataset, "_get_raw"):
                    self.logger.warn("Validation dataset doesn't define \"_get_raw\".\
                    Drawing samples from validation data will not be available.")
            elif loader == "test":
                if params:
                    if self._data is None:
                        self.test_loader = params["function"](**params["function_args"])
                    else:
                        self.test_loader = DataLoader(self._data["test"], **params)
                else:
                    self.logger.info("No Test loader. Will not do testing")
                    self.test_loader = None
                if self.test_loader and not hasattr(self.test_loader.dataset, "_get_raw"):
                    self.logger.warn("Test dataset doesn't define \"_get_raw\".\
                    Drawing samples from test data will not be available.")

    # def _init_criteria_optimizers(self):
    #     self.logger.info("Initializing Optimizers and Criteria")
    #     self.criteria = {}
    #     self.optimizers = {}
    #     for k, v in self._criteria_params.items():
    #         self.criteria[k] = v["function"](**v["params"])
    #     for k, v in self._optimizer_params.items():
    #         model_name = [x for x, y in self._model_defs.items()
    #                       if y["optimizer"] == k][0]
    #         self.optimizers[k] = v["function"](self.models[model_name].parameters(),
    #                                            **v["params"])

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
        self.logger.info("Initializing Metrics")
        self._metrics = {}
        for x in ["train", "val", "test"]:
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
                                "failed on batch %s, %s" % (x, k)
                        elif self._extra_metrics[x][k]["when"] == "epoch":
                            # NOTE: validation with inputs
                            vals = [*self.__dict__.keys(),
                                    *[_[1] for _ in self._update_functions[x].returns], "epoch"]
                            assert all(s in vals for s in self._extra_metrics[x][k]["inputs"]
                                       if isinstance(s, str)), "failed on epoch %s, %s" % (x, k)
                            # FIXME: Only "models" as tuple allowed
                            for _x in self._extra_metrics[x][k]["inputs"]:
                                if isinstance(_x, tuple):
                                    assert _x[0] == "models" and\
                                        all(_ in self.models.names for _ in _x[1]),\
                                        "Required model not in self.models"
                            # assert all(all(_d in self.__dict__[d[0]].keys() for _d in d[1])
                            #            for d in self._extra_metrics[x][k]["inputs"]
                            #            if isinstance(d, tuple)), "failed on tuple %s, %s" % (x, k)
                else:
                    self._extra_metrics[x] = {}

    def _init_update_funcs(self):
        self.logger.info("Initializing Update Functions")
        for k, v in self._update_functions.items():
            if k == "train":
                self._train_step = self._update_functions["train"]
            elif k == "val":
                self._val_step = self._update_functions["val"]
            elif k == "test":
                self._test_step = self._update_functions["test"]

    def _init_epoch_runner(self):
        class Signals(object, metaclass=PropertyProxy):
            trainer = self
        device_monitor = DeviceMonitor(self._device_handles)
        self.logger.info("Initializing Epoch Runner")
        self._epoch_runner = Epoch({"metrics": self._metrics, "extra_metrics": self._extra_metrics},
                                   Signals, device_monitor, self.extra_report)

    # TODO: Functions like this should return a json like form to update to the server
    #       For each such endpoint, there should be a "endpoint_params" endpoint which
    #       sends the required json_data format which is to be sent with the request
    #       Which should then be presented as a table to the user.
    #       There should be NO NESTING.
    @extras
    def call_adhoc_run(self, data):
        if not any(x in data for x in ["train", "val", "test"]):
            return False, "Unknown dataset"
        else:
            for x in data:
                return self.try_call_adhoch_func_with_data(x, data[x])

    def try_call_adhoch_func_with_data(self, step, params):
        # {"metrics": [list_of_metrics], "epoch": num_or_"current", fraction_of_dataset: 0 < x < 1,
        # "device", one_of_gpus}
        # maybe: {"report_function": <<function>>}
        # Or maybe device can be automatically determined
        # NOTE: Samples should be captured by default
        self.logger.warn("Ignoring \"epoch\" for now")
        try:
            iter(params)
        except TypeError:
            return False, "Incorrent format"
        if not all(x in params for x in ["metrics", "epoch", "fraction"]):
            return False, "Incorrent parameters"
        elif not (params["metrics"] != "all") or (not all(x in self._metrics[step]
                                                          for x in params["metrics"])):
            import ipdb; ipdb.set_trace()
            self.logger.debug(f'metrics given {params["metrics"]}')
            return False, "Unknown metrics or incorrect format given"

        # FIXME: WTF is self.checkpoints anyway? It has to be a dict now
        # elif not params["epoch"] in self.checkpoints:
        #     return False, "Checkpoint for epoch doesn't exist"
        elif params["fraction"] > 1 or params["fraction"] <= 0:
            return False, "Incorrect fraction"
        else:
            self.pause()
            while not self.paused:
                time.sleep(10)
            params["step"] = step
            t = Thread(target=self.call_adhoc_func, args=[params])
            if not self._flag_adhoc_func_running:
                self._flag_adhoc_func_running = True
                t.start()
                return True, "Running the given adhoc function"
            else:
                return False, "Another adhoc function is still running"
            # 1. If training, then pause trainer,
            # 2. run the requested adhoc function in a thread
            # 3. Result is stored in _adhoc_func_result
            # 4. _adhoc_func_result is checked for uniqueness
            # 5. Multiple funcs (upto 3) should be able to run, and they should be trackable
            # 6. Auto resource allocation for funcs
            # 7. If required, save state to disk before doing so

    def call_adhoc_func(self, params):
        # have to call epoch runner with specific metrics (in case some are too expensive)
        # For now call all metrics but it's fraction of the data anyway.
        step = params.pop("step")
        step_loader = getattr(self, step + "_loader")
        if "seed" in params:
            np.random.seed(params["seed"])
        indices = np.random.choice(len(step_loader.dataset),
                                   int(len(step_loader.dataset) * params["fraction"]))
        _proxy_dataset = ProxyDataset(step_loader.dataset, indices)
        if hasattr(step_loader.dataset, "_get_raw"):
            _proxy_dataset._get_raw = step_loader.dataset._get_raw
            temp_loader = MyDataLoader(_proxy_dataset, return_raw=True,
                                       **self._dataloader_params[step])
            self.logger.info(f"{step} dataset has \"_get_raw\"\
            Drawing samples from test data is available!")
        else:
            temp_loader = MyDataLoader(_proxy_dataset, **self._dataloader_params[step])
            self.logger.warn(f"{step} dataset doesn't define \"_get_raw\".\
            Drawing samples from test data will not be available.")
        models = {}
        optimizers = {}
        devices = {}
        for model_name, model_params in self._model_params.items():
            models[model_name] = self._model_defs[model_name]["model"](**model_params)
            optim_name = self._model_defs[model_name]["optimizer"]
            optimizers[model_name] = {"name": optim_name,
                                      "optimizer": self._optimizer_params[optim_name]["function"](
                                          models[model_name].parameters(),
                                          **self._optimizer_params[optim_name]["params"])}
            devices[model_name] = self._device
        temp_models = Models(models, optimizers, devices, self.gpus, self.logger)
        step_func = partial(self._update_functions[step], temp_models, self.criteria)

        # TODO: Load from checkpoint like this
        # _models.load(self._get_checkpoint(epoch)["models"])
        # TODO: Maybe let model also be altered, checkpoint of course should be
        temp_models.load(self.models.dump())  # replicate
        # TODO: Put extra metrics here

        class Signals:
            paused = False
            aborted = False
        device_monitor = DeviceMonitor(self._device_handles)
        self.logger.debug(f"params, {step}, {params}")
        temp_runner = Epoch({"metrics": {step: params["metrics"]}, "extra_metrics": {}},
                            Signals, device_monitor, self.extra_report)
        temp_runner.reset()
        temp_runner.metrics[step].append("raw")
        temp_runner.metrics[step].append("predictions")
        temp_runner.metrics[step].append("labels")
        self._temp_runner = temp_runner
        self.logger.debug(f"starting {self._temp_runner}")
        self._temp_runner.logger = self.logger
        if hasattr(step_loader.dataset, "_get_raw"):
            getattr(temp_runner, "run_" + step)(step_func, temp_loader, True)
        else:
            getattr(temp_runner, "run_" + step)(step_func, temp_loader)
        # report function only takes in targets and predictions
        Thread(target=self._check_adhoc_run).start()

    # TODO: Fix for new adhoc run
    def _check_adhoc_run(self):
        while self._temp_runner.running:
            time.sleep(1)
        self._flag_adhoc_func_running = False

    @extras
    def report_adhoc_run(self):
        if not hasattr(self._temp_runner, "running"):
            return "Adhoc function was never initialized"
        elif self._temp_runner.running:
            return "Adhoc function is still running"
        else:
            def _same(a, b):
                self.logger.debug(f"_same, {b is None}")
                if b is not None and a[1] == b[1]:
                    self.logger.debug(f"_same, {b[1], a[1]}")
                    return True
                else:
                    return False
            output = []
            temp_targets = None
            temp_predictions = None
            for x in self._temp_runner.batch_vars:
                if x[2] == "predictions":
                    temp_predictions = x
                    if _same(temp_predictions, temp_targets):
                        # self.logger.debug(self.report_function(temp_predictions[-1], temp_targets[-1]))
                        output.append((x[0], x[1], "predictions_targets",
                                       self.report_function(temp_predictions[-1], temp_targets[-1])))
                        temp_predictions = None
                        temp_targets = None
                elif x[2] in {"labels", "targets"}:
                    temp_targets = x
                    if _same(temp_targets, temp_predictions):
                        # self.logger.debug(self.report_function(temp_predictions[-1], temp_targets[-1]))
                        output.append((x[0], x[1], "predictions_targets",
                                       self.report_function(temp_predictions[-1], temp_targets[-1])))
                        temp_predictions = None
                        temp_targets = None
                else:
                    output.append(x)
            return output

    # TODO: How to resolve arbitrary callables being saved? Can they resume?
    #       In fact like I mentioned earlier, arbitrary callables shouldn't be allowed
    #       in saved states.
    def _save(self, save_path=None, best=False):
        # if isinstance(save_name_or_dict, dict):
        #     save_name = '__'.join(['_'.join([a, str(b)])
        #                            for a, b in save_name_or_dict.items()]) + '.pth'
        # elif isinstance(save_name_or_dict, str):
        #     save_name = save_name_or_dict if save_name_or_dict.endswith('.pth') else save_name_or_dict + '.pth'
        # else:
        #     raise AttributeError
        #
        # Save name is internal now
        # wrapper should have a unique id
        if not save_path:
            save_path = self._save_path
        if best:
            if not save_path.endswith(".pth"):
                save_path += "_best.pth"
            else:
                save_path = save_path.replace(".pth", "") + "_best.pth"
        elif not save_path.endswith(".pth"):
            save_path += ".pth"
        self.logger.debug("Trying to save to_names is %s" % save_path)
        save_state = {}
        save_state["epoch"] = self.epoch
        # save_state["models"] = dict((k, v.state_dict()) for k, v in self.models.items())
        save_state["models"] = self.models.dump()
        # save_state["optimizers"] = dict((k, v.state_dict()) for k, v in self.optimizers.items())
        save_state["model_params"] = self._model_params
        save_state["criteria_params"] = self._criteria_params
        save_state["dataloader_params"] = {x: {a: b.__qualname__ if a == "collate_fn" else b
                                               for a, b in y.items()}
                                           for x, y in self._dataloader_params.items()}
        if any(["collate_fn" in y for x, y in save_state["dataloader_params"].items()]):
            self.logger.warn("collate_fn will not be saved")
        save_state["trainer_params"] = self._trainer_params
        save_state["metrics"] = self._metrics
        self.logger.info("Saving to %s" % save_path)
        torch.save(save_state, save_path)

    # CHECK: resume and update will change the attrs of the trainer
    #        Perhaps some tests here.
    # TODO: Unique Id check
    # TODO: Check if {models, metrics, dataloaders, update_funcs} are resumed correctly as
    #       there may be callables in the saved_state. trainer shouldn't allow callables
    def _resume_from_path(self, resume_path):
        self._have_resumed = True
        saved_state = torch.load(resume_path)
        self.epoch = saved_state["epoch"]
        self._model_params = saved_state["model_params"]
        self._criteria_params = saved_state["criteria_params"]
        if any(["collate_fn" in y for x, y in saved_state["dataloader_params"].items()]):
            self.logger.warn("collate_fn will not be restored")
        for x in self._dataloader_params:
            self._dataloader_params[x].update({a: b for a, b in self._dataloader_params[x].items()
                                               if a != "collate_fn"})
        self._trainer_params = saved_state["trainer_params"]
        self._sanity_check()
        # CHECK: Only if model or model parameters have changed
        self._init_models()
        self._init_dataloaders()
        # Only if criteria and/or optimizer have changed.  In fact, there might
        # be a mismatch if criteria change suddenly as the model has changed,
        # but resume_weights should not really be concerned about that, at
        # least.
        # self._init_criteria_optimizers()
        # Only if new metrics are added and even then only update metrics
        self._init_metrics()
        # Only if update_funcs are changed.
        # In fact, this is not in saved state
        # self._init_update_funcs()
        self._init_epoch_runner()

        # NOTE: This is checked in Models now
        # assert all(k in self.models.keys() for k in saved_state['models'])
        # assert all(k in self.optimizers.keys() for k in saved_state['optimizers'])
        # for k in self.models:
        #     self.models[k].load_state_dict(saved_state["models"][k])
        # for k in self.optimizers:
        #     self.optimizers[k].load_state_dict(saved_state["optimizers"][k])
        self.models.load(saved_state["models"])
        # TODO: check if loaded correctly
        for k in self._metrics.keys():
            assert k in saved_state['metrics']
            for _k in self._metrics[k]:
                assert _k in saved_state['metrics'][k]
        self._metrics = copy.deepcopy(saved_state["metrics"])
        self.epoch = saved_state['epoch']
        self.logger.info("Resumed successfully")

    def check_and_save(self):
        assert ("when" in self._check_func.requires and
                self._check_func.requires["when"]
                in ["train", "val", "test"]), "Not sure when to save"
        when = self._check_func.requires["when"]
        assert all(x in self._metrics[when] for x in self._check_func.requires["metrics"]),\
            "self._check_func requirements not fulfilled"
        if self._check_func(self._metrics[when]):
            self.logger.info("Save check returned True.")
            self._save(None, True)
        else:
            self.logger.info("Save check returned False. Not saving")

    # control_validation, e.g., can't call validate if it's already running
    # Or what can be called in which state
    # TODO: Define state machine
    # def _define_controls(self):
    #     self._controls = ["train", "validate", "test", "reset",
    #                       "anneal_lr", "set_params", "pause", "abort_current_loop",
    #                       "resume", "start", "stop", "destroy"]
    #     assert all(x in self.__class__.__dict__ for x in self._controls)
    #     assert all(callable(x) for x in self.controls.values())
    def _define_controls(self):
        pass

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
    #     self.logger.debug("Trying to resume last best checkpoint %s" % self.best_save)
    #     if self.best_save:
    #         self._resume_path = self.best_save

    # def resume_checkpoint(self):
    #     self.logger.debug("Trying to resume from checkpoint path %s" % self._checkpoint_path)
    #     if os.path.exists(self._checkpoint_path):
    #         self._resume(self._checkpoint_path)
    #     else:
    #         self.logger.debug("No checkpoint found. Will train from beginning %s" %
    #                            self._checkpoint_path)

    # def resume_weights(self, weights):
    #     if os.path.exists(weights):
    #         self._load_init_weights(weights)

    # CHECK if this thing works correctly. There might be a few things I may have missed
    # TODO: For any worker loop which returns an error, the next
    #       one should pause or halt or something.
    @control
    def reset(self):
        self.logger.info("Resetting")
        backup_num = get_backup_num(".", self._savedir)
        if os.path.exists(self._savedir):
            os.rename(self._savedir, self._savedir + "." + str(backup_num))
            os.mkdir(self._savedir)
        if os.path.exists(self.logdir):
            os.rename(self.logdir, self.logdir + "." + str(backup_num))
            os.mkdir(self.logdir)
        self._init_criteria_optimizers()

    @control
    def pause(self):
        self.logger.info("Pausing")
        self._paused = True

    @control
    def resume(self):
        self.logger.info("Resuming")
        self._paused = False

    @control
    def start(self):
        self.logger.info("Starting")
        self._paused = False
        Thread(target=self.train).start()

    # What does stop even do?
    @control
    def stop(self):
        self.logger.info("Stopping")
        self.abort_current_loop()
        self.save()
        # listen for commands

    @control
    def destroy(self):
        self.logger.info("Destroying")
        self.logger.info("Does nothing for now")

    # Actually a "force_save", pause and then save
    @control
    def save(self):
        self.logger.info("Saving")
        paused = self.paused
        if not paused:
            self.pause()
        # ensure paused
        while not self._epoch_runner.waiting:
            time.sleep(1)
        self.logger.warn("Trying force save")
        self._save(self._save_path + "_force")
        # TODO: Keep track of self._abort
        if not paused and not self._abort:
            self.resume()

    # CHECK: How do I just run eval right at the beginning?
    # TODO: There should be a control to set current_loop to ["train", "val", "test"]
    @control
    def abort_current_loop(self):
        self.logger.info("Aborting")
        self._paused = False
        self._abort = True

    @property
    def logger(self):
        return self._logger

    @property
    def logfile(self):
        return open(self._logfile).read()

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

    @property
    def train_step(self):
        return partial(self._train_step, self.models, self.criteria)

    @property
    def val_step(self):
        return partial(self._val_step, self.models, self.criteria)

    @property
    def test_step(self):
        return partial(self._val_step, self.models, self.criteria)

    # exclude properties beginning with _
    @property
    def props(self):
        return [x for x, y in self.__class__.__dict__.items()
                if isinstance(y, property) and
                x != "props" and
                (x == "_extras" or not x.startswith("_"))]

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
    def controls(self):
        """Which of the functions can be accessed via the API

        :returns: API exports
        :rtype: list

        """
        # return dict((x, self.__getattribute__(x)) for x in self._controls)
        return dict((x.__name__, x) for x in control.members)

    @property
    def _extras(self):
        return dict((x.__name__, x) for x in extras.members)

    # CHECK
    # Why abort the running loop? The wrapper itself is paused?
    @property
    def aborted(self):
        """returns whether the current loop was aborted or not

        :returns: abort state
        :rtype: bool

        """
        return self._abort

    @property
    def current_run(self):
        if "_epoch_runner" not in self.__dict__:
            return "None"
        else:
            return self._epoch_runner.current_loop

    @property
    def paused(self):
        return self._paused

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
    def _save_path(self):
        model_names = "_".join(self.models.names)
        save_name = os.path.join(self._savedir, "_".join([str(self._unique_id),
                                                          model_names,
                                                          "{:03}".format(self.epoch)]))
        return save_name

    @property
    def _checkpoint_path(self):
        # model_names = "_".join(self.models.names)
        # save_name = "_".join([str(self._unique_id), model_names, "checkpoint"])
        return os.path.join(self._save_path + "_checkpoint" + ".pth")

    # TODO: Allow extra_metrics, update_funcs and any other params to be updated
    @property
    def updatable_params(self):
        params = {}
        params["model_params"] = self._model_params
        params["trainer_params"] = self._trainer_params
        params["dataloader_params"] = self._dataloader_params
        return params

    # as of now, returns all the dict. encoding is upto the backend
    # TODO: Tag each property or dict with "param", so it can be automatically viewed
    #       i.e., for_each x in self.__dict__, if self.__dict__[x]._tag == "param", then is_param
    #       This can be accomplished with the `prop` function above. prop simply tags
    #       whatever property that exists and that can be exported via @property
    #       as an observable property
    @property
    def all_params(self):
        save_state = {}
        save_state["epoch"] = self.epoch
        # save_state["models"] = dict((k, v.state_dict()) for k, v in self.models.items())
        save_state["optimizers"] = dict((k, v.state_dict()) for k, v in self.optimizers.items())
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
    def metrics(self):
        return self._metrics

    # TODO: Define what is a sample correctly
    # TODO: Get random training samples also
    @property
    def val_samples(self):
        return dict((k, v) for k, v in self._metrics["val"].items()
                    if k[0] == "sample")

    @property
    def all_post_epoch_hooks(self):
        return dict((x, y) for (x, y) in self.__class__.__dict__.items()
                    if x.endswith("post_epoch_hook"))

    @property
    def post_epoch_hooks_to_run(self):
        return self._post_epoch_hooks_to_run

    # where are the hooks run?
    @post_epoch_hooks_to_run.setter
    def post_epoch_hooks_to_run(self, x):
        assert any(_x in x for _x in ["train", "val", "test"])
        assert all(all(__x in self.all_post_batch_hooks for __x in _x) for _x in x.values())
        for _x in x:
            self._post_batch_hooks_to_run[_x] = x[_x]

    # TODO: A lot of these controls and methods which depend on params will
    #       have to be rewritten.
    # TODO: multiplier can be a trainer_param
    # FIXME: Annealing may depend on extra_metrics
    # TODO: Annealing can be an external function like CheckFunc
    def anneal_lr(self, multiplier=.9):
        self.logger.info("Annealing Learning Rate")
        check_losses = [l[2] for l in self.losses if l[0] == self.save_on]
        if len(check_losses) >= 2:
            delta = check_losses[-2] - check_losses[-1]
            if delta < .01 * check_losses[-2]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= multiplier
                self.logger.info("Annealing...")

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

        :param params: :class: `dict`
        :returns: None
        :rtype: None

        """
        self.logger.info("Trying to update")
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
        self._set_device()
        # if torch.cuda.is_available():
        #     self._device = torch.device("cuda:%d" % params.gpu)
        self.save_on = params.save_on
        self.save_var = params.save_var

    # # FIXME: Optimizer is initialized here or before it?
    # #        It can possibly be based on name or a custom func
    # # TODO: This thing doesn't do anything now
    # def _get_optimizer(self, name):
    #     if name.lower() == 'sgd':
    #         return torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
    #     elif name.lower() == 'adam':
    #         return torch.optim.Adam(self.model.parameters())

    # Train validate and stuff are relatively fine
    # TODO: custom reportables
    def train(self):
        """Handles training.

        :returns: None
        :rtype: None

        """
        self.logger.debug("Beginning training")
        self.logger.debug("Total number of batches %d" % len(self.train_loader))
        while self.epoch < self._max_epochs:
            # TODO: Things should get updated in a shared queue after each batch
            # NOTE: Maybe not really required, as only that thread writes to those
            #       variablesthingies.
            # TODO: What if run has to be aborted in the middle?
            #       Ensure that run returns
            # TODO: What if the thread dies in the middle?
            self._epoch_runner.reset()
            t = Thread(target=self._epoch_runner.run_train,
                       args=[self.train_step, self.train_loader])
            t.start()
            t.join()
            # epoch_loss, epoch_accuracy, total
            # TODO: If abort, pause and await instructions?
            if self._abort:
                self.logger.debug("Aborted training")
                self._abort = False
            # Don't run post_epoch_hooks after abort
            else:
                # TODO: CRITICAL post_epoch_hooks have to be run correctly as I'm using
                #       the same epoch_runner, if it's reset without gathering data from
                #       run_train, then all the data will be lost
                self._run_post_epoch_hooks()
                self.epoch += 1
        self.logger.info('finished training')

    def validate(self):
        self.logger.debug("Validating")
        t = Thread(target=self._epoch_runner.run_val,
                   args=[self.val_step, self.val_loader])
        t.start()
        t.join()
        if self._abort:
            self.logger.debug("Aborted validation")
            self._abort = False
            # TODO: Handle this
        else:
            self.logger.info("Finished Validation")

    def test(self):
        self.logger.debug("Testing")
        t = Thread(target=self._epoch_runner.run_test,
                   args=[self.test_step, self.test_loader])
        t.start()
        t.join()
        if self._abort:
            self.logger.debug("Aborted Testing")
            self._abort = False
            # TODO: Handle abort here
        else:
            self.logger.info("Finished Testing")

    # Basically generates a summary and saves to file for all the detailed batch logs
    def _log_post_epoch_hook(self):
        """Summarizes and log the metrics/losses etc post epoch

        :returns: None
        :rtype: None

        """
        self.logger.info("Running post epoch log hook")
        for step in self._metrics:
            metric_names = self._metrics[step]
            self._metrics[step]["num_datapoints"][self.epoch] =\
                self._epoch_runner.total_samples[step]
            for m in metric_names:
                # if m != "num_datapoints":
                all_vals = [x[3] for x in self._epoch_runner.batch_vars
                            if x[0] == step and x[2] == m]
                if len(all_vals):
                    self._metrics[step][m][self.epoch] = np.mean(all_vals)

    def _val_post_epoch_hook(self):
        self._validate_post_epoch_hook(self)

    def _validate_post_epoch_hook(self):
        self.logger.debug("Running post epoch validate hook")
        if self.val_loader is not None:
            self.validate()
        else:
            self.logger.info("No val loader. Skipping")

    def _test_post_epoch_hook(self):
        self.logger.debug("Running post epoch test hook")
        if (self.epoch+1) % self.test_frequency == 0:
            if self.test_loader is not None:
                self.test()
            else:
                self.logger.info("No test loader. Skipping")

    def _save_post_epoch_hook(self):
        self.logger.debug("Running post epoch save hook")
        self._save(self._checkpoint_path)
        self.check_and_save()

    # log_train has to run first of all
    def _run_post_epoch_hooks(self):
        self.logger.debug("Running post epoch hooks")
        all_hooks = self.all_post_epoch_hooks
        hook_prefixes = self.post_epoch_hooks_to_run
        for hook in hook_prefixes:
            all_hooks["_".join(["", hook, "post_epoch_hook"])](self)
