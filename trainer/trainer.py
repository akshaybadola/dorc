import re
import os
import sys
import json
import time
import torch
import logging
import psutil
import pynvml
from threading import Thread
import numpy as np
from torch.utils.data import DataLoader

from .epoch import Epoch


def get_backup_num(filedir, filename):
    backup_files = [x for x in os.listdir(filedir) if x.startswith(filename)]
    backup_maybe_nums = [b.split('.')[-1] for b in backup_files]
    backup_nums = [int(x) for x in backup_maybe_nums
                   if any([_ in x for _ in list(map(str, range(10)))])]
    if backup_nums:
        cur_backup_num = max(backup_nums) + 1
    else:
        cur_backup_num = 0
    return cur_backup_num


def gen_file_and_stream_logger(logdir, log_file_name):
    logger = logging.getLogger('default_logger')
    formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not log_file_name.endswith('.log'):
        log_file_name += '.log'
    log_file = os.path.abspath(os.path.join(logdir, log_file_name))
    if os.path.exists(log_file):
        backup_num = get_backup_num(logdir, log_file_name)
        os.rename(log_file, log_file + '.' + str(backup_num))
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def gen_file_logger(logdir, log_file_name):
    logger = logging.getLogger('default_logger')
    formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not log_file_name.endswith('.log'):
        log_file_name += '.log'
    # existing_files = [f for f in os.listdir(logdir) if f.startswith(log_file_name)]
    log_file = os.path.abspath(os.path.join(logdir, log_file_name))
    if os.path.exists(log_file):
        backup_num = get_backup_num(logdir, log_file_name)
        os.rename(log_file, log_file + '.' + str(backup_num))
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def _dump(x):
    return json.dumps(x, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")


class Tag:
    def __init__(self, x):
        self.tag = x
        self._funcs = []

    def __call__(self, f):
        # if self.tag not in f.__dict__:
        #     f.__dict__[self.tag] = True
        #     self._funcs.append(f)
        if f not in self._funcs:
            self._funcs.append(f)
        return f
control = Tag("control")


# Protocol:
# 1. "control" is defined as any method which changes the state of the
#    wrapper, but doesn't require any arguments, therefore doesn't change
#    the attrs of the wrapper
# 2. "update" is any operation that changes the attrs of the wrapper
# 3. "property" is any operation the retrieves an attribute
# TODO: Change the whole "returns" and "expects" paradigm to "requires" and "provides".
# TODO: The trainer should return the controls and properties in a more wholesome way.
#       Currently it's very hacky and will be error prone in the future.
class Trainer:
    """The :class: `Trainer` class is envisioned as
    an interface to any training procedure.
    """
    def __init__(self, model_params, criteria, optimizer, model_defs, update_functions,
                 extra_metrics, trainer_params, data, dataloader_params):
        """Initializes the :class: `Wrapper` object.
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
        self._logger = gen_file_and_stream_logger(self._logdir, "trainer")
        self._logger.info("Initialized logger in %s", os.path.abspath(self._logdir))
        self._logger.info("Savedir is %s", os.path.abspath(self._savedir))
        # check all params here
        self._sanity_check()
        self._init_state_vars()
        self._controls_global = control

    def _init_all(self):
        self._logger.info("Initializing trainer")
        self._init_models()
        self._init_dataloaders()
        self._init_criteria_optimizers()
        self._init_metrics()
        self._init_update_funcs()
        self._init_epoch_runner()

    def _sanity_check(self):
        self._logger.info("Performing Sanity Check")
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
        assert all(isinstance(x, dict)
                   for x in [self._trainer_params, self._criteria_params,
                             self._optimizer_params])
        assert all(len(x) > 0 for x in [self._trainer_params, self._criteria_params,
                                        self._optimizer_params])
        assert all(isinstance(x, dict) and callable(x["function"])
                   for x in self._criteria_params.values())
        assert all(isinstance(x, dict) and callable(x["function"])
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
    # assert anneal_lr_on in some metric
    # check metric decease or increase?

    # TODO: What if there are other keys besides train/val/test
    def _check_data_params(self):
        assert all([x in self._dataloader_params for x in ["train", "val", "test"]])
        assert self._dataloader_params["train"] is not None
        if self._data is None:
            for x in ["train", "val", "test"]:
                if self._dataloader_params[x] is not None:
                    assert "function" in self._dataloader_params[x]
        else:
            assert all([x in self._data for x in ["train", "val", "test"]])
            assert self._data["train"] is not None

    def _set_device(self):
        self._gpus = list(map(int, self._trainer_params["gpus"].split(",")))
        if self._trainer_params["cuda"] and not torch.cuda.is_available():
            self._logger.error("cuda specified but not available. Will run on cpu")
            self._device = torch.device("cpu")
            self._gpus = [-1]
        elif len(self._gpus) == 1 and torch.cuda.is_available():
            self._logger.info("GPU %d detected and specified" % self._gpus[0])
            self._device = torch.device("cuda:%d" % self._gpus[0])
        elif len(self._gpus) > 1 and torch.cuda.is_available():
            self._logger.info("Data parallel specified with gpus %s" % str(self._gpus))
            if torch.cuda.device_count() >= len(self._gpus):
                self._logger.info("%d gpus are available" % torch.cuda.device_count())
                self._device = "parallel"
            else:
                self._logger.error("%d gpus are not available" % torch.cuda.device_count())
                raise AttributeError
        else:
            self._logger.info("cuda not specified. Using cpu")
            self._device = torch.device("cpu")
            self._gpus = [-1]
        torch.cuda.manual_seed(self._trainer_params["seed"])
        for t, v in self._trainer_params.items():
            if t != "gpus":
                self.__dict__[t] = v

    def to_(self, x):
        if self.device == "parallel":
            return x.cuda()
        else:
            return x.to(self.device)

    def _init_state_vars(self):
        self._logger.info("Initializing State Variables")
        self._paused = True
        self._abort = False
        self._post_epoch_hooks_to_run = ["validate", "test", "save", "log"]
        self._set_device()
        if "extra_report" not in self._trainer_params:
            self.extra_report = {}
        self._epoch = 0
        self._init_nvml()

    def _init_nvml(self):
        self._logger.info("Initializing nvml")
        # Assumes torch.cuda devices are of the same order as PCI BUS for
        # getting correct info with pynvml
        if self._gpus[0] != -1:
            pynvml.nvmlInit()
            self._device_handles = {x: pynvml.nvmlDeviceGetHandleByIndex(x) for x in self._gpus}

    def _init_models(self):
        self._logger.info("Initializing Models")
        self.models = {}
        for model_name, model_params in self._model_params.items():
            model = self._model_defs[model_name]["model"]
            if self._device == "parallel":
                self.models[model_name] = model(**model_params).cuda()
            else:
                self.models[model_name] = model(**model_params).to(self._device)

    def _init_dataloaders(self):
        self._logger.info("Initializing Dataloaders")
        for loader, params in self._dataloader_params.items():
            if loader == "train":
                if self._data is None:
                    self.train_loader = params["function"](**params["function_args"])
                else:
                    self.train_loader = DataLoader(self._data["train"], **params)
            elif loader == "val":
                if params:
                    if self._data is None:
                        self.val_loader = params["function"](**params["function_args"])
                    else:
                        self.val_loader = DataLoader(self._data["val"], **params)
                else:
                    self._logger.info("No Val loader. Will not do validation")
                    self.val_loader = None
            elif loader == "test":
                if params:
                    if self._data is None:
                        self.test_loader = params["function"](**params["function_args"])
                    else:
                        self.test_loader = DataLoader(self._data["test"], **params)
                else:
                    self._logger.info("No Test loader. Will not do testing")
                    self.test_loader = None

    def _init_criteria_optimizers(self):
        self._logger.info("Initializing Optimizers and Criteria")
        self.criteria = {}
        self.optimizers = {}
        for k, v in self._criteria_params.items():
            self.criteria[k] = v["function"](**v["params"])
        for k, v in self._optimizer_params.items():
            model_name = [x for x, y in self._model_defs.items()
                          if y["optimizer"] == k][0]
            self.optimizers[k] = v["function"](self.models[model_name].parameters(), **v["params"])

    # TODO: Should I use namedtuple instead?
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
        self._logger.info("Initializing Metrics")
        self._metrics = {}
        for x in ["train", "val", "test"]:
            if self._dataloader_params[x] is not None:
                self._metrics[x] = dict((l[1], {}) for l in self._update_functions[x].returns
                                        if l[0] == "metric")
                # CHECK: Why's samples in train metrics?
                self._metrics[x]["samples"] = {}
                if x in self._extra_metrics:
                    for k in self._extra_metrics[x].keys():
                        self._metrics[x][k] = {}
                        if self._extra_metrics[x][k]["when"] == "batch":
                            retvals = [_[1] for _ in self._update_functions[x].returns]
                            assert all(_ in retvals for _ in self._extra_metrics[x][k]["inputs"]),\
                                "failed on batch %s, %s" % (x, k)
                        elif self._extra_metrics[x][k]["when"] == "epoch":
                            vals = [*self.__dict__.keys(),
                                    *[_[1] for _ in self._update_functions[x].returns], "epoch"]
                            assert all(s in vals for s in self._extra_metrics[x][k]["inputs"]
                                       if isinstance(s, str)), "failed on epoch %s, %s" % (x, k)
                            assert all(all(_d in self.__dict__[d[0]].keys() for _d in d[1])
                                       for d in self._extra_metrics[x][k]["inputs"]
                                       if isinstance(d, tuple)), "failed on tuple %s, %s" % (x, k)
                else:
                    self._extra_metrics[x] = {}

    def _init_update_funcs(self):
        self._logger.info("Initializing Update Functions")
        for k, v in self._update_functions.items():
            if k == "train":
                self._train_step = self._update_functions["train"]
            elif k == "val":
                self._val_step = self._update_functions["val"]
            elif k == "test":
                self._test_step = self._update_functions["test"]

    def _init_epoch_runner(self):
        self._logger.info("Initializing Epoch Runner")
        self._epoch_runner = Epoch(self, self.extra_report)

    # TODO: The case of saving when there's only iterations and no epochs isn't there.
    # TODO: How to resolve arbitrary callables being saved? Can they resume?
    def _save(self, best=False):
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
        model_names = "_".join(self.models.keys())
        save_name = os.path.join(self._savedir, "_".join([str(self._unique_id),
                                                          model_names,
                                                          "{:03}".format(self.epoch)]))
        self._logger.debug("model_names is %s" % model_names)
        if best:
            save_name += "_best.pth"
        else:
            save_name += ".pth"
        self._logger.debug("Save Name is %s" % save_name)
        save_state = {}
        save_state["epoch"] = self.epoch
        save_state["models"] = dict((k, v.state_dict()) for k, v in self.models.items())
        save_state["optimizers"] = dict((k, v.state_dict()) for k, v in self.optimizers.items())
        save_state["model_params"] = self._model_params
        save_state["criteria_params"] = self._criteria_params
        save_state["dataloader_params"] = self._dataloader_params
        save_state["trainer_params"] = self._trainer_params
        save_state["metrics"] = self._metrics
        self._logger.info("Saving to %s" % save_name)
        torch.save(save_state, save_name)

    # CHECK: resume and update will change the attrs of the wrapper
    # TODO: Unique Id check
    # TODO: Check if {models, metrics, dataloaders, update_funcs} are resumed correctly as
    #       there may be callables in the saved_state. trainer shouldn't allow callables
    def _resume(self, resume_path):
        saved_state = torch.load(resume_path)
        self.epoch = saved_state["epoch"]
        self._model_params = saved_state["model_params"]
        self._criteria_params = saved_state["criteria_params"]
        self._dataloader_params = saved_state["dataloader_params"]
        self._trainer_params = saved_state["trainer_params"]
        self._sanity_check()
        self._init_state_vars()
        self._init_models()
        self._init_dataloaders()
        self._init_criteria_optimizers()
        self._init_metrics()
        self._init_update_funcs()
        self._init_epoch_runner()
        # update optimizers
        # dict((k, v.state_dict()) for k, v in self.optimizers) = saved_state["optimizers"]
        self._metrics = saved_state["metrics"]
        assert all(k in self.models.keys() for k in saved_state['models'])
        assert all(k in self.optimizers.keys() for k in saved_state['optimizers'])
        for k in self.models:
            self.models[k].load_state_dict(saved_state["models"][k])
        for k in self.optimizers:
            self.optimizers[k].load_state_dict(saved_state["optimizers"][k])
        # TODO: check if loaded correctly
        self._metrics = saved_state["metrics"]
        self.epoch = saved_state['epoch']
        self._logger.info("Resumed successfully")

    def check_and_save(self):
        assert ("when" in self._check_func.requires and
                self._check_func.requires["when"] in ["train", "val", "test"]), "Not sure when to save"
        when = self._check_func.requires["when"]
        assert all(x in self._metrics[when] for x in self._check_func.requires["metrics"]),\
            "self._check_func requirements not fulfilled"
        if self._check_func(self._metrics[when]):
            self._logger.info("Save check returned True.")
            self._save(True)
        else:
            self._logger.info("Save check returned False. Not saving")

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

    def _memory_info(self):
        info = psutil.virtual_memory()
        return {"total": info.total, "used": info.used}

    def _cpu_info(self):
        return {"cpu_count": psutil.cpu_count(), "cpu_util": psutil.cpu_percent()}

    def _gpu_util(self):
        def _get_util(h):
            info = pynvml.nvmlDeviceGetUtilizationRates(h)
            return {"gpu": info.gpu, "memory": info.memory}
        if self._gpus[0] != -1:
            return {gpu_id: _get_util(h) for gpu_id, h in self._device_handles.items()}
        else:
            return None

    # FIXME: resume_best can only be done if an index is kept which keeps
    #        track of what's best.
    # TODO: So basically save all the metrics outside in a separate file
    # TODO: It may be some arbitrary predicate
    def resume_best(self):
        """Resumes from the last best saved checkpoint. By default checks for lowest
        `val_acc`

        :returns: None
        :rtype: None

        """
        self._logger.debug("Trying to resume last best checkpoint %s" % self.best_save)
        if self.best_save:
            self._resume(self.best_save)

    def resume_checkpoint(self):
        self._logger.debug("Trying to resume from checkpoint path %s" % self.checkpoint_path)
        if self.checkpoint_path:
            self._resume(self.checkpoint_path)

    def resume_weights(self, weights):
        if os.path.exists(weights):
            self._load_init_weights(weights)

    def _load_init_weights(self, weights):
        self._logger.warn("Warning! Loading directly to model")
        self.model.load_state_dict(torch.load(weights).state_dict())

    # CHECK if this thing works correctly. There might be a few things I may have missed
    @control
    def reset(self):
        self._logger.info("Resetting")
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
        self._logger.info("Pausing")
        self._paused = True

    @control
    def resume(self):
        self._logger.info("Resuming")
        self._paused = False

    @control
    def start(self):
        self._logger.info("Starting")
        # self._init_all()
        self._paused = False
        Thread(target=self.train).start()

    # What does stop even do?
    @control
    def stop(self):
        self._logger.info("Stopping")
        self.abort_current_loop()
        self.save()
        # listen for commands

    @control
    def destroy(self):
        self._logger.info("Destroying")
        self._logger.info("Does nothing for now")

    # Actually a "force_save", pause and then save
    @control
    def save(self):
        self._logger.info("Saving")
        paused = self.paused
        if not paused:
            self.pause()
        # ensure paused
        while not self._epoch_runner.waiting:
            time.sleep(1)
        # TODO: What if epoch already exists? Overwrite?
        self._save()
        if not paused:
            self.resume()

    @control
    def abort_current_loop(self):
        self._logger.info("Aborting")
        self._paused = False
        self._abort = True

    @property
    def gpus(self):
        return self._gpus

    @property
    def system_info(self):
        return {"gpu_util": self._gpu_util(), "cpu_info": self._cpu_info(),
                "memory": self._memory_info()}

    @property
    def device(self):
        return self._device

    @property
    def props(self):
        return [x for x, y in self.__class__.__dict__.items() if isinstance(y, property)
                if x != "props"]

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
        return dict((x.__name__, x) for x in self._controls_global._funcs)

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

    @property
    def checkpoint_path(self):
        if os.path.exists(os.path.join(self._savedir, "checkpoint.pth")):
            return os.path.join(self._savedir, "checkpoint.pth")
        else:
            return None

    @property
    def updatable_params(self):
        params = {}
        params["model_params"] = self._model_params
        params["trainer_params"] = self._trainer_params
        params["dataloader_params"] = self._dataloader_params
        return params

    # as of now, returns all the dict. encoding is upto the backend
    @property
    def all_params(self):
        save_state = {}
        save_state["epoch"] = self.epoch
        save_state["models"] = dict((k, v.state_dict()) for k, v in self.models.items())
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
        return _dump(self.__dict__)

    # TODO: What about other losses
    @property
    def train_losses(self):
        return dict((k, v) for k, v in self._metrics["train"].items()
                    if k[0] == "loss")

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
        self._logger.info("Annealing Learning Rate")
        check_losses = [l[2] for l in self.losses if l[0] == self.save_on]
        if len(check_losses) >= 2:
            delta = check_losses[-2] - check_losses[-1]
            if delta < .01 * check_losses[-2]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= multiplier
                self._logger.info("Annealing...")

    # FIXME: THIS IS totally broken :-(
    # TODO: params should only be predefined names. As such, the required python
    #       objects must already be available to the wrapper. The search
    #       protocol can be developed later.
    # TODO: Implies reset
    def update(self, params):
        """Update the trainer w.r.t to any of the possible variables.
        `params` must be a dict where the keys can be any of the
        following values:

        model_params, model_defs, criteria, optimizer, update_funcs,
        dataloader_params, trainer_params extra_metrics

        As of now non-serializable updates are not supported, so the both
        key/value pairs must be json-encodeable.

        :param params: :class: `dict`
        :returns: None
        :rtype: None

        """
        self._logger.info("Trying to update")
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
        self._logger.debug("Beginning training")
        self._logger.debug("Total number of batches %d" % len(self.train_loader))
        while self.epoch < self._max_epochs:
            # TODO: Things should get updated in a shared queue after each batch
            # NOTE: Maybe not really required, as only that thread writes to those
            #       variablesthingies.
            # TODO: What if run has to be aborted in the middle?
            #       Ensure that run returns
            self._epoch_runner.reset()
            t = Thread(target=self._epoch_runner.run_train)
            t.start()
            t.join()
            # epoch_loss, epoch_accuracy, total
            # TODO: If abort, pause and await instructions?
            if self._abort:
                self._abort = False
            # Don't run post_epoch_hooks after abort
            else:
                self._run_post_epoch_hooks()
                self.epoch += 1
        self._logger.info('finished training')

    def validate(self):
        self._logger.debug("Validating")
        t = Thread(target=self._epoch_runner.run_val)
        t.start()
        t.join()
        if self._abort:
            self._abort = False
        else:
            self._logger.debug("Aborted validation")
            return
            # TODO: Handle this
        self._logger.info("Finished Validation")

    def test(self):
        self._logger.debug("Testing")
        t = Thread(target=self._epoch_runner.run_test)
        t.start()
        t.join()
        if self._abort:
            self._abort = False
        else:
            self._logger.debug("Aborted Testing")
            return
            # TODO: Handle this
        self._logger.info("Finished Testing")

    # Basically generates a summary and saves to file for all the detailed batch logs
    def _log_post_epoch_hook(self):
        """Summarizes and log the metrics/losses etc post epoch

        :returns: None
        :rtype: None

        """
        self._logger.info("Running post epoch log hook")
        for step in self._metrics:
            metric_names = self._metrics[step]
            self._metrics[step]["samples"][self.epoch] = self._epoch_runner.total_samples[step]
            for m in metric_names:
                # FIXME
                if m != "samples" and m != "perplexity":
                    all_vals = [x[3] for x in self._epoch_runner.batch_vars if x[0] == step and x[2] == m]
                    if len(all_vals):
                        self._metrics[step][m][self.epoch] = np.mean(all_vals)

    def _val_post_epoch_hook(self):
        self._validate_post_epoch_hook(self)

    def _validate_post_epoch_hook(self):
        self._logger.debug("Running post epoch validate hook")
        if self.val_loader is not None:
            self.validate()
        else:
            self._logger.info("No val loader. Skipping")

    def _test_post_epoch_hook(self):
        self._logger.debug("Running post epoch test hook")
        if (self.epoch+1) % self.test_frequency == 0:
            if self.test_loader is not None:
                self.test(self.epoch)
            else:
                self._logger.info("No test loader. Skipping")

    def _save_post_epoch_hook(self):
        self._logger.debug("Running post epoch save hook")
        self._save(self.checkpoint_path)
        self.check_and_save(self._check_func)

    def _run_post_epoch_hooks(self):
        self._logger.debug("Running post epoch hooks")
        all_hooks = self.all_post_epoch_hooks
        hook_prefixes = self.post_epoch_hooks_to_run
        for hook in hook_prefixes:
            all_hooks["_".join(["", hook, "post_epoch_hook"])](self)


class ClassificationTrainStep:
    def __init__(self):
        self.returns = [("metric", "cross_entropy_loss"), ("io", "outputs"), ("io", "labels"),
                        ("var", "total")]

    # DONE: Apply data parallel here
    def _train_step(self, wrp, batch, model_name):
        wrp.optimizers["default"].zero_grad()
        wrp.models[model_name].train()
        if wrp.device == "parallel":
            inputs, labels = batch[0].cuda(), batch[1].cuda()
        else:
            inputs, labels = batch[0].to(wrp.device), batch[1].to(wrp.device)
        outputs = wrp.models[model_name](inputs)
        loss = wrp.criteria["cross_entropy_loss"](outputs, labels)
        loss.backward()
        wrp.optimizers["default"].step()
        return {"cross_entropy_loss": loss.data.item().detach(), "outputs": outputs.detach(),
                "labels": labels.detach(), "total": len(labels)}


class ClassificationTestStep:
    def __init__(self):
        self.returns = [("metric", "cross_entropy_loss"), ("io", "outputs"), ("io", "labels"),
                        ("var", "total")]

    # DONE: Apply data parallel here maybe
    def __call__(self, wrp, batch):
        with torch.no_grad():
            wrp._set_models_eval()
            if wrp.device == "parallel":
                inputs, labels = batch[0].cuda(), batch[1].cuda()
            else:
                inputs, labels = batch[0].to(wrp.device), batch[1].to(wrp.device)
            outputs = wrp.model(inputs)
            loss = wrp.criteria["cross_entropy_loss"](outputs, labels)
        return {"cross_entropy_loss": loss.data.item().detach(), "outputs": outputs.detach(),
                "labels": labels, "total": len(labels)}


# NOTE: Why are these functions doing this requires provides anyway? Is this
#       type checking or error checking? If I want robust adherence to a spec, I
#       should make provides and requires abstract functions. Checkable has
#       "virtual" functions/properties "provides" and "requires" like below. It
#       is to facilitate module interaction. Two components can easily and
#       immediately know if they can work together or not, instead of making
#       assumptions, forcing things and creating subtle difficulties later.
#
#       On the theoretical front, generally functions would be defined by the
#       signatures, leaving the implementation to the programmer. However, I
#       would like to have a deeper understanding of the components so that
#       these things aren't just defined by the signature but also their
#       behaviour. Can static typing help here?
#
#       Adhering to an Interface solves these problems, but what if the
#       interface is needed to be flexible? In the below example, metrics needs
#       to be a list, but there's an additional attribute that needs to be there
#       which is when the function is to be called. Should I make it a property?
#       @property
#       def when(self):
#           return "train_end"
#
#       In CheckAccuracy below the data structure of metrics is implicit, while
#       it actually is checked at CheckFunc. Perhaps a better check can be put
#       there. Except we won't know the type and structure of "metrics" until
#       it's called, which may leak an error later in the code. Perhaps mypy or
#       pyright can help here.  Perhaps I should use type annotations for the
#       functions. They seem like a good idea, especially since they can
#       describe complicated types. To facilitate communication over the network
#       they should be json-serializable also.
class CheckFunc:
    def __init__(self, when):
        """Example metrics:

        {'train': {'loss': {0: 7.294781831594614}, 'samples': {0: 6561}, 'perplexity': {}},
        'val': {'loss': {}, 'samples': {0: 0}, 'perplexity': {}, 'sentence_metrics': {}},
        'test': {'loss': {}, 'samples': {0: 0}, 'perplexity': {}, 'sentence_metrics': {}}}

        :returns: None
        :rtype: None

        """
        assert when in ["train", "val", "test"]
        self._requires = {}
        self._provides = {}

    @property
    def requires(self):
        if not isinstance(self._requires["metrics"], list):
            metrics = self._requires.pop("metrics")
            assert isinstance(metrics, str)
            self._requires["metrics"] = [metrics]
        return self._requires

    @property
    def provides(self):
        return self._provides

    def __call__(self, metrics) -> bool:
        return False


class CheckGreater(CheckFunc):
    def __init__(self, when):
        super().__init__(when)

    def __call__(self, metrics):
        raise NotImplementedError


class CheckGreaterName(CheckGreater):
    def __init__(self, when, name):
        super().__init__(when)
        self._name = name
        self._requires = {"when": when, "metrics": name}

    def __call__(self, metrics):
        vals = [*metrics[self._name].items()]
        if vals:
            vals.sort(key=lambda x: x[0])
            vals = [v[1] for v in vals]
            if vals[-1] > vals[:-1]:
                return True
            else:
                return False
        else:
            return False


class CheckAccuracy(CheckGreaterName):
    def __init__(self, when):
        super().__init__(when, "accuracy")


# Example of using extra metrics
# extra_metrics = {"train": {"accuracy": {"function": accuracy,
#                                         "inputs": ["outputs", "batch[1]"]}},
#                  "val": {"accuracy": {"function": accuracy,
#                                       "inputs": ["outputs", "batch[1]"]}}}
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum()
    return float(correct)/len(predicted)
