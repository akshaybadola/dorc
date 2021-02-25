from typing import List, Dict, Callable, Iterable, Union, Any
from torch.utils.data import Dataset
import numpy as np
from .overrides import MyDataLoader


class Exposes:
    def __init__(self, *args):
        assert all((isinstance(x, str)) for x in args)
        self._args = args

    def __call__(self, fn, *args, **kwargs):
        def new_func(*args, **kwargs):
            return fn(*args, **kwargs)
        new_func.exposes = set()
        for x in self._args:
            assert x in fn.__code__.co_names or x in fn.__code__.co_varnames,\
                f"{x} not available in function {fn}"
            new_func.exposes.add(x)
        # HACK: Big hack
        new_func.__name__ = fn.__name__
        return new_func


class Artefacts:
    def __init__(self):
        self._artefacts: Dict[str, Callable] = {}

    def add(self, x: str, f: Callable, overwrite: bool = False):
        if x not in self._artefacts or overwrite:
            self._artefacts[x] = f
        else:
            raise AttributeError("Item already exists. Use overwrite to update")

    def remove(self, x):
        self._artefacts.pop(x)

    def __iter__(self):
        return iter(self._artefacts.items())

    def __getitem__(self, x):
        return self._artefacts[x]

    def __repr__(self) -> str:
        return str({x: y.__name__ for x, y in self._artefacts.items()})


class Hook:
    """A mapping of functions.

    A :code:`hook` is usually a list of functions with no arguments. Here we augment
    that so that it's a mapping of type :code:`Dict[str, Callable]`. The :class:`Hook`
    is then, a :class:`dict` of :class:`Callable` which may or may not take
    arguments and return nothing.  They can be executed at arbitrary points in a
    codebase with separate concerns.

    We refer to these as named hooks, with :attr:`permanent_items` which cannot
    be changed.

    Args:
        permanent_items: A dictionary of functions which will never be removed from the hook

    """

    def __init__(self, permanent_items: Dict[str, Callable[..., None]]):
        self._funcs: Dict[str, Callable[..., None]] = {}
        self._list: List[str] = []
        self._permanent_items = permanent_items
        self._funcs.update(self._permanent_items)
        self._list = [x for x in self._permanent_items]

    def pop(self, name: str):
        """Pop :code:`name` from hook.
        """
        self._list.remove(name)
        self._funcs.pop(name)

    def append(self, name: str, func: Callable[..., None]) -> None:
        """Append :code:`func` with :code:`name` to hook.

        If hook with :code:`name` already exists, then the existing hook is
        removed and then the new one is appended.

        """
        self.insert(len(self._list), name, func)

    def remove(self, name: str) -> None:
        """Remove :code:`name` from hook."""
        if name in self._permanent_items:
            raise ValueError(f"Cannot remove permanent item {name}")
        if name in self._list:
            self._list.remove(name)
            self._funcs.pop(name)

    def push(self, name: str, func: Callable[..., None]) -> None:
        """Add hook to front.

        If hook with :code:`name` already exists, then the existing hook is
        removed and then the new one is pushed at front.

        """
        self.insert(0, name, func)

    def insert(self, i: int, name: str, func: Callable[..., None]) -> None:
        if name in self._list:
            self.remove(name)
        if name not in self._permanent_items:
            if i < 0:
                i = len(self._list) + i
            if i >= len(self._permanent_items):
                self._list.insert(i, name)
                self._funcs[name] = func
            else:
                self._list.insert(len(self._permanent_items), name)
                self._funcs[name] = func
        else:
            raise ValueError(f"Cannot modify permanent item {name}")

    def keys(self) -> List[str]:
        return self._list

    def values(self) -> List[Callable]:
        return [*self._funcs.values()]

    def index(self, item: str) -> int:
        return self._list.index(item)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, x: Union[int, str]) -> Callable[..., None]:
        if isinstance(x, int):
            return self._funcs[self._list[x]]
        else:
            return self._funcs[x]

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self) -> str:
        return self._list.__repr__()


class PropertyProxy(type):
    @property
    def paused(cls):
        return cls.trainer.paused

    @property
    def aborted(cls):
        return cls.trainer.aborted


class Tag:
    def __init__(self, x):
        self.tag = x
        self._members = {}

    @property
    def names(self) -> List[str]:
        return [*self._members.keys()]

    @property
    def members(self) -> Dict[str, Any]:
        return self._members

    def __call__(self, f):
        if f not in self._members:
            if isinstance(f, property):
                self._members[f.fget.__name__] = f
            else:
                self._members[f.__name__] = f
        return f


def GET(f):
    if not hasattr(f, "__http_methods__"):
        f.__http_methods__ = ["GET"]
    elif "GET" not in f.__http_methods__:
        f.__http_methods__.append("GET")
    return f


def POST(f):
    if not hasattr(f, "__http_methods__"):
        f.__http_methods__ = ["POST"]
    elif "POST" not in f.__http_methods__:
        f.__http_methods__.append("POST")
    return f


class ProxyDataset(Dataset):
    def __init__(self, dataset, indices):
        self._indices = indices
        self._dataset = dataset

    def __getitem__(self, indx):
        return self._dataset[self._indices[indx]]

    def __len__(self):
        return len(self._indices)


def get_proxy_dataloader(dataset, params, fraction_or_number, logger=None):
    if "seed" in params:
        np.random.seed(params["seed"])
    if isinstance(fraction_or_number, float):
        indices = np.random.choice(len(dataset),
                                   int(len(dataset) * fraction_or_number))
    elif isinstance(fraction_or_number, int):
        indices = np.random.choice(len(dataset),
                                   fraction_or_number)
    else:
        raise ValueError
    proxy_dataset = ProxyDataset(dataset, indices)
    temp_params = params.copy()
    temp_params.update({"batch_size": 1})  # stick to 1 right now
    if hasattr(dataset, "_get_raw"):
        proxy_dataset._get_raw = lambda x: dataset._get_raw(proxy_dataset._indices[x])
        temp_loader = MyDataLoader(proxy_dataset, return_raw=True,
                                   **temp_params)
        if logger:
            logger.info(f"Dataset has \"_get_raw\"\
            Drawing samples from test data is available!")
    else:
        temp_loader = MyDataLoader(proxy_dataset, **temp_params)
        if logger:
            logger.warn(f"Dataset dataset doesn't define \"_get_raw\".\
            Drawing samples from test data will not be available.")
    return temp_loader


def log_metrics_for_step(step, key_name, step_loader, metrics,
                         update_key, log_func):
    metric_names = set(metrics.keys())
    log_func(f"Total datapoints processed for {step} step in {key_name}: {update_key}," +
             f" {metrics['num_datapoints'][update_key]}")
    for m in metric_names:
        if update_key in metrics[m]:
            log_func(f"Value of metric {m} for {step} step in {key_name} is:" +
                     f" {metrics[m][update_key]}")
        else:
            log_func(f"No value recorded for {step}_step," +
                     f" metric {m} and {key_name} {update_key}")


def log_metrics(cls):
    if "iterations" in cls.trainer_params.training_steps:
        update_key = cls.iterations / cls._hooks_run_iter_frequency
        key_name = "iterations chunk"
    else:
        update_key = cls.epoch
        key_name = "epoch"
    log_func = cls._logd
    for step in cls._metrics:
        if getattr(cls, step + "_loader"):
            log_metrics_for_step(step, key_name, getattr(cls, step + "_loader"),
                                 cls._metrics[step], update_key, log_func)
        else:
            cls._logd(f"No dataloader for {step}")


# TODO: A lot of these controls and methods which depend on params will
#       have to be rewritten.
# TODO: multiplier can be a trainer_param
# FIXME: Annealing may depend on extra_metrics
# TODO: Annealing can be an external function like CheckFunc
def anneal_lr(cls, multiplier=.9):
    cls._logi("Annealing Learning Rate")
    check_losses = [loss[2] for loss in cls.losses if loss[0] == cls.save_on]
    if len(check_losses) >= 2:
        delta = check_losses[-2] - check_losses[-1]
        if delta < .01 * check_losses[-2]:
            for param_group in cls.optimizer.param_groups:
                param_group['lr'] *= multiplier
            cls._logi("Annealing...")


# FIXME: TRAINING_STEPS
# NOTE: For this a sample function has to be defined
def log_samples(cls, fraction=0.01):
    """For a few randomly selected datapoints, log the datapoint_name and
    corresponding model output
    """
    if "iterations" in cls.trainer_params.training_steps:
        raise NotImplementedError
    for step in cls.trainer_params.training_steps:
        dataset = getattr(cls, step + "_loader").dataset
        loader = get_proxy_dataloader(dataset,
                                      cls.dataloader_params[step],
                                      10,  # seems like a good number
                                      cls.logger)
        step_func = getattr(cls, step + "_step_func")
        # reset, launch each in a separate thread, wait for finish
        # CHECK: Is this a good idea? Maybe separate runner from epoch
        getattr(cls._epoch_runner, "run_" + step)(step_func, loader, True)
