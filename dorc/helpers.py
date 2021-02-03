from typing import List, Dict, Callable, Iterable, Union, Any
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import OrderedDict
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


class HookDict:
    def __init__(self, permanent_items: Dict[str, Callable[[], None]]):
        self._items = OrderedDict()
        self._permanent_items = permanent_items
        for k, v in self._permanent_items.items():
            self._items[k] = v

    def add(self, k: str, v: Callable[[], None]):
        if k not in self._permanent_items:
            self._items[k] = v

    def remove(self, k: str):
        if k not in self._permanent_items:
            self._items.pop(k)

    @property
    def keys(self) -> Iterable:
        return self._items.keys()

    @property
    def values(self) -> Iterable:
        return self._items.values()

    def __len__(self):
        return len(self._items)

    def __getitem__(self, k: str):
        return self._items[k]

    def __repr__(self):
        return self._items.__repr__()


class Hook:
    """A list of functions.

    A :class:`Hook` is a list of :class:`Callable` usually which take no
    arguments and return nothing.

    They can be executed at arbitrary points in a codebase with separate concerns.

    An instance of a :class:`Hook` must be called with :func:`run_hook` and if the
    functions in the hook require args, then :func:`run_hook_with_args`

    Args:
        permanent_items: List of functions which will never be removed from the hook

    """

    def __init__(self, permanent_items: List[Callable]):
        self._list = []
        self._permanent_items = permanent_items
        self._list = [x for x in self._permanent_items]

    def append(self, item: Callable[[], None]) -> None:
        if item not in self._list:
            self._list.append(item)
        else:                   # move to back
            self.remove(item)
            self.append(item)

    def remove(self, item: Callable[[], None]) -> None:
        if item not in self._permanent_items and item in self._list:
            self._list.remove(item)

    def push(self, item: Callable[[], None]) -> None:
        self._list.insert(len(self._permanent_items), item)

    def insert(self, i: int, item: Callable[[], None]) -> None:
        if i >= len(self._permanent_items):
            self._list.insert(i, item)
        else:
            self._list.insert(len(self._permanent_items), item)

    def __getitem__(self, x: int) -> Callable[[], None]:
        return self._list[x]

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


def _log_metrics_for_step(step, key_name, step_loader, metrics,
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
