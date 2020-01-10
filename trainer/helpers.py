import torch
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
    def __init__(self, permanent_items):
        self._items = OrderedDict()
        self._permanent_items = permanent_items
        for k, v in self._permanent_items.items():
            self._items[k] = v

    def add(self, k, v):
        if k not in self._permanent_items:
            self._items[k] = v

    def remove(self, k):
        if k not in self._permanent_items:
            self._items.pop(k)

    @property
    def keys(self):
        return self._items.keys()

    @property
    def values(self):
        return self._items.values()

    def __len__(self):
        return len(self._items)

    def __getitem__(self, k):
        return self._items[k]

    def __repr__(self):
        return self._items.__repr__()


class HookList:
    def __init__(self, permanent_items):
        self._list = []
        self._permanent_items = permanent_items
        self._list = [x for x in self._permanent_items]

    def append(self, item):
        self._list.append(item)

    def remove(self, k):
        if k not in self._permanent_items:
            self._list.remove(k)

    def insert(self, i, k):
        if i > len(self._permanent_items):
            self._list.insert(i, k)

    def __getitem__(self, x):
        return self._list[x]

    def __len__(self):
        return len(self._list)

    def __repr__(self):
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
        self._members = []

    @property
    def members(self):
        return self._members

    def __call__(self, f):
        # if self.tag not in f.__dict__:
        #     f.__dict__[self.tag] = True
        #     self._funcs.append(f)
        # if callable(f):
        #     if f not in self._members:
        #         self._members.append(f)
        #     return f
        # else:
        #     self._members.append(f)
        if f not in self._members:
            self._members.append(f)
        return f


control = Tag("control")
prop = Tag("prop")
extras = Tag("extras")
helpers = Tag("helpers")


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


class ProxyDataset(torch.utils.data.Dataset):
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


# class Dummy:
#     pass


# def get_dummy_runner(trainer):
#     dummy = Dummy()
#     dummy._device_handles = trainer._device_handles
#     dummy._train_step = trainer._train_step
#     dummy._val_step = trainer._val_step
#     dummy._test_step = trainer._test_step
#     dummy.aborted = False
#     dummy.paused = False
#     dummy._metrics = None
#     dummy._extra_metrics = None
#     temp_runner = Epoch(dummy, {})
#     return temp_runner
