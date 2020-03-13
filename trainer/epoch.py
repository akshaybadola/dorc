from typing import Callable, Iterable, List, Dict, Any, Tuple, Union, Set
import time
import numpy as np
import queue
from functools import partial
from threading import Thread, Event
import traceback
# import multiprocessing as mp

from .task import LoopTaskWithHooks, Signals
from .device import DeviceMonitor


class BatchVars:
    """BatchVars makes minimal assumptions on the return values of the function. At
    the very least it assumes that:
        1. all batches give output structure.
        2. each one has a format of (name, batch_num, prop_namme, value)

    At each append it automatically updates its index so that it's convenient to
    get values in the form of [batch_num][prop_name].

    The iterator returned is of its list and not the index.

    """
    def __init__(self) -> None:
        self._list: List[tuple] = []
        self._index: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def append(self, x: tuple) -> None:
        self._list.append(x)
        if x[0] not in self._index:
            self._index[x[0]] = {}  # self._index["train"] = {}
        if x[1] not in self._index[x[0]]:
            self._index[x[0]][x[1]] = {}  # self._index["train"][""] = {}
        self._index[x[0]][x[1]][x[2]] = len(self._list) - 1

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i: int) -> tuple:
        return self._list[i]

    def __iter__(self):
        return iter(self._list)

    @property
    def batches(self):
        return list(self._index.keys())

    @property
    def prop_names(self) -> Iterable[str]:
        if len(self._index):
            return list([*([*self._index.values()][0].values())][0].keys())
        else:
            return []

    def get(self, step, batch_num, prop_name):
        if (step in self._index and batch_num in self._index[step]
                and prop_name in self._index[step][batch_num]):
            indx = self._index[step][batch_num][prop_name]
            return self._list[indx][-1]
        else:
            return None


def _log_post_batch_hook(epoch: 'Epoch', **kwargs):
    step = kwargs["step"]
    metric_names = epoch.metrics[step]
    for m in metric_names:
        if m in kwargs:
            epoch.batch_vars.append((step, epoch.batch_num[step], m, kwargs[m]))
        elif step in epoch.extra_metrics and m in epoch.extra_metrics[step] and\
                epoch.extra_metrics[step][m]["when"] == "batch":
            em_func = epoch.extra_metrics[step][m]["function"]
            em_inputs = epoch.extra_metrics[step][m]["inputs"]
            f_inputs = dict((x, kwargs[x]) for x in em_inputs)
            epoch.batch_vars.append((step, epoch.batch_num[step], m, em_func(**f_inputs)))
    # NOTE: Deprecated, commented
    # if hasattr(epoch, "keep_time") and epoch.keep_time[step]:
    #     epoch.batch_vars.append((step, epoch.batch_num[step], "time", kwargs["time"]))
    if "device_mon" in kwargs:
        dm = kwargs["device_mon"]
        # print("LOGGING device_mon in kwargs", dm.__dict__)
        gpu_util = dm.gpu_util
        gpu_mem_util = dm.gpu_mem_util
        if gpu_util is not None:
            epoch.batch_vars.append((step, epoch.batch_num[step], "gpu_util",
                                     [(k, np.mean(v)) for k, v in gpu_util.items()]))
        if gpu_mem_util is not None:
            epoch.batch_vars.append((step, epoch.batch_num[step], "gpu_mem_util",
                                     [(k, np.mean(v)) for k, v in gpu_mem_util.items()]))
        epoch.batch_vars.append((step, epoch.batch_num[step], "cpu_util", np.mean(dm.cpu_util)))
        epoch.batch_vars.append((step, epoch.batch_num[step], "mem_util", np.mean(dm.mem_util)))
        epoch.batch_vars.append((step, epoch.batch_num[step], "time", dm.time))
    if "batch_time" in kwargs:
        epoch.batch_vars.append((step, epoch.batch_num[step], "batch_time", kwargs["batch_time"]))
    if epoch.extra_reportables[step]:
        for x in epoch.extra_reportables[step]:
            epoch.batch_vars.append((step, epoch.batch_num[step], x, kwargs[x]))
    epoch.total_samples[step] += kwargs["total"]  # always there


class EpochLoop(LoopTaskWithHooks):
    def __init__(self, func: Callable, signals: Signals,
                 data_iterator: Iterable, hooks: Iterable[Callable[[], None]],
                 device_mon: DeviceMonitor, max_iters: Union[int, None] = None):
        super().__init__(func, signals, data_iterator, hooks)
        self.device_mon = device_mon
        self._max_iters = max_iters
        self._iter_num = 0
        self._aborted = Event()
        self._data_buffer = 20  # batches, will depend actually
        self._data_q: queue.Queue = queue.Queue(maxsize=self._data_buffer)
        self._iter_finished = False
        self._data_thread = Thread(target=self._fetch_data)
        self._data_thread.start()
        self._hooks = hooks

    def finish(self):
        super().finish()
        self._iter_finished = True

    def _run_hooks(self, **kwargs):
        for hook in self._hooks:
            hook(**kwargs)

    # NOTE: What if we're waiting to either:
    #       a. put data on to a full queue
    #       b. take data from an empty queue
    #       and abort or something is called?
    def _fetch_data(self):
        data_iter = self.data_iterator.__iter__()
        while True:
            # NOTE: The _fetch_data thread also used to wait but now it won't,
            #       it'll simply fill up its size
            #
            # self.signals.paused.wait()  # wait if paused
            if self._max_iters is not None:
                self._iter_num += 1
                if self._iter_num > self._max_iters:
                    self._iter_finished = True
                    break
            start = time.time()
            try:
                batch = data_iter.__next__()
                if batch is None or self.finished:
                    # if batch is None:
                    #     print("BATCH NONE")
                    # else:
                    #     print("FINISHED")
                    self._iter_finished = True
                    break
            except StopIteration as e:
                self._iter_finished = True
                self.status = True, f"{e}" + "\n" + traceback.format_exc()
                break
            batch_time = time.time() - start
            if not self.finished:
                try:
                    self._data_q.put([batch_time, batch], False)
                    # print("PUT")
                except queue.Full:
                    while self._data_q.full():
                        time.sleep(.001)
                        if self.finished:
                            break

    def run_task(self, **kwargs):
        self._init = False
        self._running.set()
        self._toggle_waiting()
        self.signals.paused.wait()  # wait if paused
        self._toggle_waiting()
        try:
            while not self._iter_finished:  # or not self._data_q.empty():
                # print("RUNNING", self.running, self.waiting,
                #       self.signals.paused.is_set(), self.finished)
                # start = time.time()
                # x = self.data_iterator.__iter__().__next__()
                # batch_time = time.time() - start
                # print("Should get more here")
                # NOTE: Wait for data
                # CHECK: How to break out of here?
                try:
                    batch_time, x = self._data_q.get(False)
                    # print("GOT BATCH")
                except queue.Empty:
                    if self._iter_finished:
                        print("ITER FINISHED?")
                        x = None
                    else:
                        while self._data_q.empty():
                            # print("HERE!", self._iter_finished)
                            time.sleep(.001)
                            if self._iter_finished:
                                break
                        continue
                if not x:
                    print("NOT X")
                    break
                else:
                    with self.device_mon.monitor():
                        # print("MONITORING result")
                        # try:
                        result = self.func(x, **kwargs)
                        # except Exception as e:
                        #     print(e, kwargs)
                        #     print("THREAD CRASH")
                    result["batch_time"] = batch_time
                    # NOTE: Hooks are passed as callables with no positional
                    #       args from epoch and only take kwargs
                    # print("BEFORE Running hooks in train_loop")
                    self._run_hooks(**{"device_mon": self.device_mon, **result})
                    # print("DO WE GET HERE?")
                if hasattr(self.signals, "aborted") and self.signals.aborted:
                    if self.running:
                        self._toggle_running()
                    self.status = False, "Terminated"
                    self.finish()
                    self._aborted.set()
                    break
                else:               # wait after each iteration
                    self._toggle_waiting()
                    self.signals.paused.wait()
                    self._toggle_waiting()
        except Exception as e:
            self._aborted.set()
            self.status = False, f"{e}" + "\n" + traceback.format_exc()
        self.finish()


class Epoch:
    """Epoch is a class which manages the loop (train, val etc.) spawning, runs
    hooks and gathers the results

    """
    def __init__(self, metrics: Dict[str, Dict], signals: Signals,
                 device_poll: DeviceMonitor, extra_reportables: Dict[str, Dict],
                 **kwargs):
        self.metrics = metrics["metrics"]
        self.extra_metrics = metrics["extra_metrics"]
        self.signals = signals
        self.device_mon = device_poll
        self.device_poll = self.device_mon
        self.reset()
        self.extra_reportables: Dict[str, Dict] = {}
        for step in ["train", "val", "test"]:
            self.extra_reportables[step] = {}
            if step in extra_reportables:
                self.extra_reportables[step] = extra_reportables[step].copy()
        self._log_train_post_batch_hook: Callable[[], None] =\
            partial(_log_post_batch_hook, self, **{"step": "train"})
        self._log_val_post_batch_hook: Callable[[], None] =\
            partial(_log_post_batch_hook, self, **{"step": "val"})
        self._log_test_post_batch_hook: Callable[[], None] =\
            partial(_log_post_batch_hook, self, **{"step": "test"})
        self._post_batch_hooks = {"train": {"log": self._log_train_post_batch_hook},
                                  "val": {"log": self._log_val_post_batch_hook},
                                  "test": {"log": self._log_test_post_batch_hook}}
        for x in ["logi", "logd", "loge", "logw"]:
            setattr(self, "_" + x, kwargs[x])

    @property
    def init_or_finished(self) -> bool:
        return (not self.running) and (not self.waiting)

    @property
    def finished(self) -> Union[None, bool]:
        if self._current_loop is None:
            return None
        else:
            return self._current_loop.finished

    @property
    def status(self) -> Union[Tuple[bool, str], bool]:
        if self._current_loop is None:
            return False, "idle"
        elif isinstance(self._current_loop.status, bool):
            return self._current_loop.status, "Alive"
        else:
            return self._current_loop.status

    @property
    def current_loop(self) -> str:
        if self._current_loop is not None:
            return self._current_loop.name
        else:
            return "idle"

    @property
    def waiting(self) -> bool:
        if self._current_loop is not None:
            return self._current_loop.waiting
        else:
            return False

    @property
    def running(self) -> bool:
        if self._current_loop is not None:
            return self._current_loop.running
        else:
            return False

    @property
    def aborted(self) -> bool:
        if self._current_loop is not None:
            return self._current_loop._aborted
        else:
            return False

    @property
    def was_aborted(self) -> bool:
        if self._current_loop is not None:
            return self._current_loop._aborted.is_set()
        else:
            return False

    @property
    def info(self) -> Dict[str, Union[Dict[str, int], BatchVars]]:
        return {"total_samples": self.total_samples,
                "batch_nums": self.batch_num,
                "batch_vars": self.batch_vars}

    def toggle_waiting(self) -> None:
        if hasattr(self._current_loop, "_toggle_waiting"):
            self._current_loop._toggle_waiting()

    @property
    def post_batch_hooks(self) -> Set[Dict[str, Iterable[str]]]:
        return {{k: v.keys()} for k, v in self._post_batch_hooks.items()}

    def add_post_batch_hook(self, step: str, name: str, hook: Callable):
        if step in {"train", "val", "test"} and callable(hook):
            try:
                self._all_post_batch_hooks[step][name] = partial(hook, self)
                return True
            except Exception as e:
                return False, f"Error occurred {e}\n" + traceback.format_exc()
        else:
            return False, "Wrong step or not callable hook"

    def remove_post_batch_hook(self, step: str, name: str, hook: Callable) ->\
            Union[bool, Tuple[bool, str]]:
        assert step in {"train", "val", "test"}
        if name in self._all_post_batch_hooks[step]:
            self._all_post_batch_hooks[step].pop(name)
            return True
        else:
            return False, self._loge(f"Hook {name} not in {step} hooks")

    def reset(self) -> None:
        self._current_loop = None
        self.total_samples = {"train": 0, "val": 0, "test": 0}
        self.batch_num = {"train": 0, "val": 0, "test": 0}
        self.batch_vars = BatchVars()

    def run_train(self, train_step, train_loader, loop_type, num_iterations=0,
                  get_raw=False, callback=None):
        def train_one_batch(batch):
            if get_raw:
                raw, batch = batch[0], batch[1]
            received = train_step(batch)
            if get_raw:
                received["raw"] = raw
            self.batch_num["train"] += 1
            return received
        if loop_type == "iterations":
            assert num_iterations, "num_iterations cannot be zero with loop_type iterations"
        assert iter(train_loader), "train_loader has no iterator"
        assert train_loader.batch_size, "train_loader has no batch_size"
        self._logd("Starting run_train")
        self._logd(f"Train Loader properties: {len(train_loader)}")
        if loop_type != "iterations":
            num_iterations = len(train_loader)
        hooks = [*self._post_batch_hooks["train"].values()]
        self.train_loop = EpochLoop(train_one_batch, self.signals, train_loader, hooks,
                                    self.device_mon, max_iters=num_iterations)
        self.train_loop.name = "train"
        self._current_loop = self.train_loop
        self.train_loop.run_task()

    def run_val(self, val_step, val_loader, get_raw=False, callback=None):
        def val_one_batch(batch):
            if get_raw:
                raw, batch = batch[0], batch[1]
            received = val_step(batch)
            if get_raw:
                received["raw"] = raw
            self.batch_num["val"] += 1
            return received
        assert iter(val_loader), "val_loader has no iterator"
        assert val_loader.batch_size, "val_loader has no batch_size"
        self._logd("Starting run_val")
        self._logd(f"Val Loader properties: {len(val_loader)}")
        hooks = [*self._post_batch_hooks["val"].values()]
        num_iterations = len(val_loader)
        self.val_loop = EpochLoop(val_one_batch, self.signals, val_loader, hooks,
                                  self.device_mon, max_iters=num_iterations)
        self.val_loop.name = "val"
        self._current_loop = self.val_loop
        self.val_loop.run_task()

    def run_test(self, test_step, test_loader, get_raw=False, callback=None):
        def test_one_batch(batch):
            if get_raw:
                raw, batch = batch[0], batch[1]
            # print("TEST ONE", test_step, type(batch))
            received = test_step(batch)
            if get_raw:
                received["raw"] = raw
            self.batch_num["test"] += 1
            return received
        assert iter(test_loader), "test_loader has no iterator"
        assert test_loader.batch_size, "test_loader has no batch_size"
        self._logd("Starting run_test")
        self._logd(f"Test Loader properties: {len(test_loader)}")
        hooks = [*self._post_batch_hooks["test"].values()]
        num_iterations = len(test_loader)
        self.test_loop = EpochLoop(test_one_batch, self.signals, test_loader, hooks,
                                   self.device_mon, max_iters=num_iterations)
        self.test_loop.name = "test"
        self._current_loop = self.test_loop
        self.test_loop.run_task()
