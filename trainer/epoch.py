import time
from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
from threading import Thread, Event


class BatchVars:
    """BatchVars makes minimal assumptions on the return values of the function. At
    the very least it assumes that:
        1. all batches give output structure.
        2. each one has a format of (name, batch_num, prop_namme, value)

    At each append it automatically updates its index so that it's convenient to
    get values in the form of [batch_num][prop_name].

    The iterator returned is of its list and not the index.

    """
    def __init__(self):
        self._list = []
        self._index = {}

    def append(self, x):
        self._list.append(x)
        if x[1] not in self._index:
            self._index[x[1]] = {}
        self._index[x[1]][x[2]] = len(self._list) - 1

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __iter__(self):
        return iter(self._list)

    @property
    def batches(self):
        return list(self._index.keys())

    @property
    def prop_names(self):
        if len(self._index):
            return list(self._index[0].keys())
        else:
            return []

    def get(self, batch_num, prop_name):
        if batch_num in self._index and prop_name in self._index[batch_num]:
            indx = self._index[batch_num][prop_name]
            return self._list[indx][-1]
        else:
            return None


class Task:
    """Task should be a generic task which runs either:
        1. Over an iterator
        2. A discrete function

    In case it runs over an iterator it should support waiting, finishing and
    gather result.

    In case it is a discrete function, it should simply return the results and
    should indicate whether it's running or not.

    Epoch is a kind of task, but it isn't a subclass right now.
    """

    def __init__(self, func, signals):
        self.func = func
        self.signals = signals
        self._running = Event()
        self.status = True
        self._init = True

    @abstractmethod
    def finish(self):
        """Finishes the current execution loop by resetting running and waiting flags
        `self._waiting` and `self._running` are set to False

        :returns: None
        :rtype: None

        """
        pass

    @property
    def init(self):
        return self._init

    @property
    def running(self):
        return self._running.is_set()

    @abstractmethod
    def finished(self):
        pass

    def _toggle_running(self):
        if self.running:
            self._running.clear()
        elif not self.running:
            self._running.set()

    @abstractmethod
    def run_task(self):
        pass


class DiscreteTask(Task):
    """A discrete task. Runs till completion and returns the result.

    It can be in {"running", "finished"} states. There's no "init+paused" like
    state. It starts to run as soon as it is initialized.

    """
    def __init__(self, func, signals):
        super().__init__(func, signals)
        self.task_type = "discrete"
        self._states = {"running", "finished"}
        self._queue = Queue()
        self.result = None

    def _check_p(self):
        while self._p.is_alive():
            time.sleep(1)
        self.result = self._queue.get()
        self._toggle_running()

    @property
    def finished(self):
        return (not self.init) and (not self.running)

    def finish(self):
        if self.running:
            self._p.terminate()
            if self._p.is_alive():
                self._p.kill()
            self._queue.put(None)
            self.status = False, "Terminated"
            self._running.clear()

    def run_task(self, *args, **kwargs):
        self._init = False
        args = [self._queue, *args]
        self._p = Process(target=self.func, args=args, kwargs=kwargs)
        self._toggle_running()
        try:
            self._p.start()
            self._t = Thread(target=self._check_p)
            self._t.start()
        except Exception as e:
            self.status = False, e
            self._toggle_running()


class LoopTask(Task):
    """Task should be a generic task which runs either:
        1. Over an iterator
        2. A discrete function

    In case it runs over an iterator it should support waiting, finishing and
    gather result.

    In case it is a discrete function, it should simply return the results and
    should indicate whether it's running or not.

    Epoch is a kind of task, but it isn't a subclass right now.
    """
    def __init__(self, func, signals, data_iterator):
        super().__init__(func, signals)
        self.task_type = "loop"
        self.data_iterator = data_iterator
        self._states = {"paused", "init", "finished", "running"}
        self._waiting = False
        self.result = {}
        self.aborted = False

    def reset(self):
        self.result = {}
        self.finish()
        self._init = True

    def finish(self):
        """Finishes the current execution loop by resetting running and waiting flags
        `self._waiting` and `self._running` are set to False

        :returns: None
        :rtype: None

        """
        self._running.clear()

    @property
    def waiting(self):
        return self._waiting

    @property
    def running(self):
        return self._running.is_set()

    @property
    def paused(self):
        return self.running and self.waiting

    @property
    def finished(self):
        return (not self.running) and (not self.waiting) and (not self.init)

    def _toggle_waiting(self):
        if self.waiting:
            self._waiting = False
        elif not self.waiting:
            self._waiting = True

    def run_task(self, **kwargs):
        self._init = False
        self._running.set()
        self._toggle_waiting()
        self.signals.paused.wait()  # wait if paused
        self._toggle_waiting()
        try:
            for i, x in enumerate(self.data_iterator):
                self.result[i] = self.func(x, **kwargs)
                if hasattr(self.signals, "aborted") and self.signals.aborted():
                    if self.running:
                        self._toggle_running()
                    self.status = False, "Terminated"
                    self.finish()
                    self.aborted = True
                    break
                else:               # wait after each iteration
                    self._toggle_waiting()
                    self.signals.paused.wait()
                    self._toggle_waiting()
        except Exception as e:
            self.status = False, e
        self.finish()


class LoopTaskWithHooks(LoopTask):
    def __init__(self, func, signals, data_iterator, hooks):
        super().__init__(func, signals, data_iterator)
        self._all_hooks = hooks     # key, value pairs, values are functions

    @property
    def all_hooks(self):
        return [h for h in self._hooks]

    @property
    def hooks_to_run(self):
        return [h for h in self._hooks_to_run]

    @hooks_to_run.setter
    def hooks_to_run(self, x):
        hooks_to_run = []
        for _x in x:
            if _x in self._all_hooks:
                hooks_to_run.append(_x)
            else:
                self._loge(f"Hook {_x} not in available hooks")
        self._hooks_to_run = hooks_to_run

    def _run_hooks(self):
        for h, hook in self._hooks_to_run:
            hook(self)

    def run_task(self, **kwargs):
        self._init = False
        self._running.set()
        self._toggle_waiting()
        self.signals.paused.wait()  # wait if paused
        self._toggle_waiting()
        try:
            for i, x in enumerate(self.data_iterator):
                self.result[i] = self.func(x, **kwargs)
                self._run_hooks()
                if hasattr(self.signals, "aborted") and self.signals.aborted():
                    if self.running:
                        self._toggle_running()
                    self.status = False, "Terminated"
                    self.finish()
                    self.aborted = True
                    break
                else:               # wait after each iteration
                    self._toggle_waiting()
                    self.signals.paused.wait()
                    self._toggle_waiting()
        except Exception as e:
            self.status = False, e
        self.finish()


class NewEpoch(LoopTaskWithHooks):
    def __init__(self, metrics, signals, device_poll, extra_reportables, **kwargs):
        self.metrics = metrics["metrics"]
        self.signals = signals
        self.device_poll = device_poll
        self._waiting = Event()
        self._running = Event()
        self.aborted = Event()
        self.reset()
        self._post_batch_hooks_to_run = {"train": ["log", "paused"],
                                         "val": ["log"],
                                         "test": ["log"]}

    def reset(self):
        self.total_samples = {"train": 0, "val": 0, "test": 0}
        self.batch_num = {"train": 0, "val": 0, "test": 0}
        self.finish()
        self.batch_vars = BatchVars()
        self.aborted.clear()

    def finish(self):
        """Finishes the current execution loop by resetting running and waiting flags
        `self._waiting` and `self._running` are set to False

        :returns: None
        :rtype: None

        """
        self._waiting.clear()
        self._running.clear()
        self._current_loop = "idle"

    def run_task(self, **kwargs):
        self._init = False
        self._running.set()
        self._toggle_waiting()
        self.signals.paused.wait()  # wait if paused
        self._toggle_waiting()
        try:
            for i, x in enumerate(self.data_iterator):
                with self.device_poll.monitor():
                    self.result[i] = self.func(x, **kwargs)
                self._run_hooks()
                if hasattr(self.signals, "aborted") and self.signals.aborted():
                    if self.running:
                        self._toggle_running()
                    self.status = False, "Terminated"
                    self.finish()
                    self.aborted = True
                    break
                else:               # wait after each iteration
                    self._toggle_waiting()
                    self.signals.paused.wait()
                    self._toggle_waiting()
        except Exception as e:
            self.status = False, e
        self.finish()


class Epoch:
    """Epoch is an abstraction of epoch and in fact can not be a cyclical train/val
    type but can be iterations or arbitrary training procedure. It's simply a
    wrapper to hold the metrics and other variables collected while training.

    The `Epoch` has two variables which control its state:
        `running` and `waiting`

    In addition `current_loop` reports attribute of the current state. State
    transition is simple in the sense that either the runner is running or
    waiting. While running implies that some task is currently underway, waiting
    means that it's waiting for the trainer to finish doing some other task
    before it can resumes the task.

    Even if it has reached the end of train_loader or some other iterator, it
    has in fact no way to know that and will pause regardless. Therefore it can
    be instructed to wrap up everything at any point in any loop

    `current_loop` determines which task is underway and "idle" signifies that
    nothing is running right now.

    For example, at the beginning, `current_loop` is idle and both `running` and
    `waiting` are False. When `run_train` is called `current_loop` is "train",
    `running` is true and `waiting` is False, after each batch, the runner waits
    if "paused" is one of the post_batch_hooks, i.e., after each batch it checks
    whether to pause itself or not. At that point it's in `waiting`
    state. `current_loop` again doesn't really correspond to any iterator but
    simply refers to whichever task it has been assigned right now.

    """
    def __init__(self, metrics, signals, device_poll, extra_reportables):
        self.name = "no_name"
        self.metrics = metrics["metrics"]
        self.extra_metrics = metrics["extra_metrics"]
        self.signals = signals
        self.device_poll = device_poll
        self.keep_time = {}
        self.extra_reportables = {}
        for step in ["train", "val", "test"]:
            self.extra_reportables[step] = {}
            self.keep_time[step] = False
            if step in extra_reportables:
                if "time" in extra_reportables[step]:
                    self.keep_time[step] = True
                else:
                    self.keep_time[step] = False
                self.extra_reportables[step] = extra_reportables[step].copy()
        self._waiting = Event()
        self._running = Event()
        self.aborted = Event()
        self._current_loop = "idle"
        self.reset()
        self._post_batch_hooks_to_run = {"train": ["log", "paused"],
                                         "val": ["log"],
                                         "test": ["log"]}

    def reset(self):
        self.total_samples = {"train": 0, "val": 0, "test": 0}
        self.batch_num = {"train": 0, "val": 0, "test": 0}
        self.finish()
        self.batch_vars = BatchVars()
        self.aborted.clear()

    def finish(self):
        """Finishes the current execution loop by resetting running and waiting flags
        `self._waiting` and `self._running` are set to False

        :returns: None
        :rtype: None

        """
        self._waiting.clear()
        self._running.clear()
        self._current_loop = "idle"

    @property
    def init_or_finished(self):
        return (not self.running) and (not self.waiting)

    @property
    def current_loop(self):
        return self._current_loop

    @property
    def waiting(self):
        return self._waiting.is_set()

    @property
    def running(self):
        return self._running.is_set()

    # TODO: May not be threadsafe. Should avoid writes by other threads
    #       copy/deepcopy may not be enough
    @property
    def info(self):
        return {"total_samples": self.total_samples,
                "batch_nums": self.batch_num,
                "batch_vars": self.batch_vars}

    @property
    def all_post_batch_hooks(self):
        return dict((x, y) for (x, y) in self.__class__.__dict__.items()
                    if x.endswith("post_batch_hook"))

    @property
    def post_batch_hooks_to_run(self):
        return self._post_batch_hooks_to_run

    @post_batch_hooks_to_run.setter
    def post_batch_hooks_to_run(self, x):
        assert any(_x in x for _x in ["train", "val", "test"])
        assert all(all(__x in self.all_post_batch_hooks for __x in _x) for _x in x.values())
        for _x in x:
            self._post_batch_hooks_to_run[_x] = x[_x]

    def _toggle_running(self):
        if self.running:
            self._running.clear()
        elif not self.running:
            self._running.set()

    def toggle_waiting(self):
        if self.waiting:
            self._waiting.clear()
        elif not self.waiting:
            self._waiting.set()

    # TODO: Format it better in a yield manner so it isn't in a loop.
    # CHECK: Why are there three functions here? Backprop happens in funcs
    #        anyway. Since I've decoupled the epoch, this should be better.
    #        One issue may be the batch_vars for each, as if the epoch is
    #        reset I may lose the information. It would be better to separate
    #        epoch from runner maybe, so that vars are collected in the epoch
    #        while the runner simply runs.

    # TODO: For all the timing hooks, ensure that paused time is not
    #       counted towards running
    def run_train(self, train_step, train_loader, loop_type, num_iterations=0, get_raw=False):
        """Run the training loop until completion. `train_step` is purposely really
        simple.

        :param train_step:   :class:`function` which takes a batch, processes it and returns output
        :param train_loader: :class:`torch.nn.utils.data.DataLoader`
        :param loop_type:    :class:`str` `completion` or `iterations`. If the loop_type is 
                             `completion` the loop is run until the train_loader is exhausted.
                              Otherwise, the loop is run for num_iterations.
        :returns: None
        :rtype: None

        """
        if loop_type == "iterations":
            assert num_iterations, "num_iterations cannot be zero with loop_type iterations"
        if not self.running:
            self._toggle_running()
        self._current_loop = "train"
        assert iter(train_loader), "train_loader has no iterator"
        assert train_loader.batch_size, "train_loader has no batch_size"
        self._log("Starting run_train")
        self._log(f"Train Loader properties: {len(train_loader)}")

        def train_one_batch(batch):
            with self.device_poll.monitor():
                if get_raw:
                    raw, batch = batch[0], batch[1]
                received = train_step(batch)
            if self.keep_time["train"]:
                received["time"] = self.device_poll._data
            if get_raw:
                received["raw"] = raw
            self.batch_num["train"] += 1
            self._run_post_batch_hooks(**{"step": "train", **received})
        if loop_type == "iterations":
            for i in range(num_iterations):
                batch_time = time.time()
                # NOTE: This is a synchronous operation. Could be async
                batch = train_loader.__iter__().__next__()
                batch_time = time.time() - batch_time
                if not batch:
                    break
                train_time = time.time()
                train_one_batch(batch)
                train_time = time.time() - train_time
                print("batch_time, train_time", batch_time, train_time, batch_time > train_time)
                if self.signals.aborted():  # has to be here else, break won't work
                    print("aborting from epoch runner")
                    if self.running:
                        self._toggle_running()
                    self._current_loop = "idle"
                    self.finish()
                    self.aborted.set()
                    return
        else:
            for i, batch in enumerate(train_loader):
                if not batch:
                    break
                train_one_batch(batch)
                if self.signals.aborted():  # has to be here else, break won't work
                    print("aborting from epoch runner")
                    if self.running:
                        self._toggle_running()
                    self._current_loop = "idle"
                    self.finish()
                    self.aborted.set()
                    return
        if self.running:
            self._toggle_running()
        self._current_loop = "idle"

    def _log(self, x):
        if hasattr(self, "logger"):
            self.logger.debug(x)

    # NOTE: run_val and run_test don't have option to run with only iterations right
    #       now.
    def run_val(self, val_step, val_loader, get_raw=False):
        if not self.running:
            self._toggle_running()
        self._current_loop = "val"
        # CHECK: There may not be an iter
        assert iter(val_loader), "val_loader has no iterator"
        assert val_loader.batch_size, "val_loader has no batch_size"
        self._log("Starting run_val")
        self._log(f"Val Loader properties: {len(val_loader)}")
        for i, batch in enumerate(val_loader):
            start = time.time()
            if not batch:
                break
            if get_raw:
                raw, batch = batch[0], batch[1]
            received = val_step(batch)
            end = time.time()
            if self.keep_time["val"]:
                received["time"] = end - start
            if get_raw:
                received["raw"] = raw
            self.batch_num["val"] += 1
            self._run_post_batch_hooks(**{"step": "val", **received})
            if self.signals.aborted():
                if self.running:
                    self._toggle_running()
                self._current_loop = "idle"
                self.finish()
                self.aborted.set()
                return
            if i > len(val_loader):
                break
        if self.running:
            self._toggle_running()
        self._current_loop = "idle"

    def run_test(self, test_step, test_loader, get_raw=False):
        if not self.running:
            self._toggle_running()
        self._current_loop = "test"
        for i, batch in enumerate(test_loader):
            start = time.time()
            if not batch:
                break
            if get_raw:
                raw, batch = batch[0], batch[1]
            received = test_step(batch)
            end = time.time()
            if self.keep_time["test"]:
                received["time"] = end - start
            if get_raw:
                received["raw"] = raw
            self.batch_num["test"] += 1
            self._run_post_batch_hooks(**{"step": "test", **received})
            if self.signals.aborted():
                if self.running:
                    self._toggle_running()
                self._current_loop = "idle"
                self.finish()
                self.aborted.set()
                return
            if i > len(test_loader):
                break
        if self.running:
            self._toggle_running()
        self._current_loop = "idle"

    # def run_val(self, val_step, val_loader, loop_type, num_iterations=0, get_raw=False):
    #     if loop_type == "iterations":
    #         assert num_iterations, "num_iterations cannot be zero with loop_type iterations"
    #     self._running = True
    #     self._current_loop = "val"
    #     # CHECK: There may not be an iter
    #     assert iter(val_loader), "val_loader has no iterator"
    #     assert val_loader.batch_size, "val_loader has no batch_size"
    #     self._log("Starting run_val")
    #     self._log(f"Val Loader properties: {len(val_loader)}")

    #     def do_val(batch):
    #         start = time.time()
    #         if get_raw:
    #             raw, batch = batch[0], batch[1]
    #         received = val_step(batch)
    #         end = time.time()
    #         if self.keep_time["val"]:
    #             received["time"] = end - start
    #         if get_raw:
    #             received["raw"] = raw
    #         self.batch_num["val"] += 1
    #         self._run_post_batch_hooks(**{"step": "val", **received})
    #     if loop_type == "iterations":
    #         for i in num_iterations:
    #             batch = val_loader.__next__()  # or iter.next?
    #             if not batch:
    #                 break
    #             do_val(batch)
    #             if self.signals.aborted():
    #                 self._running = False
    #                 self._current_loop = "idle"
    #                 break
    #     else:
    #         for i, batch in enumerate(val_loader):
    #             if not batch:
    #                 break
    #             do_val(batch)
    #             if self.signals.aborted():
    #                 self._running = False
    #                 self._current_loop = "idle"
    #                 break
    #     self._running = False
    #     self._current_loop = "idle"

    # def run_test(self, test_step, test_loader, loop_type, num_iterations=0, get_raw=False):
    #     if loop_type == "iterations":
    #         assert num_iterations, "num_iterations cannot be zero with loop_type iterations"
    #     self._running = True
    #     self._current_loop = "test"
    #     # CHECK: There may not be an iter
    #     assert iter(test_loader), "test_loader has no iterator"
    #     assert test_loader.batch_size, "test_loader has no batch_size"
    #     self._log("Starting run_test")
    #     self._log(f"Test Loader properties: {len(test_loader)}")

    #     def do_test(batch):
    #         start = time.time()
    #         if get_raw:
    #             raw, batch = batch[0], batch[1]
    #         received = test_step(batch)
    #         end = time.time()
    #         if self.keep_time["test"]:
    #             received["time"] = end - start
    #         if get_raw:
    #             received["raw"] = raw
    #         self.batch_num["test"] += 1
    #         self._run_post_batch_hooks(**{"step": "test", **received})
    #     if loop_type == "iterations":
    #         for i in num_iterations:
    #             batch = test_loader.__next__()  # or iter.next?
    #             if not batch:
    #                 break
    #             do_test(batch)
    #             if self.signals.aborted():
    #                 self._running = False
    #                 self._current_loop = "idle"
    #                 break
    #     else:
    #         for i, batch in enumerate(test_loader):
    #             if not batch:
    #                 break
    #             do_test(batch)
    #             if self.signals.aborted():
    #                 self._running = False
    #                 self._current_loop = "idle"
    #                 break
    #     self._running = False
    #     self._current_loop = "idle"

    def _run_post_batch_hooks(self, **kwargs):
        all_hooks = self.all_post_batch_hooks
        hook_prefixes = self.post_batch_hooks_to_run[kwargs["step"]]
        for hook in hook_prefixes:
            all_hooks["_".join(["", hook, "post_batch_hook"])](self, **kwargs)

    # NOTE: Aborting while paused is handled externally.
    # TODO: Ideally it should be async, but that's a bit complicated
    #       I can use async_get maybe from concurrent.futures or something
    def _paused_post_batch_hook(self, **kwargs):
        self.toggle_waiting()
        # while self.signals.paused:
        #     time.sleep(5)
        # wait for paused to clear (or actually unpaused to set)
        self.signals.paused.wait()
        self.toggle_waiting()

    # def _abort_post_batch_hook(self, **kwargs):
    #     if self.signals.aborted():
    #         pass

    # TODO: log GPU, CPU, Memory per batch
    #       GPU, CPU usage is a problem as they'll be idle when
    #       being reported. They have to be collected while running.
    # FIXME: This thingy is making an assumption that the other thingy will be
    #        in kwargs, which isn't quite right. How do I fix this? There will
    #        have to be checks that "x" value is not reportable.
    def _log_post_batch_hook(self, **kwargs):
        step = kwargs["step"]
        metric_names = self.metrics[step]
        # if hasattr(self, "logger"):
        #     self.logger.debug(f"runner extra metrics {self.extra_metrics}")
        for m in metric_names:
            if m in kwargs:
                self.batch_vars.append((step, self.batch_num[step], m, kwargs[m]))
            elif step in self.extra_metrics and m in self.extra_metrics[step] and\
                    self.extra_metrics[step][m]["when"] == "batch":
                em_func = self.extra_metrics[step][m]["function"]
                em_inputs = self.extra_metrics[step][m]["inputs"]
                f_inputs = dict((x, kwargs[x]) for x in em_inputs)
                self.batch_vars.append((step, self.batch_num[step], m, em_func(**f_inputs)))
        if self.keep_time[step]:
            self.batch_vars.append((step, self.batch_num[step], "time", kwargs["time"]))
        if self.extra_reportables[step]:
            for x in self.extra_reportables[step]:
                self.batch_vars.append((step, self.batch_num[step], x, kwargs[x]))
        # CHECK: Is total here total samples or total batches?
        self.total_samples[step] += kwargs["total"]  # always there
