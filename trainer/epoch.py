# import ipdb
import time


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


# DONE: Why's epoch not reporting even standard losses?
# Perhaps decouple the Epoch entirely from the wrapper
# with a tcp socket like thingy
class Epoch:
    def __init__(self, metrics, signals, device_poll, extra_reportables):
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
        self._waiting = False
        self._running = False
        self._current_loop = "idle"
        self.reset()
        self._post_batch_hooks_to_run = {"train": ["log", "paused"],
                                         "val": ["log"],
                                         "test": ["log"]}

    def reset(self):
        self.total_samples = {"train": 0, "val": 0, "test": 0}
        self.batch_num = {"train": 0, "val": 0, "test": 0}
        self.batch_vars = BatchVars()

    @property
    def current_loop(self):
        return self._current_loop

    @property
    def waiting(self):
        return self._waiting

    @property
    def running(self):
        return self._running

    # TODO: May not be threadsafe. Should avoid writes by other threads
    #       copy/deepcopy may not be enough
    @property
    def info(self):
        return {"total_samples": self.total_samples.copy(),
                "batch_nums": self.batch_num.copy(),
                "batch_vars": self.batch_vars.copy()}

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

    # TODO: Format it better in a yield manner so it isn't in a loop.

    # CHECK: Why are there three functions here? Backprop happens in funcs
    #        anyway. Since I've decoupled the epoch, this should be better.
    #        One issue may be the batch_vars for each, as if the epoch is
    #        reset I may lose the information. It would be better to separate
    #        epoch from runner maybe, so that vars are collected in the epoch
    #        while the runner simply runs.

    # TODO: For all the timing hooks, ensure that paused time is not
    #       counted towards running
    def run_train(self, train_step, train_loader, get_raw=False):
        """Run the training loop until completion. `train_step` is purposely really simple

        :param train_step: :class: `function` which takes a batch, processes it and returns output
        :param train_loader: :class: `torch.nn.utils.data.DataLoader`
        :returns: None
        :rtype: None

        """

        self._running = True
        self._current_loop = "train"
        for i, batch in enumerate(train_loader):
            if not batch:
                break
            start = time.time()
            self.device_poll.start()
            if get_raw:
                raw, batch = batch[0], batch[1]
            received = train_step(batch)
            self.device_poll.end()
            end = time.time()
            if self.keep_time["train"]:
                received["time"] = end - start
            if get_raw:
                received["raw"] = raw
            received["gpu_util"] = self.device_poll.gpu_util
            received["gpu_max_mem"] = self.device_poll.gpu_max_mem
            received["gpu_min_mem"] = self.device_poll.gpu_min_mem
            received["cpu_util"] = self.device_poll.cpu_util
            received["cpu_max_mem"] = self.device_poll.cpu_max_mem
            received["cpu_min_mem"] = self.device_poll.cpu_min_mem
            self.batch_num["train"] += 1
            self._run_post_batch_hooks(**{"step": "train", **received})
            if self.signals.aborted:  # has to be here else, break won't work
                self._running = False
                self._current_loop = "idle"
                break
        self._running = False
        self._current_loop = "idle"

    def _log(self, x):
        if hasattr(self, "logger"):
            self.logger.debug(x)

    def run_val(self, val_step, val_loader, get_raw=False):
        self._running = True
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
            if self.signals.aborted:
                self._running = False
                self._current_loop = "idle"
                break
        self._running = False
        self._current_loop = "idle"

    def run_test(self, test_step, test_loader, get_raw=False):
        self._running = True
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
            if self.signals.aborted:
                self._running = False
                self._current_loop = "idle"
                break
        self._running = False
        self._current_loop = "idle"

    def _run_post_batch_hooks(self, **kwargs):
        all_hooks = self.all_post_batch_hooks
        hook_prefixes = self.post_batch_hooks_to_run[kwargs["step"]]
        for hook in hook_prefixes:
            all_hooks["_".join(["", hook, "post_batch_hook"])](self, **kwargs)

    # NOTE: Aborting while paused is handled externally.
    # TODO: Ideally it should be async, but that's a bit complicated
    def _paused_post_batch_hook(self, **kwargs):
        self._waiting = True
        while self.signals.paused:
            time.sleep(5)
        self._waiting = False

    # def _abort_post_batch_hook(self, **kwargs):
    #     if self.signals.aborted:
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
