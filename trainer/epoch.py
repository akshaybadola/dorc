# import ipdb
import time

from .device import DevicePoll


# TODO: Why's epoch not reporting even standard losses?
# Perhaps decouple the Epoch entirely from the wrapper
# with a tcp socket like thingy
class Epoch:
    def __init__(self, wrp, extra_reportables):
        self.wrp = wrp
        self.device_poll = DevicePoll(self.wrp._device_handles)
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
        self.batch_vars = []

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
    # TODO: For all the timing hooks, ensure that paused time is not
    #       counted towards running
    def run_train(self):
        self._running = True
        self._current_loop = "train"
        for i, batch in enumerate(self.wrp.train_loader):
            start = time.time()
            self.device_poll.start()
            received = self.wrp._train_step(self.wrp, batch)
            self.device_poll.end()
            end = time.time()
            if self.keep_time["train"]:
                received["time"] = end - start
            received["gpu_util"] = self.device_poll.gpu_util
            received["gpu_max_mem"] = self.device_poll.gpu_max_mem
            received["gpu_min_mem"] = self.device_poll.gpu_min_mem
            received["cpu_util"] = self.device_poll.cpu_util
            received["cpu_max_mem"] = self.device_poll.cpu_max_mem
            received["cpu_min_mem"] = self.device_poll.cpu_min_mem
            self.batch_num["train"] += 1
            self._run_post_batch_hooks(**{"step": "train", **received})
            if self.wrp.aborted:  # has to be here else, break won't work
                self._running = False
                self._current_loop = "idle"
                break

    def run_val(self):
        self._running = True
        self._current_loop = "val"
        for i, batch in enumerate(self.wrp.val_loader):
            start = time.time()
            received = self.wrp._val_step(self.wrp, batch)
            end = time.time()
            if self.keep_time["val"]:
                received["time"] = end - start
            self.batch_num["val"] += 1
            self._run_post_batch_hooks(**{"step": "val", **received})
            if self.wrp.aborted:
                self._running = False
                self._current_loop = "idle"
                break

    def run_test(self):
        self._running = True
        self._current_loop = "test"
        for i, batch in enumerate(self.wrp.test_loader):
            start = time.time()
            received = self.wrp._test_step(self.wrp, batch)
            end = time.time()
            if self.keep_time["test"]:
                received["time"] = end - start
            self.batch_num["test"] += 1
            self._run_post_batch_hooks(**{"step": "test", **received})
            if self.wrp.aborted:
                self._running = False
                self._current_loop = "idle"
                break

    def _run_post_batch_hooks(self, **kwargs):
        all_hooks = self.all_post_batch_hooks
        hook_prefixes = self.post_batch_hooks_to_run[kwargs["step"]]
        for hook in hook_prefixes:
            all_hooks["_".join(["", hook, "post_batch_hook"])](self, **kwargs)

    # NOTE: Aborting while paused is handled externally.
    # TODO: Ideally it should be async, but that's a bit complicated
    def _paused_post_batch_hook(self, **kwargs):
        self._waiting = True
        while self.wrp.paused:
            time.sleep(5)
        self._waiting = False

    # def _abort_post_batch_hook(self, **kwargs):
    #     if self.wrp.aborted:
    #         pass

    # TODO: log GPU, CPU, Memory per batch
    #       GPU, CPU usage is a problem as they'll be idle when
    #       being reported. They have to be collected while running.
    # FIXME: This thingy is making an assumption that the other thingy will be
    #        in kwargs, which isn't quite right. How do I fix this? There will
    #        have to be checks that "x" value is not reportable.
    def _log_post_batch_hook(self, **kwargs):
        step = kwargs["step"]
        metric_names = self.wrp._metrics[step]
        for m in metric_names:
            if m in kwargs:
                self.batch_vars.append((step, self.batch_num[step], m, kwargs[m]))
            elif m in self.wrp._extra_metrics[step] and\
                    self.wrp._extra_metrics[step][m]["when"] == "batch":
                em_func = self.wrp._extra_metrics[step][m]["function"]
                em_inputs = self.wrp._extra_metrics[step][m]["inputs"]
                f_inputs = dict((x, kwargs[x]) for x in em_inputs)
                self.batch_vars.append((step, self.batch_num[step], m, em_func(**f_inputs)))
        if self.keep_time[step]:
            self.batch_vars.append((step, self.batch_num[step], "time", kwargs["time"]))
        if self.extra_reportables[step]:
            for x in self.extra_reportables[step]:
                self.batch_vars.append((step, self.batch_num[step], x, kwargs[x]))
        # CHECK: Is total here total samples or total batches?
        self.total_samples[step] += kwargs["total"]  # always there
