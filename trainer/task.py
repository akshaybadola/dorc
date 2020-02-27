import time
import multiprocessing as mp
from abc import ABC, abstractmethod
from threading import Thread, Event


class Task(ABC):
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
        self._queue = mp.Queue()
        self.result = None
        self.aborted = None

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
        self._p = mp.Process(target=self.func, args=args, kwargs=kwargs)
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
        self._hooks = hooks     # key, value pairs, values are functions

    @property
    def hooks(self):
        return self._hooks

    def _run_hooks(self):
        for h, hook in self._hooks:
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
            self.aborted = True
            self.status = False, e
        self.finish()
