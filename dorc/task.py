from typing import Callable, Iterable, Dict, Any, Optional
import time
import traceback
import multiprocessing as mp
from abc import ABC, abstractmethod, abstractproperty
from threading import Thread, Event


class Signals:
    def __init__(self, paused: Event, aborted: Event):
        self._paused = paused
        self._aborted = aborted

    @property
    def paused(self) -> Event:
        return self._paused

    @property
    def aborted(self) -> bool:
        return self._aborted.is_set()


class Task(ABC):
    """An abstraction for a generic task which runs either:
        1. Over an iterator
        2. A discrete function
    and has properties :attr:`running` and :attr:`finished`

    Args:
        func: The function which is run.
              The core of the `task`
        signals: An instance of :class:`Signals` to signal events to the task

    In case it is a discrete function, it should simply return the results and
    should indicate whether it's running or not. See :class:`DiscreteTask`

    In case the task runs over an iterator it should support waiting, finishing
    and gather result. See :class:`LoopTask` and :class:`LoopTaskWithHooks`

    """

    def __init__(self, func: Callable, signals: Signals):
        self.func = func
        self.signals = signals
        self._running = Event()
        self.status = True
        self._init = True

    @abstractmethod
    def finish(self) -> None:
        """Finishes the current execution loop by resetting running and waiting flags
        :attr:`waiting` and :attr:`running` are set to False

        """
        pass

    @property
    def init(self) -> bool:
        return self._init

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @abstractproperty
    def finished(self) -> bool:
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
    """A discrete :class:`Task`.

    Runs the given function in a separate process :class:`~mp.Process` till
    completion and stores the result.

    It can be in either :attr:`running` or :attr:`finished` states. There's no
    `init+paused` like state. It starts to run as soon as it is initialized.

    Args:
        func: The function which is run.
              The core of the `task`
        signals: An instance of :class:`Signals` to signal events to the task

    """
    def __init__(self, func: Callable, signals: Signals):
        super().__init__(func, signals)
        self.task_type = "discrete"
        self._states = {"running", "finished"}
        self._queue: mp.Queue = mp.Queue()
        self.result: Any = None
        self.aborted: bool = False

    def _check_p(self):
        while self._p.is_alive():
            time.sleep(1)
        self.result = self._queue.get()
        self._toggle_running()

    @property
    def finished(self) -> bool:
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
            self.status = False, f"{e}" + "\n" + traceback.format_exc()
            self._toggle_running()


class LoopTask(Task):
    """A :class:`Task` with an iterator that loops.

    Has attributes :attr:`paused`, :attr:`waiting`, :attr:`running` and
    :attr:`finished` and methods :meth:`pause` and :meth:`resume`

    Args:
        func: The function to run (over the loop)
        signals: An instance of :class:`Signals`.
        data_iterator: The iterator which yields some data.

    """
    def __init__(self, func: Callable, signals: Signals, data_iterator: Iterable):
        super().__init__(func, signals)
        self.task_type = "loop"
        self.data_iterator = data_iterator
        self._states = {"paused", "init", "finished", "running"}
        self._waiting = False
        self.result: Dict[Any, Any] = {}
        self.aborted = False

    def reset(self):
        self.result = {}
        self.finish()
        self._init = True

    def finish(self):
        """Finishes the current execution loop by resetting running and waiting flags
        :attr:`waiting` and :attr:`running` are set to False

        """
        self._running.clear()
        self._waiting = False

    @property
    def waiting(self) -> bool:
        return self._waiting

    @property
    def paused(self) -> bool:
        return self.running and self.waiting

    @property
    def finished(self) -> bool:
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
                # NOTE: signals.aborted is a lambda though it should really not
                #       be implementation specific. Whether it is a function or
                #       a property shouldn't be up to us, it should be available
                #       as a property in fact in the sense, signals should be an
                #       object.
                if hasattr(self.signals, "aborted") and self.signals.aborted:
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
            self.status = False, f"{e}" + "\n" + traceback.format_exc()
        self.finish()


class LoopTaskWithHooks(LoopTask):
    """A :class:`LoopTask` with hooks that run after each iteration.

    Args:
        func: The function to run (over the loop)
        signals: An instance of :class:`Signals`.
        data_iterator: The iterator which yields some data.
        hooks: An iterable of callables with no args
        hooks_with_args: An iterable of callables with arbitrary args

    A `hook` is a :class:`callable` without any arguments which can be used to
    gather result, record some data or any other side effects.

    A `hook_with_args` is :class:`callable` with any number of `args` or
    `kwargs`

    """
    def __init__(self, func: Callable, signals: Signals, data_iterator,
                 hooks: Iterable[Callable[[], None]],
                 hooks_with_args: Optional[Iterable[Callable[..., None]]] = None):
        super().__init__(func, signals, data_iterator)
        self._hooks = hooks     # key, value pairs, values are functions
        self._hooks_with_args = hooks_with_args or []

    @property
    def hooks(self) -> Iterable[Callable[[], None]]:
        return self._hooks

    @property
    def hooks_with_args(self) -> Iterable[Callable[..., None]]:
        return self._hooks_with_args

    def _run_hooks(self):
        for hook in self._hooks:
            hook(self)

    def _run_hooks_with_args(self, *args, **kwargs):
        for hook in self._hooks_with_args:
            hook(self, *args, **kwargs)

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
                self._run_hooks_with_args()
                if hasattr(self.signals, "aborted") and self.signals.aborted:
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
            self.status = False, f"{e}" + "\n" + traceback.format_exc()
        self.finish()
