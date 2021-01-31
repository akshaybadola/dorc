import unittest
import sys
# from types import SimpleNamespace
from threading import Thread, Event
import pytest
import time
sys.path.append("../")
from dorc.task import DiscreteTask, LoopTask, Signals


@pytest.mark.threaded
class TaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.running_event = Event()
        cls.aborted_event = Event()
        cls.running_event.clear()

    def put_func(self, result):
        "Wait for 2 seconds and put some data in result"
        time.sleep(2)
        result.put({"val": True})

    def loop_func(self, x):
        "Return some data after .5 seconds"
        time.sleep(.5)
        return True, x

    def test_discrete_init_pause_resume_finish(self):
        self.running_event.clear()
        self.aborted_event.clear()
        self.signals = Signals(self.running_event, self.aborted_event)
        task = DiscreteTask(self.put_func, self.signals)
        for x in task._states:
            with self.subTest(i=x):
                self.assertTrue(hasattr(task, x))
        self.assertTrue(task.init)
        self.assertFalse(task.finished)
        self.assertFalse(hasattr(task, "paused"))
        task.run_task()
        self.assertTrue(task.running)
        self.assertTrue(task.status)
        while task.running:
            time.sleep(1)
        self.assertTrue(task.finished)
        self.assertTrue(task.status)
        self.assertDictEqual(task.result, {"val": True})

    def test_discrete_abort(self):
        self.running_event.clear()
        self.aborted_event.clear()
        self.signals = Signals(self.running_event, self.aborted_event)
        task = DiscreteTask(self.put_func, self.signals)
        for x in task._states:
            with self.subTest(i=x):
                self.assertTrue(hasattr(task, x))
        self.assertTrue(task.init)
        self.assertFalse(task.finished)
        self.assertFalse(hasattr(task, "paused"))
        task.run_task()
        task.finish()
        self.assertTrue(task.finished)
        self.assertEqual(task.status, (False, "Terminated"))

    def test_discrete_catch_result_or_error(self):
        pass

    def test_loop_init_pause_resume_finish(self):
        self.running_event.clear()
        self.aborted_event.clear()
        self.signals = Signals(self.running_event, self.aborted_event)
        _iter = range(10)
        task = LoopTask(self.loop_func, self.signals, _iter)
        for x in task._states:
            with self.subTest(i=x):
                self.assertTrue(hasattr(task, x))
        self.assertTrue(task.init)
        self.assertFalse(task.finished)
        t = Thread(target=task.run_task)
        t.start()
        self.assertFalse(task.init)
        self.assertTrue(task.paused)
        self.running_event.set()
        self.assertTrue(task.running)
        while task.running:
            time.sleep(1)
        self.assertTrue(task.finished)
        self.assertTrue(len(task.result) == 10)
        self.assertFalse(task.aborted)

    def test_loop_abort(self):
        self.running_event.clear()
        self.aborted_event.clear()
        self.signals = Signals(self.running_event, self.aborted_event)
        _iter = range(10)
        task = LoopTask(self.loop_func, self.signals, _iter)
        for x in task._states:
            with self.subTest(i=x):
                self.assertTrue(hasattr(task, x))
        self.assertTrue(task.init)
        self.assertFalse(task.finished)
        t = Thread(target=task.run_task)
        t.start()
        self.assertFalse(task.init)
        self.assertTrue(task.paused)
        self.running_event.set()
        self.assertTrue(task.running)
        self.aborted_event.set()
        while task.running:
            time.sleep(.2)
        self.assertTrue(task.finished)
        self.assertTrue(task.aborted)
        self.assertFalse(task.status[0])
        self.assertTrue(task.result)
        task.reset()
        self.assertFalse(task.result)
        self.assertFalse(task.waiting)
        self.assertFalse(task.running)
        self.assertTrue(task.init)

    def test_loop_task_with_hooks(self):
        pass


if __name__ == '__main__':
    unittest.main()
