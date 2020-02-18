import unittest
import sys
from types import SimpleNamespace
from threading import Thread, Event
import time
sys.path.append("../")
from trainer.epoch import DiscreteTask, LoopTask


class TaskTest(unittest.TestCase):
    def setUp(self):
        self.running_event = Event()
        self.aborted_event = Event()
        self.running_event.clear()
        self.signals = SimpleNamespace()
        self.signals.paused = self.running_event
        self.signals.aborted = lambda: self.aborted_event.is_set()

    def func(self, result):
        time.sleep(3)
        result.put({"val": True})

    def loop_func(self, x):
        time.sleep(.5)
        return True, x

    def test_discrete(self):
        task = DiscreteTask(self.func, self.signals)
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
        task = DiscreteTask(self.func, self.signals)
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

    def test_loop(self):
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


if __name__ == '__main__':
    unittest.main()
