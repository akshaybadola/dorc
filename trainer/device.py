from threading import Thread, Event
import time
import psutil
import pynvml


def cpu_info():
    return {"cpu_count": psutil.cpu_count(), "cpu_util": psutil.cpu_percent()}


def memory_info():
    info = psutil.virtual_memory()
    return {"total": info.total, "used": info.used}


def gpu_util(handles):
    def _get_util(h):
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {"gpu": util.gpu, "memory": 100 * (mem.used / mem.total)}
    return {gpu_id: _get_util(h) for gpu_id, h in handles.items()}


def init_nvml(gpus):
    pynvml.nvmlInit()
    return {x: pynvml.nvmlDeviceGetHandleByIndex(x) for x in gpus}


class Monitor:
    def __init__(self, dm):
        self._dm = dm

    def __enter__(self):
        self._dm._reset()
        self._dm._start()

    def __exit__(self, *args):
        self._dm._end()


class DeviceMonitor:
    def __init__(self, gpu_handles, poll_interval=0.05):
        self._handles = gpu_handles
        self._interval = poll_interval
        self._running_event = Event()
        self._t = None

    def _reset(self):
        if self._handles:
            self._data = {"cpu_info": [], "memory_info": [], "gpu_info": []}
        else:
            self._data = {"cpu_info": [], "memory_info": []}

    def monitor(self):
        return Monitor(self)

    def _monitor_func(self):
        self._data["time"] = 0
        if self._handles:
            while self._running_event.is_set():
                self._data["cpu_info"].append(cpu_info())
                self._data["memory_info"].append(memory_info())
                self._data["gpu_info"].append(gpu_util(self._handles))
                self._data["time"] += self._interval
                time.sleep(self._interval)
        else:
            while self._running_event.is_set():
                self._data["cpu_info"].append(cpu_info())
                self._data["memory_info"].append(memory_info())
                self._data["time"] += self._interval
                time.sleep(self._interval)

    def _start(self):
        self._running_event.set()
        self._t = Thread(target=self._monitor_func)
        self._t.start()

    def _end(self):
        self._running_event.clear()
        self._t.join()

    @property
    def gpu_util(self):
        if self._handles:
            return self._data["gpu_util"]
        else:
            return None

    @property
    def cpu_util(self):
        return [x["cpu_util"] for x in self._data["cpu_info"]]

    @property
    def mem_util(self):
        return [x["used"]/x["total"] for x in self._data["memory_info"]]

    @property
    def time(self):
        return self._data["time"]
