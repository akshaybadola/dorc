from typing import List, Dict, Any, Union, Tuple
from threading import Thread, Event
import time
import psutil
import numpy as np
import pynvml as nv


try:
    nv.nvmlInit()
    NVML_PRESENT = True
except nv.NVMLError_LibraryNotFound:
    NVML_PRESENT = False


def cpu_info() -> Dict[str, Union[int, float]]:
    return {"cpu_count": psutil.cpu_count(), "cpu_util": psutil.cpu_percent()}


def memory_info() -> Dict[str, int]:
    info = psutil.virtual_memory()
    return {"total": info.total, "used": info.used}


def gpu_util(handles: Dict[int, Any]) -> Union[Dict[int, Dict[str, Union[int, float]]], None]:
    def _get_util(h):
        util = nv.nvmlDeviceGetUtilizationRates(h)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        return {"gpu": util.gpu, "memory": 100 * (mem.used / mem.total)}
    if NVML_PRESENT:
        return {gpu_id: _get_util(h) for gpu_id, h in handles.items()}
    else:
        return None


def gpu_ranking(handles: Dict[int, Any]) -> Union[Dict[int, int], None]:
    ranking = ["geforce rtx 3080", "geforce rtx 3080", "geforce rtx 2080 ti",
               "geforce gtx 1080 ti", "geforce rtx 2080",
               "geforce rtx 2070", "geforce gtx 1080",
               "geforce gtx 1070", "geforce gtx 1060"]
    ranking.reverse()
    if NVML_PRESENT:
        retval = {}
        for key, handle in handles.items():
            mem = int(np.round(nv.nvmlDeviceGetMemoryInfo(handle).total / 1024 / 1024 / 1024))
            name = nv.nvmlDeviceGetName(handle).decode("utf-8").lower()
            retval[key] = {"compute": ranking.index(name), "memory": mem}
        return retval
    else:
        return None


def gpu_temp(handles: Dict[int, Any]) -> Union[Dict[int, Dict[str, int]], None]:
    def _get_temp(h):
        temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
        thresh = nv.nvmlDeviceGetTemperatureThreshold(
            h, nv.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
        return {"temp": temp, "thresh": thresh}
    if NVML_PRESENT:
        return {gpu_id: _get_temp(h) for gpu_id, h in handles.items()}
    else:
        return None


def all_devices() -> List[int]:
    if not NVML_PRESENT:
        return []
    x = 0
    inds = []
    while True:
        try:
            nv.nvmlDeviceGetHandleByIndex(x)
            inds.append(x)
            x += 1
        except Exception:
            break
    return inds


def useable_devices() -> List[int]:
    if not NVML_PRESENT:
        return []
    x = 0
    inds = []
    while True:
        try:
            handle = nv.nvmlDeviceGetHandleByIndex(x)
            gpu_temp({x: handle})
            inds.append(x)
            x += 1
        except Exception as e:
            if isinstance(e, nv.NVMLError_NotSupported):
                x += 1
                continue
            else:
                break
    return inds


def init_nvml(gpus: List[int]) -> Union[Dict[int, Any], Tuple[None, None]]:
    if not NVML_PRESENT:
        return None, None
    remove = []
    handles = {}
    for x in gpus:
        try:
            handles[x] = nv.nvmlDeviceGetHandleByIndex(x)
            gpu_temp({x: handles[x]})
        except Exception:
            if x in handles:
                handles.pop(x)
            remove.append(x)
    return handles, remove


def gpu_name(handle) -> Union[str, None]:
    if NVML_PRESENT:
        return nv.nvmlDeviceGetName(handle).decode("utf-8")
    else:
        None


class DeviceMonitor:
    def __init__(self, gpu_handles: Dict[int, Any], poll_interval: float = 0.05):
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
            all_keys = set.union(*[set(x.keys()) for x in self._data["gpu_info"]])
            util = {}
            for k in all_keys:
                util[k] = [x[k]["gpu"] for x in self._data["gpu_info"]]
            return util
        else:
            return None

    @property
    def gpu_temp(self) -> Dict[int, Dict[str, int]]:
        if self._handles:
            return gpu_temp(self._handles)
        else:
            return None

    @property
    def gpu_mem_util(self):
        if self._handles:
            all_keys = set.union(*[set(x.keys()) for x in self._data["gpu_info"]])
            mem = {}
            for k in all_keys:
                mem[k] = [x[k]["memory"] for x in self._data["gpu_info"]]
            return mem
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


class Monitor:
    def __init__(self, dm: DeviceMonitor):
        self._dm = dm

    def __enter__(self):
        self._dm._reset()
        self._dm._start()

    def __exit__(self, *args):
        self._dm._end()
