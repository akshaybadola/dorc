import psutil
import pynvml


def cpu_info():
    return {"cpu_count": psutil.cpu_count(), "cpu_util": psutil.cpu_percent()}


def memory_info(self):
    info = psutil.virtual_memory()
    return {"total": info.total, "used": info.used}


def gpu_util(handles):
    def _get_util(h):
        info = pynvml.nvmlDeviceGetUtilizationRates(h)
        return {"gpu": info.gpu, "memory": info.memory}
    return {gpu_id: _get_util(h) for gpu_id, h in handles.items()}


def init_nvml(gpus):
    pynvml.nvmlInit()
    return {x: pynvml.nvmlDeviceGetHandleByIndex(x) for x in gpus}


def in_thread(f):
    return f


class DevicePoll:
    def __init__(self, gpu_handles):
        self._handles = gpu_handles
        self._interval = 0.2

    @in_thread
    def start(self):
        # get all device info every .2 seconds for the batch
        pass

    @in_thread
    def end(self):
        pass

    # For each property delete after accessed
    @property
    def gpu_util(self):
        pass

    @property
    def gpu_max_mem(self):
        pass

    @property
    def gpu_min_mem(self):
        pass

    @property
    def cpu_util(self):
        pass

    @property
    def cpu_max_mem(self):
        pass

    @property
    def cpu_min_mem(self):
        pass
