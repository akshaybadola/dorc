from typing import List, Iterable
import os
import sys
import time
import json
import logging
import shutil
import warnings
import numpy
import torch


def diff_as_sets(a: Iterable, b: Iterable) -> set:
    a = set([*a])
    b = set([*b])
    return a - b


def concat(list_var: Iterable[List]) -> List:
    """Concat all items in a given list of lists"""
    temp = []
    for x in list_var:
        temp.extend(x)
    return temp


def deprecated(f):
    warn_str = f"Function {f.__name__} is deprecated."
    warnings.warn(warn_str)
    return f


def _serialize_defaults(x):
    if isinstance(x, numpy.ndarray):
        return json.dumps(x.tolist())
    elif isinstance(x, torch.Tensor):
        return json.dumps(x.cpu().numpy().tolist())
    elif x is None:
        return json.dumps(False)
    else:
        return str(x)


def _dump(x):
    return json.dumps(x, default=_serialize_defaults)
    # return json.dumps(x, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")


def gen_file_logger(logdir, log_file_name):
    logger = logging.getLogger('default_logger')
    formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not log_file_name.endswith('.log'):
        log_file_name += '.log'
    # existing_files = [f for f in os.listdir(logdir) if f.startswith(log_file_name)]
    log_file = os.path.abspath(os.path.join(logdir, log_file_name))
    if os.path.exists(log_file):
        backup_num = get_backup_num(logdir, log_file_name)
        os.rename(log_file, log_file + '.' + str(backup_num))
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return log_file, logger


def get_backup_num(filedir, filename):
    backup_files = [x for x in os.listdir(filedir) if x.startswith(filename)]
    backup_maybe_nums = [b.split('.')[-1] for b in backup_files]
    backup_nums = [int(x) for x in backup_maybe_nums
                   if any([_ in x for _ in list(map(str, range(10)))])]
    if backup_nums:
        cur_backup_num = max(backup_nums) + 1
    else:
        cur_backup_num = 0
    return cur_backup_num


def gen_file_and_stream_logger(logdir, log_file_name, file_loglevel=None,
                               stream_loglevel=None, logger_level=None):
    logger = logging.getLogger('default_logger')
    formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not log_file_name.endswith('.log'):
        log_file_name += '.log'
    log_file = os.path.abspath(os.path.join(logdir, log_file_name))
    if os.path.exists(log_file):
        backup_num = get_backup_num(logdir, log_file_name)
        os.rename(log_file, log_file + '.' + str(backup_num))
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    if stream_loglevel is not None and hasattr(logging, stream_loglevel.upper()):
        stream_handler.setLevel(getattr(logging, stream_loglevel.upper()))
    else:
        stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    if file_loglevel is not None and hasattr(logging, file_loglevel.upper()):
        file_handler.setLevel(getattr(logging, file_loglevel.upper()))
    else:
        file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    if logger_level is not None and hasattr(logging, logger_level.upper()):
        logger.setLevel(getattr(logging, logger_level.upper()))
    else:
        logger.setLevel(logging.DEBUG)
    return log_file, logger



def create_module(module_dir, module_files=[]):
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    if not os.path.exists(os.path.join(module_dir, "__init__.py")):
        with open(os.path.join(module_dir, "__init__.py"), "w") as f:
            f.write("")
    for f in module_files:
        shutil.copy(f, module_dir)


def make_test_daemon(get_cookies=False):
    import requests
    from .daemon import _start_daemon

    port = 23232
    hostname = "127.0.0.1"
    host = "http://" + ":".join([hostname, str(port) + "/"])
    try:
        response = requests.get(host + "_ping", timeout=.5)
        if response.status_code == 200:
            cookies = requests.request("POST", host + "login",
                                       data={"username": "admin",
                                             "password": "AdminAdmin_33"}).cookies
            requests.get(host + "_shutdown", cookies=cookies)
            time.sleep(1)
    except requests.ConnectionError:
        pass
    data_dir = ".test_dir"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    daemon = _start_daemon(hostname, port, ".test_dir")
    time.sleep(.5)
    if get_cookies:
        cookies = requests.request("POST", host + "login",
                                   data={"username": "admin",
                                         "password": "AdminAdmin_33"}).cookies
        return daemon, cookies
    else:
        return daemon


def make_test_interface(setup_path, autoloads_path):
    from threading import Thread
    from .interfaces import FlaskInterface

    hostname = "127.0.0.1"
    port = 12321
    data_dir = ".test_dir"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    iface = FlaskInterface(hostname, port, data_dir, no_start=True)
    create_module(os.path.abspath(os.path.join(data_dir, "global_modules")),
                  [os.path.abspath(autoloads_path)])
    sys.path.append(os.path.abspath(data_dir))
    with open(setup_path, "rb") as f:
        f_bytes = f.read()
        status, message = iface.create_trainer(f_bytes)
    iface_thread = Thread(target=iface.start)
    iface_thread.start()
    time.sleep(1)
    return iface


# def make_apps():
#     global apps
#     apps = {"$iface": make_interface(os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                                   "../tests/_setup_local.py"),
#                                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                                   "autoloads.py")),
#             "$daemon": make_daemon()}
