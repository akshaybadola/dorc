from typing import List, Dict, Iterable, Any, Callable, Tuple, Optional, Union
import os
import sys
import time
import json
import logging
import shutil
import warnings
import numpy
import torch
from threading import Thread
import requests
from flask import Response


BasicType = Union[str, int, bool]


def identity(x: Any):
    return x


def dget(obj, *args):
    if args:
        return dget(obj.get(args[0]), *args[1:])
    else:
        return obj


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


# def deprecated(f: Union[Callable, property]) -> Union[Callable, property]:
#     if isinstance(f, property):
#         return pdeprecated(f)
#     else:
#         return fdeprecated(f)


def deprecated(f: Callable) -> Callable:
    warn_str = f"Function {f.__name__} is deprecated."
    warnings.warn(warn_str)
    return f


# def pdeprecated(f: property) -> property:
#     warn_str = f"Function {f.fget.__name__} is deprecated."
#     warnings.warn(warn_str)
#     return f


def serialize_defaults(x: Any) -> str:
    if isinstance(x, numpy.ndarray):
        return json.dumps(x.tolist())
    elif isinstance(x, torch.Tensor):
        return json.dumps(x.cpu().numpy().tolist())
    elif x is None:
        return json.dumps(False)
    else:
        return str(x)


def _dump(x: Any) -> str:
    return json.dumps(x, default=serialize_defaults)
    # return json.dumps(x, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")


def make_json(x: Any, dump: bool = True) -> Response:
    return Response(_dump(x) if dump else x, 200, mimetype="application/json")


def gen_file_logger(logdir: str, logger_name: str,
                    log_file_name: str) -> Tuple[str, logging.Logger]:
    logger = logging.getLogger(logger_name)
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


def get_backup_num(filedir: str, filename: str) -> int:
    backup_files = [x for x in os.listdir(filedir) if x.startswith(filename)]
    backup_maybe_nums = [b.split('.')[-1] for b in backup_files]
    backup_nums = [int(x) for x in backup_maybe_nums
                   if any([_ in x for _ in list(map(str, range(10)))])]
    if backup_nums:
        cur_backup_num = max(backup_nums) + 1
    else:
        cur_backup_num = 0
    return cur_backup_num


def gen_file_and_stream_logger(logdir: str, logger_name: str,
                               log_file_name: str,
                               file_loglevel: Optional[str] = None,
                               stream_loglevel: Optional[str] = None,
                               logger_level: Optional[str] = None):
    logger = logging.getLogger(logger_name)
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



def create_module(module_dir: str, module_files: List = []):
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    if not os.path.exists(os.path.join(module_dir, "__init__.py")):
        with open(os.path.join(module_dir, "__init__.py"), "w") as f:
            f.write("")
    for fl in module_files:
        shutil.copy(fl, module_dir)


def stop_test_daemon(port=23232):
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


def make_test_daemon(hostname="127.0.0.1", port=23232,
                     root_dir=".test_dir", name="test_daemon",
                     get_cookies=False, no_clear=False):
    import requests
    from .daemon import Daemon

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
    if os.path.exists(root_dir) and not no_clear:
        shutil.rmtree(root_dir)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    daemon = Daemon(hostname, port, root_dir, name)
    thread = Thread(target=daemon.start)
    thread.start()
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
    gmods_dir = os.path.abspath(os.path.join(data_dir, "global_modules"))
    gdata_dir = os.path.abspath(os.path.join(data_dir, "global_datasets"))
    create_module(gmods_dir,
                  [os.path.abspath(autoloads_path)])
    create_module(gdata_dir)
    sys.path.append(os.path.abspath(data_dir))
    iface = FlaskInterface(hostname, port, data_dir, gmods_dir, gdata_dir, no_start=True)
    setup_dir = os.path.dirname(setup_path)
    sys.path.append(setup_dir)
    sfile = setup_path.split("/")[-1].replace(".py", "")
    ldict = {}
    exec(f"from {sfile} import config", globals(), ldict)
    config = ldict["config"]
    sys.path.remove(setup_dir)
    with open(data_dir + "/config.json", "w") as f:
        json.dump(config, f)
    status, message = iface.create_trainer()
    if status:
        iface_thread = Thread(target=iface.start)
    else:
        print("Could not create trainer")
        sys.exit(1)
    iface_thread.start()
    time.sleep(1)
    return iface


def stop_test_interface():
    print(requests.get("http://127.0.0.1:12321/" + "_shutdown").content)


def exec_and_return(exec_str: str,
                    modules: Optional[Dict[str, Any]] = None) -> Any:
    """Execute the exec_str with :meth:`exec` and return the value

    Args:
        exec_str: The string to execute

    Returns:
        The value in the `exec_str`

    """
    ldict: Dict[str, Any] = {}
    if modules is not None:
        exec("testvar = " + exec_str, {**modules, **globals()}, ldict)
    else:
        exec("testvar = " + exec_str, globals(), ldict)
    retval = ldict['testvar']
    return retval


def recurse_multi_dict(jdict: Dict[str, Any],
                       preds: List[Callable[[str, Any], str]],
                       repls: Dict[str, Callable[[str], str]]) -> Dict[str, Any]:
    """Recurse over a :class:`dict` and perform replacement.

    This function replaces the values of the dictionary in place. It's used to
    fix the generated OpenAPI schema

    Args:
        jdict: A dictionary
        pred: Predicate to check when to perform replacement
        repl: Function which performs the replacement

    Returns:
        A modified dictionary

    """
    if not (isinstance(jdict, dict) or isinstance(jdict, list)):
        return jdict
    if isinstance(jdict, dict):
        for k, v in jdict.items():
            if isinstance(v, dict):
                jdict[k] = recurse_multi_dict(v, preds, repls)
            if isinstance(v, list):
                for i, item in enumerate(v):
                    v[i] = recurse_multi_dict(item, preds, repls)
            for pred in preds:
                result = pred(k, v)
                if result:
                    jdict[k] = repls[result](v)
    elif isinstance(jdict, list):
        for i, item in enumerate(jdict):
            jdict[i] = recurse_multi_dict(item, preds, repls)
    return jdict


def recurse_dict(jdict: Dict[str, Any],
                 pred: Callable[[str, Any], bool],
                 repl: Callable[[str, str], str],
                 repl_only: bool = False) -> Dict[str, Any]:
    """Recurse over a :class:`dict` and perform replacement.

    This function replaces the values of the dictionary in place. Used to
    fix the generated schema :class:`dict`.

    Args:
        jdict: A dictionary
        pred: Predicate to check when to perform replacement
        repl: Function which performs the replacement

    Returns:
        A modified dictionary

    """
    if not (isinstance(jdict, dict) or isinstance(jdict, list)):
        return jdict
    if isinstance(jdict, dict):
        for k, v in jdict.items():
            if pred(k, v):
                jdict[k] = repl(k, v)
                if repl_only:
                    continue
            if isinstance(v, dict):
                jdict[k] = recurse_dict(v, pred, repl, repl_only)
            if isinstance(v, list):
                for i, item in enumerate(v):
                    v[i] = recurse_dict(item, pred, repl, repl_only)
    elif isinstance(jdict, list):
        for i, item in enumerate(jdict):
            jdict[i] = recurse_dict(item, pred, repl, repl_only)
    return jdict


def pop_if(jdict: Dict[str, Any], pred: Callable[[str, Any], bool]) -> Dict[str, Any]:
    """Pop a (key, value) pair based on predicate `pred`.

    Args:
        jdict: A dictionary
        pred: According to which the value is popped

    Returns:
        A :class:`dict` of popped values.

    """
    to_pop = []
    popped = {}
    for k, v in jdict.items():
        if pred(k, v):
            to_pop.append(k)
        if isinstance(v, dict):
            popped.update(pop_if(v, pred))
    for p in to_pop:
        popped.update(jdict.pop(p))
    return popped


# def make_apps():
#     global apps
#     apps = {"$iface": make_interface(os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                                   "../tests/_setup_local.py"),
#                                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                                   "autoloads.py")),
#             "$daemon": make_daemon()}
