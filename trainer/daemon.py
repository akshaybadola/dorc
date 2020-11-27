from typing import List, Dict, Any, Union, Callable
import os
import sys
import ssl
import glob
import json
import time
import socket
import shutil
import shlex
import atexit
import requests
import argparse
import datetime
import logging
import hashlib
import traceback
import configparser
import zipfile
import pathlib
from queue import Queue
from threading import Thread
import multiprocessing as mp
from subprocess import Popen, PIPE, TimeoutExpired
from markupsafe import escape
from functools import partial

import flask_login
from flask import Flask, render_template, request, Response
from flask_cors import CORS
from werkzeug import serving

from .version import __daemon__version__
from .mods import Modules
from .helpers import Tag
from .interfaces import FlaskInterface
# from .util import _dump as dump
from .util import _dump
from ._log import Log
from .auth import __unti__, __inti__, User


session_method = Tag("session_method")


# def _dump(*args):
#     if len(args) > 1:
#         if args[0] in [True, False]:
#             dump({"status": args[0], "payload": args[1:]})
#         else:
#             dump(args)
#     else:
#         if isinstance(args, list) and args[0] in [True, False]:
#             dump({"status": args[0], "payload": args[1:]})
#         else:
#             dump(args)


def get_hostname() -> str:
    p = Popen("hostname", stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return out.decode("utf-8")


def check_ssh_port(host: str, port: int) -> int:
    timeout = 2
    while True:
        print(f"Checking port {port}")
        out, err = b"", b""
        ip_addr = host.split("@")[1]
        p = Popen(f"nc -z -v {ip_addr} 22", shell=True, stdout=PIPE, stderr=PIPE)
        try:
            out, err = p.communicate(timeout=timeout)
        except TimeoutExpired:
            port = "UNREACHABLE"
            break
        p = Popen(shlex.split(f"ssh -R {port}:localhost:20202 {host} hostname"),
                  stdout=PIPE, stderr=PIPE)
        try:
            out, err = p.communicate(timeout=timeout)
        except Exception:
            pass
        print(f"Got values {out}, {err}")
        if out.decode("utf-8") and "warn" in err.decode("utf-8").lower():
            port += 101
        elif out.decode("utf-8") and not err.decode("utf-8").lower():
            break
        p.kill()
    return port


def have_internet():
    auth_cmd = ('curl -L -k -d username="15mcpc15" -d password="unmission@123"' +
                ' -d mode=191 http://192.168.56.2:8090/login.xml')

    def communicate(p, vals):
        vals['out'], vals['err'] = p.communicate()

    def connect(auth_cmd):
        vals = {'out': None, 'err': None}
        p = Popen(auth_cmd, shell=True, stdout=PIPE, stderr=PIPE)
        t = Thread(target=communicate, args=[p, vals])
        t.start()
        t.join(timeout=5)
        p.kill()
        if vals['out'] and "You have successfully logged in" in vals['out'].decode('utf-8'):
            return True
        else:
            return False

    while True:
        vals = {'out': None, 'err': None}
        p = Popen("curl google.com".split(), stdout=PIPE, stderr=PIPE)
        t = Thread(target=communicate, args=[p, vals])
        t.start()
        t.join(timeout=5)
        p.kill()
        if vals['out']:
            if "the document has moved" not in vals['out'].decode('utf-8').lower():
                connect(auth_cmd)
        time.sleep(60)


def register_with_tracker(tracker, host, port):
    status = False
    fwd_port = 11111
    procs = []
    while not status:
        procs.append(Popen(shlex.split(f"ssh -N -L {fwd_port}:localhost:11111 {tracker}"),
                           stdout=PIPE, stderr=PIPE))
        time.sleep(3)
        try:
            print(f"Registering port {port} at {tracker}")
            resp = requests.request("POST", f"http://localhost:{fwd_port}/",
                                    json={"put": True,
                                          "hostname": host,
                                          "port": port}).content
            status = True
        except requests.ConnectionError as e:
            print(f"Connection refused from server {e}")
            resp = None
            status = True
        except Exception as e:
            print(f"Register request at port {fwd_port} with {tracker} failed {e}. Trying again")
            resp = None
    for p in procs:
        p.kill()
    return resp


def create_module(module_dir, module_files=[]):
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    if not os.path.exists(os.path.join(module_dir, "__init__.py")):
        with open(os.path.join(module_dir, "__init__.py"), "w") as f:
            f.write("")
    for f in module_files:
        shutil.copy(f, module_dir)


Path = Union[str, pathlib.Path]


class Daemon:
    __version__ = __daemon__version__

    def __init__(self, hostname: str, port: int, data_dir: Path,
                 production: bool = False, template_dir: Union[Path, None] = None,
                 static_dir: Union[Path, None] = None, root_dir: Union[Path, None] = None,
                 trackers: List[str] = [], daemon_name: Union[str, None] = None,
                 register: bool = True):
        self.ctx = mp.get_context("spawn")
        self._hostname = hostname
        self._port = port
        self._trackers = trackers
        self.daemon_name = daemon_name
        self.register = register
        # NOTE: fwd_hosts and fwd_ports are hard coded
        self.fwd_port_start = 8181    # starts with 8181
        if "droid" not in get_hostname().lower():
            self._have_internet = mp.Process(target=have_internet)
            self._have_internet.start()
        else:
            self._have_internet = None
        self.data_dir = os.path.abspath(data_dir)
        # NOTE: init data_dir
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.tmp_dir = os.path.join(self.data_dir, ".tmp")
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)
        self.production = production
        # FIXME: Code duplication here
        # NOTE: init modules_dir
        self.modules_dir = os.path.join(self.data_dir, "global_modules")
        create_module(self.modules_dir,
                      [os.path.join(os.path.dirname(__file__), "autoloads.py")])
        # NOTE: Append data_dir path
        self.env_str = f"""
import sys
sys.path.append("{self.data_dir}")
"""
        # NOTE: init datasets_dir
        self.datasets_dir = os.path.join(self.data_dir, "global_datasets")
        create_module(self.datasets_dir)
        # NOTE: Set exclude_dirs, dirs not to scan for sessions
        self._exclude_dirs = [*map(os.path.basename,
                                   [self.modules_dir, self.datasets_dir,
                                    self.tmp_dir])]
        self._session_exclude_dirs = ["modules", "datasets"]
        # NOTE: init template and static dirs
        if template_dir is None:
            self._template_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "dist")
        else:
            self._template_dir = template_dir
        if static_dir is None:
            self._static_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "dist")
        else:
            self._static_dir = static_dir
        # NOTE: root_dir is relative (CHECK why?)
        if root_dir is None:
            self._root_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
        else:
            self._root_dir = root_dir
        if not os.path.exists(self._template_dir) or not os.path.exists(self._static_dir)\
           or not self._root_dir:
            print("FATAL ERROR! Cannot initialize with template, static and root dirs")
            return
        self.app = Flask(__name__, static_folder=self._static_dir,
                         template_folder=self._template_dir)
        # NOTE: FIXME Fix for CSRF etc.
        #       see https://flask-cors.corydolphin.com/en/latest/api.html#using-cors-with-cookies
        CORS(self.app, supports_credentials=True)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        # NOTE: Not sure if this is really useful
        self.app.secret_key = __unti__("_sxde#@_")
        self.use_https = False
        self.verify_user = True
        self._last_free_port = self.port
        # NOTE: Daemon Resources
        self._threads = {}
        self._task_q = Queue()
        self._sessions = {}
        self._devices = {}
        self._modules = {}
        self._datasets = {}
        self._init_context()
        self._task_id = 0
        self.__task_ids = []
        self._results = []
        # NOTE: Logger
        self._logger = logging.getLogger("daemon_logger")
        log_file = os.path.join(self.data_dir, "logs")
        formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.DEBUG)
        log = Log(self._logger, self.production)
        self._logd = log._logd
        self._loge = log._loge
        self._logi = log._logi
        self._logw = log._logw
        # NOTE: Module Loader
        self._module_loader = Modules(self.data_dir, self._logd, self._loge,
                                      self._logi, self._logw)
        # NOTE: Initialize modules and datasets
        self._load_available_global_modules()
        self._load_available_global_datasets()
        self._logi("Initialized Daemon")
        # NOTE: Initialize login manager
        self.login_manager = flask_login.LoginManager()
        self.login_manager.init_app(self.app)
        # NOTE: login_view just creates a redirect. Annoying.
        # self.login_manager.login_view = "__login"
        try:
            self._ids = __ids__
            self._users = __users__
        except NameError:
            self._ids = {0: "admin", 1: "joe"}
            self._users = {"admin": User(0, "admin"),
                           "joe": User(1, "joe")}
        self._passwords = lambda x: __inti__(x)     # {"admin": "admin", "joe": "admin"}
        self._fwd_ports_event = None
        self._fwd_ports_thread = None
        self._fwd_procs = None
        if self.register:
            self._fwd_ports = {}
            self._fwd_procs = {}
            self._fwd_ports_event = mp.Event()
            self.fwd_ports_event.set()
            self._fwd_ports_thread = mp.Process(target=self.fwd_ports_func)
            self.fwd_ports_thread.start()
        else:
            print(f"Not registering with trackers")
        # self._fwd_ports()

    @property
    def hostname(self) -> str:
        "Hostname on which to serve"
        return self._hostname

    @property
    def port(self) -> int:
        "port on which to listen"
        return self._port

    @property
    def fwd_ports(self) -> Dict:
        "A :class:`dict` mapping trackers and ports forwarded to them"
        return self._fwd_ports

    @property
    def fwd_procs(self) -> Dict:
        "A :class:`dict` mapping trackers and SSH :class:`subprocess.Popen` processes"
        return self._fwd_procs

    @property
    def fwd_ports_event(self) -> mp.Event:
        """Event :class:`multiprocessing.Event` which controls `self.fwd_port_thread`"""
        return self._fwd_ports_event

    @property
    def fwd_ports_thread(self) -> mp.Process:
        """Process :class:`multiprocessing.Process` which checks if the ports are
        correctly forwarded to the trackers.

        """
        return self._fwd_ports_thread

    @property
    def trackers(self) -> List[str]:
        """List of user@host strings where a tracker is present.

        Trackers are http servers which map the hostnames to forwarded ports on
        that machine. Each `daemon` when started, can register with a list of
        trackers and forward its ports. The trackers can then be used to forward
        those ports back to user machine. Convoluted I know.

        """
        return self._trackers

    @property
    def reserved_devices(self) -> List[int]:
        devices = []
        for x in self._devices.values():
            devices.extend(x)
        return devices

    def fwd_ports_func(self) -> None:
        """Forward ports at one minute interval to `self.trackers`

        This function runs in a separate process and checks the SSH forwarded
        ports and forwards any stale/dead ports if required.

        """
        while self.fwd_ports_event.is_set():
            self._logi("Checking port forwards")
            self.check_and_register_with_trackers()
            if not self.fwd_ports_event.is_set():
                self._logi("Exiting from fwd_ports_func")
                return
            else:
                time.sleep(60)

    # FIXME: This thing will start in a thread and if this is available to call
    #        from an endpoint then there could be race conditions
    def check_and_register_with_trackers(self):
        """Checks if the all the ports are correctly forwarded.

        It checks all the ports with internal functions which use various shell
        commands. If any port is not correctly forwarded it forwards that port
        and registers that port correctly with the tracker.

        """
        if not self.trackers:
            self._logd(f"Empty tracker list. Will not do do anything")
            return
        if self.daemon_name is None:
            try:
                with open("daemon_name", "r") as f:
                    daemon_name = f.read().split("\n")[0].strip()
            except Exception:
                daemon_name = "No Nmae"
        else:
            daemon_name = self.daemon_name

        def _check_fwd_port(host, port):
            if port == "UNREACHABLE":
                return False
            try:
                self._logd(f"Checking {host}:{port}")
                p = Popen(shlex.split(f"ssh {host} \"curl http://localhost:{port}/_name\""),
                          stdout=PIPE, stderr=PIPE)
                out, err = p.communicate(timeout=3)
                if daemon_name == out.decode("utf-8"):
                    return True
                else:
                    self._logd(f"Incorrect port registerd with tracker {host}")
                    return False
            except TimeoutExpired:
                p.kill()
                return False

        def _fwd_port(host):
            if host in self.fwd_procs:
                self.fwd_procs[host].kill()
            self.fwd_ports[host] = port = check_ssh_port(host, self.fwd_port_start)
            if port != "UNREACHABLE":
                self.fwd_procs[host] = Popen(shlex.split(f"ssh -N -R {port}:localhost:20202 {host}"),
                                             stdout=PIPE, stderr=PIPE)
            return port

        def _register(host, daemon_name, port):
            resp = register_with_tracker(tracker, daemon_name, port)
            if resp is not None:
                self._logi(f"Forwarded port {port}, with name {daemon_name} to {tracker}." +
                           f"Response is {resp}")
            else:
                self._loge(f"Connection error from {daemon_name}. Could not forward port")

        _check = self.fwd_ports_event.is_set()
        print(f"Checking ports and Registering with {self.trackers}")
        if _check:
            for tracker in self.trackers:
                if tracker in self.fwd_ports:
                    port = self.fwd_ports[tracker]
                    if port == "UNREACHABLE":
                        # NOTE: Skip if unreachable
                        continue
                    elif _check_fwd_port(tracker, port):
                        _register(tracker, daemon_name, port)
                    else:
                        self._logi(f"Bad port {port}, with {daemon_name} to {tracker}.")
                        new_port = _fwd_port(tracker)
                        if new_port == "UNREACHABLE":
                            continue
                        else:
                            _register(tracker, daemon_name, new_port)
                else:
                    print(f"Finding new port for {tracker}")
                    port = _fwd_port(tracker)
                    if port == "UNREACHABLE":
                        continue
                    else:
                        _register(tracker, daemon_name, port)

    def _init_context(self):
        self.api_crt = "res/server.crt"
        self.api_key = "res/server.key"
        self.api_ca_cert = "res/ca-crt.pem"
        if self.use_https:
            self.context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            if self.verify_user:
                self.context.verify_mode = ssl.CERT_REQUIRED
                self.context.load_verify_locations(self.api_ca_cert)
            try:
                self.context.load_cert_chain(self.api_crt, self.api_key)
            except Exception as e:
                sys.exit("Error starting flask server. " +
                         f"Missing cert or key. Details: {e}" + f"\n{traceback.format_exc()}")
        else:
            self.context = None

    def _update_results(self):
        while not self._task_q.empty():
            self._results.append(self._task_q.get())

    def _check_config(self, config) -> bool:
        self._logw(f"This is a placeholder function")
        # Need python file or module, that's it
        return True

    def _find_open_port(self):
        if not self._last_free_port:
            self._last_free_port = 1
        flag = True
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while flag:
            self._last_free_port = (self._last_free_port + 1) % 65532
            flag = s.connect_ex(('localhost', self._last_free_port)) == 0
        s.close()
        return self._last_free_port

    def _create_id(self) -> int:
        # 0 reserverd for instance
        self._task_id += 1
        self.__task_ids.append(self._task_id)
        return self._task_id

    def _get_task_id_launch_func(self, func: Callable, *args):
        task_id = self._create_id()
        self._threads[task_id] = Thread(target=func, args=[task_id, *args])
        self._threads[task_id].start()
        return task_id

    def _check_result(self, task_id: int) -> Any:
        self._update_results()
        for x in self._results:
            if task_id == x[0]:
                return x
        else:
            return None

    def _wait_for_task(self, func: Callable, task_id: int, args) -> Any:
        func(task_id, *args)
        result = self._check_result(task_id)
        while result is None:
            time.sleep(1)
            result = self._check_result(task_id)
        return result[1]

    def _update_init_file(self, init_file: str, module_names: List[str]):
        lines = []
        for m in module_names:
            lines.append(f"from . import {m}\n")
        with open(init_file, "w") as f:
            f.writelines(lines)

    def _load_available_global_modules(self):
        mods_dir = self.modules_dir
        self._modules = self._module_loader.read_modules_from_dir(
            mods_dir, excludes=[lambda x: x.startswith("__")])
        self._update_init_file(os.path.join(self.modules_dir, "__init__.py"),
                               self._modules.keys())

    def _load_available_global_datasets(self):
        json_filenames = [x for x in os.listdir(self.datasets_dir)
                          if x.endswith(".json")]
        for x in json_filenames:
            with open(os.path.join(self.datasets_dir, x)) as f:
                self._datasets[x.replace(".json", "")] = json.load(f)
        self._update_init_file(os.path.join(self.datasets_dir, "__init__.py"),
                               self._modules.keys())

    def _dataset_valid_p(self, data_dict):
        if len(data_dict.keys()) != 1:
            return False, "Too many modules. Shouldn't happen."
        if "dataset" not in [*data_dict.values()][0]:
            return False, "'dataset' not in module"
        else:
            if self.datasets_dir not in sys.path:
                sys.path.append(self.datasets_dir)
            mod_name = [*data_dict.keys()][0]
            ldict = {}
            exec(f"import {mod_name}", globals(), ldict)
            dataset = ldict[mod_name]
            if not hasattr(dataset, "dataset"):
                del dataset, ldict
                return False, f"variable 'dataset' not in {mod_name}"
            else:
                del dataset, ldict
                return True, f"Added dataset {mod_name}"
            # check for splits?
            # NOTE: Old code which checks for __len__ and __getitem__
            # # NOTE: Conflicts can arrive in datasets and modules
            # #       Should append _dataset to every dataset
            # mod_name = [*data_dict.keys()][0]
            # ldict = {}
            # exec(f"import {mod_name}", globals(), ldict)
            # dataset = ldict[mod_name]
            # x = dataset.dataset
            # if not hasattr(x, "__len__"):
            #     return False, f"{mod_name}.dataset doesn't have __len__"
            # elif not hasattr(x, "__getitem__"):
            #     return False, f"{mod_name}.dataset doesn't have __getitem__"
            # else:
            #     return True, f"Added dataset {mod_name}"

    def _get_with_prefix(self, name, prefix):
        name = name.lstrip("_")
        return f"_{prefix}_" + name.lstrip(prefix).lstrip("_")

    def _get_module_name(self, mod_name):
        return self._get_with_prefix(mod_name, "module")

    def _get_dataset_name(self, mod_name):
        return self._get_with_prefix(mod_name, "dataset")

    def _load_dataset(self, task_id, data):
        name = self._get_dataset_name(data["name"])
        desc = data["description"]
        dtype = data["type"]
        file_bytes = data["data_file"]
        data_dir = self.datasets_dir
        result = self._module_loader.add_named_module(data_dir, file_bytes, name)
        if result[0]:
            status, message = self._dataset_valid_p(result[1])
            if status:
                self._datasets[name] = {}
                self._datasets[name]["members"] = result[1][name]
                self._datasets[name]["description"] = desc
                self._datasets[name]["type"] = dtype
                with open(os.path.join(data_dir, name + ".json"), "w") as f:
                    json.dump({name: self._datasets[name]}, f)
                self._task_q.put((task_id, True, message))
            else:
                self._task_q.put((task_id, status, message))
        else:
            self._error_and_put(task_id, False, f"Could not add dataset. {result}")

    def _load_module(self, task_id, data):
        mod_name = self._get_module_name(data["name"])
        mod_file = data["data_file"]
        print("MODULE name", mod_name)
        mods_dir = self.modules_dir
        status, result = self._module_loader.add_named_module(mods_dir, mod_file, mod_name)
        if status:
            self._modules.update(result)
            self._task_q.put((task_id, True))
        else:
            self._error_and_put(task_id, False, f"Could not load module. {result}")

    # The following line should execute in the session env
    # self._read_modules_from_dir(mods_dir)
    def _load_available_session_modules(self, session_dir):
        mods_dir = self.modules_dir
        if not os.path.exists(mods_dir):
            self._logd("No existing modules dir. Creating")
            os.mkdir(mods_dir)
        raise NotImplementedError

    # TODO: scan_sessions should be called at beginning and after that should
    #       raise error (or atleast warn) unless testing
    def scan_sessions(self):
        self._logd("Scanning Sessions")
        session_names = [x for x in os.listdir(self.data_dir) if
                         os.path.isdir(os.path.join(self.data_dir, x))
                         and x not in self._exclude_dirs]
        for s in session_names:
            self._sessions[s] = {}
            self._sessions[s]["path"] = os.path.join(self.data_dir, s)
            self._sessions[s]["sessions"] = {}
            for d in os.listdir(self._sessions[s]["path"]):
                if d not in self._session_exclude_dirs:
                    try:
                        data_dir = os.path.join(self._sessions[s]["path"], d)
                        self._sessions[s]["sessions"][d] = {}
                        self._sessions[s]["sessions"][d]["data_dir"] = data_dir
                        with open(os.path.join(self._sessions[s]["path"], d, "session_state"),
                                  "r") as f:
                            self._sessions[s]["sessions"][d]["state"] = json.load(f)
                    except Exception as e:
                        self._sessions[s]["sessions"][d] = "Error " + str(e) +\
                            "\n" + traceback.format_exc()

    def create_session(self, task_id, data):
        """Creates a new training session from given data

        A session has a structure `session[key]` where `key` in `{"path", "sessions",
        "modules"}` Modules are loaded from the module path which is appended to
        sys.path for that particular session. Each module has a separate
        namespace as such and since each trainer instance is separate, it should
        be easy to separate. The modules are shared among all the subsessions.

        """
        session_name = data["name"]
        if session_name not in self._sessions:
            self._sessions[session_name] = {}
            self._sessions[session_name]["path"] = os.path.join(self.data_dir, session_name)
            self._sessions[session_name]["sessions"] = {}
            self._sessions[session_name]["modules"] = {}
            if not os.path.exists(self._sessions[session_name]["path"]):
                os.mkdir(self._sessions[session_name]["path"])  # /sessions/funky_session
            modules_path = os.path.join(self._sessions[session_name]["path"], "modules")
            if not os.path.exists(modules_path):
                os.mkdir(modules_path)  # /sessions/funky_session/modules
        time_str = datetime.datetime.now().isoformat()
        # Like: /sessions/funky_session/2020-02-17T10:53:06.458827
        self._logd(f"Trying to create session {session_name}/{time_str}")
        data_dir = os.path.join(self._sessions[session_name]["path"], time_str)
        os.mkdir(data_dir)
        self._sessions[session_name]["sessions"][time_str] = {}
        overrides = None
        if "overrides" in data:
            overrides = data["overrides"]
        if self._wait_for_task(self._create_trainer, task_id,
                               args=[session_name, time_str, data_dir,
                                     data["config"], overrides]):
            with open(os.path.join(self._sessions[session_name]["path"], time_str,
                                   "session_state")) as f:
                self._logd(f"Loading sessions state")
                self._sessions[session_name]["sessions"][time_str]["state"] = json.load(f)
            # NOTE: For cloning/migrating sessions
            if "saves" in data and data["saves"]:
                self._logd(f"Copying save files: {data['saves'].keys()}")
                savedir = os.path.join(data_dir, "savedir")
                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                for fname, fbytes in data["saves"].items():
                    with open(os.path.join(savedir, fname), "wb") as f:
                        f.write(fbytes)
        else:
            self._logd(f"Failed task {task_id}. Cleaning up")
            self._sessions[session_name]["sessions"].pop(time_str)
            shutil.rmtree(data_dir)

    # NOTE: Only load_session sends load=True to _create_trainer
    def _create_trainer(self, task_id, name, time_str, data_dir, config,
                        overrides=None, load=False):
        self._logd(f"Trying to create trainer with data_dir {data_dir}")
        if not os.path.isabs(data_dir):
            data_dir = os.path.abspath(data_dir)
        if not load and self._check_config(config):  # create but don't load
            try:
                self._logd(f"Adding new config")
                iface = FlaskInterface(None, None, data_dir, production=self.production,
                                       config_overrides=overrides)
                status, result = iface.check_config(config, env=self.env_str)
                # print("IFACE", status, result, data_dir)
                # print("CONFIG EXISTS", os.path.exists(os.path.join(data_dir)),
                #       (os.path.exists(os.path.join(data_dir, "session_config"))
                #        or os.path.exists(os.path.join(data_dir, "session_config.py"))))
                # status, result = self._modules.add_config(data_dir, config)
                if status:
                    # trainer = Trainer(**{"data_dir": data_dir, **result})
                    # trainer._init_all()
                    status, result = iface.create_trainer()
                    # print("URGH", status, result)
                    del iface
                    self._task_q.put((task_id, True))
                else:
                    self._error_and_put(task_id, False, f"Could not read config. {result}")
            except Exception as e:
                self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())
        elif load and self._check_config(config):  # create and load
            try:
                self._logd(f"Config already existed")
                port = self._find_open_port()
                self._sessions[name]["sessions"][time_str]["config"] = config
                self._sessions[name]["sessions"][time_str]["port"] = port
                self._sessions[name]["sessions"][time_str]["data_dir"] = data_dir
                if overrides:
                    with open(os.path.join(data_dir, "config_overrides.json")) as f:
                        json.dump(overrides, f)
                cmd = f"python if_run.py {self.hostname} {port} {data_dir} {self.production} " +\
                    "--config-overrides True"
                cwd = self._root_dir
                self._logd(f"Running command {cmd} in {cwd}")
                p = Popen(shlex.split(cmd), env=os.environ, cwd=cwd)
                self._sessions[name]["sessions"][time_str]["process"] = p
                Thread(target=p.communicate).start()
                # print("Popen?", type(p))
                self._task_q.put((task_id, True))
            except Exception as e:
                self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())
        else:
            self._error_and_put(task_id, False, "Check failed on config")

    def _refresh_state(self, session_key):
        name, time_str = session_key.split("/")
        with open(os.path.join(self._sessions[name]["path"], time_str,
                               "session_state"), "r") as f:
            self._sessions[name]["sessions"][time_str]["state"] = json.load(f)

    def _unload_finished_session(self, session_key):
        self._logd(f"Unloading finished session {session_key}")
        self._refresh_state(session_key)
        name, time_str = session_key.split("/")
        state = self._sessions[name]["sessions"][time_str]["state"]
        if self._session_finished_p(state):
            self._unload_session_helper(-1, name, time_str)

    def _refresh_all_loaded_sessions(self):
        self._logd("Refreshing all loaded sessions' states")
        for name in self._sessions:
            for time_str, sess in self._sessions[name]["sessions"].items():
                if "process" in sess:
                    self._refresh_state("/".join([name, time_str]))

    def load_unfinished_sessions(self):
        self._logd("Loading Unfinished Sessions")
        for name, session in self._sessions.items():
            try:
                for sub_name, sub_sess in session["sessions"].items():
                    state = sub_sess["state"]
                    if not self._session_finished_p(state):
                        self._load_session_helper(0, name, sub_name)
            except Exception as e:
                self._loge(f"Could not load session {name}. Error {e}" +
                           "\n" + traceback.format_exc())
                continue

    def _session_finished_p(self, state) -> bool:
        epoch = state["epoch"]
        max_epochs = state["trainer_params"]["max_epochs"]
        iterations = state["iterations"]
        max_iterations = state["trainer_params"]["max_iterations"]
        if (not iterations and not max_iterations and epoch < max_epochs) or\
           (not epoch and not max_epochs and iterations < max_iterations):
            return False
        else:
            return True

    def _session_alive_p(self, session_name: str, timestamp: str) -> bool:
        "Check if a given session is still alive"
        status = self._check_session_valid(session_name, timestamp)
        if not status[0]:
            return status
        else:
            retcode = self._sessions[session_name]["sessions"][timestamp]["process"].poll()
            if retcode is None:
                return True, True
            else:
                return False, retcode

    def _check_session_valid(self, session_name, timestamp):
        if session_name not in self._sessions:
            return False, f"Unknown session, {session_name}"
        elif (session_name in self._sessions and
              timestamp not in self._sessions[session_name]["sessions"]):
            return False, f"Given session instance not in sessions"
        elif not os.path.exists(os.path.join(self.data_dir, "/".join([session_name, timestamp]))):
            return False, f"Session has been deleted"
        else:
            return True, None

    def compare_sessions(self):
        pass

    @property
    def _sessions_list(self):
        # return _dump(self._sessions)
        retval = {}
        for k, v in self._sessions.items():
            session_stamps = v["sessions"].keys()
            for ts in session_stamps:
                key = k + "/" + ts
                session = v["sessions"][ts]
                retval[key] = {}
                retval[key]["loaded"] = "process" in session and session["process"].poll() is None
                retval[key]["port"] = session["port"] if retval[key]["loaded"] else None
                retval[key]["state"] = session["state"]
                retval[key]["finished"] = self._session_finished_p(session["state"])
        return retval

    def _check_username_password(self, data):
        if (data["username"] in self._users):
            __hash = hashlib.sha1(("2ads;fj4sak#)" + data["username"])
                                  .encode("utf-8")).hexdigest()
            if data["password"] == self._passwords(__hash):
                return True, self._users[data["username"]]
            else:
                return False, None
        else:
            return False, None

    def _get_config_file(self, name, time_str=None):
        if time_str is not None:
            data_dir = os.path.join(self.data_dir, name, time_str)
        else:
            data_dir = os.path.join(self.data_dir, name)
        if os.path.exists(os.path.join(data_dir, "session_config.py")):
            config_file = os.path.join(data_dir, "session_config.py")
        elif os.path.exists(os.path.join(data_dir, "session_config", "__init__.py")):
            root_dir = os.path.join(data_dir, "session_config/")
            config_file = os.path.join(self.tmp_dir, "_".join([name, time_str]) + ".zip")
            zf = zipfile.ZipFile(config_file, "w", compression=zipfile.ZIP_STORED)
            for x in glob.glob(os.path.join(root_dir, "**"), recursive=True):
                if not x.endswith(".pyc") and x.replace(root_dir, ""):
                    zf.write(x, arcname=x.replace(root_dir, ""))
            zf.close()
        return config_file

    def _error_and_put(self, task_id, status, message):
        self._loge(f"Error occured {message}")
        self._task_q.put((task_id, status, message))

    def _info_and_put(self, task_id, status, message):
        self._logi(message)
        self._task_q.put((task_id, status, message))

    def _debug_and_put(self, task_id, status, message):
        self._logd(message)
        if task_id is not None:
            self._task_q.put((task_id, status, message))

    @property
    def _session_methods(self):
        return [x[1:].replace("_helper", "") for x in session_method.names]

    @session_method
    def _load_session_helper(self, task_id, name, time_str, data=None):
        key = name + "/" + time_str
        data_dir = os.path.join(self.data_dir, name, time_str)
        config_candidates = [x for x in os.listdir(data_dir)
                             if "session_config" in x and
                             not x.endswith(".bak")]
        if not len(config_candidates) == 1:
            self._error_and_put(task_id, False, f"More than one config detected for {key}" +
                                f"{config_candidates}")
        else:
            self._logd(f"Checks passed. Creating session {key}")
        try:
            self._sessions[name]["sessions"][time_str] = {}
            self._create_trainer(task_id, name, time_str, data_dir, config=None, load=True)
            with open(os.path.join(self._sessions[name]["path"], time_str,
                                   "session_state"), "r") as f:
                self._sessions[name]["sessions"][time_str]["state"] = json.load(f)
            self._debug_and_put(task_id, True, f"Loaded Session for {key}")
        except Exception as e:
            self._error_and_put(task_id, False, f"Could not load Session for {key}. {e}"
                                + "\n" + traceback.format_exc())

    @session_method
    def _unload_session_helper(self, task_id, name, time_str=None, data=None):
        def _unload(name, time_str):
            self._logd(f"Unloading {name}/{time_str}")
            if "process" in self._sessions[name]["sessions"][time_str]:
                self._sessions[name]["sessions"][time_str]["process"].terminate()
                self._sessions[name]["sessions"][time_str].pop("process")
            self._sessions[name]["sessions"][time_str]["config"] = None
            self._sessions[name]["sessions"][time_str].pop("config")
            if "port" in self._sessions[name]["sessions"][time_str]:
                port = self._sessions[name]["sessions"][time_str].pop("port")
                if port in self._devices:
                    self._devices.pop(port)
            if "iface" in self._sessions[name]["sessions"][time_str]:
                self._sessions[name]["sessions"][time_str].pop("iface")
        if time_str is None:
            for k in self._sessions[name]["sessions"].keys():
                try:
                    _unload(name, k)
                    self._debug_and_put(task_id, True, f"Unloaded all sessions for {name}")
                except Exception as e:
                    self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())
        else:
            try:
                _unload(name, time_str)
                self._debug_and_put(task_id, True, f"Unloaded session {name}/{time_str}")
            except Exception as e:
                self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())

    @session_method
    def _purge_session_helper(self, task_id, name, time_str, data=None):
        self._logd(f"Purging {name}/{time_str}")
        try:
            sub_task_id = self._create_id()
            self._unload_session_helper(sub_task_id, name, time_str)
            result = self._check_result(sub_task_id)  # cannot be None
            if result[1]:
                shutil.rmtree(os.path.join(self.data_dir, name, time_str))
                dirlist = os.listdir(os.path.join(self.data_dir, name))
                # Remove the full directory if only modules remain
                if dirlist == ["modules"]:
                    shutil.rmtree(self.data_dir, name)
                self._sessions[name]["sessions"].pop(time_str)
                self._debug_and_put(task_id, True, f"Purged session {name}/{time_str}")
            else:
                self._error_and_put(*result)
        except Exception as e:
            self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())

    @session_method
    def _archive_session_helper(self, task_id, name, time_str, data=None):
        # How do I mark a session as archived?
        if "saves" in data:
            # keep those files in save
            pass
        if "keep_checkpoint" in data:
            # keep last checkpoint
            pass
        if "notes" in data:
            # insert notes in state
            pass
        self._info_and_put(task_id, True, "Archive session helper")

    @session_method
    def _clone_to_helper(self, task_id, name, time_str, data=None):
        """Clones the session to a given server"""
        self._logd(f"Trying to clone session {name}/{time_str} with data {data}")
        # NOTE: no need to read file, name is enough

        def _valid_server(server):
            splits = server.split(":")
            a = len(splits) == 2
            try:
                int(splits[1])
                b = True
            except ValueError:
                b = False
            return a and b
        try:
            if "server" not in data:
                self._error_and_put(task_id, False, "server not in data")
            elif "server" in data and not _valid_server(data["server"]):
                self._error_and_put(task_id, False, f"Given server: {data['server']}" +
                                    "not a combination of host and port")
            else:
                print(self._logd(f"DATA: {data}"))
                server = data["server"]
                config_file = self._get_config_file(name, time_str)
                overrides = []
                _files = {"config_file": open(config_file, "rb")}
                _data = {"name": json.dumps(name)}
                if "config" in data and data["config"]:
                    for k, v in data["config"].items():
                        overrides.append(k.split(":") + [v])
                    self._logd(f"Config overrides given {overrides}")
                    _data["overrides"] = json.dumps(overrides)
                else:
                    self._logd(f"No config overrides")
                warn_str_list = []
                if "saves" in data:
                    savedir = os.path.join(self.data_dir, name, time_str, "savedir")
                    savefiles = os.listdir(savedir)
                    self._logd(f"Savefiles: {savefiles}")
                    _data["saves"] = []
                    if isinstance(data["saves"], bool) and data["saves"]:
                        for f in savefiles:
                            _files[f] = open(os.path.join(savedir, f), "rb")
                            _data["saves"].append(f)
                    else:
                        for f in data["saves"]:
                            if f in savefiles:
                                _files[f] = open(os.path.join(savedir, f), "rb")
                                _data["saves"].append(f)
                            else:
                                warn_str_list.append(self._logd(f"File {f} not in saves"))
                if not _data["saves"]:
                    self._logd(f"NULL Save?")
                    _data.pop("saves")
                _data["saves"] = json.dumps(_data["saves"])
                if "modules" in data:
                    self._loge("Cannot copy modules right now")
                cookies = requests.request("POST", f"http://{server}/login",
                                           data={"username": "admin",
                                                 "password": "AdminAdmin_33"}).cookies
                print(self._logd(f"SENDING DATA: {_data}"))
                response = requests.request("POST", f"http://{server}/upload_session",
                                            files=_files, data=_data, cookies=cookies)
                if "task_id" in str(response.content):
                    self._debug_and_put(task_id, True, "Sent clone request successfully" +
                                        f"Response: {response.content}" +
                                        "\n".join(warn_str_list))
                else:
                    self._error_and_put(task_id, False, f"Bad cloning response from {server}" +
                                        f"Response: {response.content}")
        except Exception as e:
            self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())

    @session_method
    def _clone_session_helper(self, task_id, name, time_str, data=None):
        """Clones the session with optional given config differences.
        """
        self._logd(f"Trying to clone session {name}/{time_str} with data {data}")
        config_file = self._get_config_file(name, time_str)
        with open(config_file, "rb") as f:
            config = f.read()
        overrides = []
        if "config" in data and data["config"]:
            for k, v in data["config"].items():
                overrides.append(k.split(":") + [v])
            self._logd(f"Config overrides given {overrides}")
        else:
            self._logd(f"No config overrides")
        time_str = datetime.datetime.now().isoformat()
        data_dir = os.path.join(self._sessions[name]["path"], time_str)
        os.mkdir(data_dir)
        self._sessions[name]["sessions"][time_str] = {}
        if self._wait_for_task(self._create_trainer, task_id,
                               args=[name, time_str, data_dir,
                                     config, overrides]):
            with open(os.path.join(self._sessions[name]["path"], time_str,
                                   "session_state")) as f:
                self._logd(f"Loading sessions state")
                self._sessions[name]["sessions"][time_str]["state"] = json.load(f)
            if "saves" in data:
                self._loge("Cannot copy saves right now")
            self._logd("Cloned session successfully")
        else:
            self._logd(f"Failed to clone with task_id {task_id}")

    @session_method
    def _reinit_session_helper(self, task_id, name, time_str, data=None):
        config_file = self._get_config_file(name, time_str)
        with open(config_file, "rb") as f:
            config = f.read()
        overrides = []
        if "config" in data and data["config"]:
            for k, v in data["config"].items():
                overrides.append(k.split(":") + [v])
            self._logd(f"Config overrides given {overrides}")
        else:
            self._logd(f"No config overrides. Will only reinitialize.")
        data_dir = os.path.join(self.data_dir, name, time_str)
        self._logd(f"Removing logs and saves")
        shutil.rmtree(os.path.join(data_dir, "logs"))
        shutil.rmtree(os.path.join(data_dir, "savedir"))
        if self._wait_for_task(self._create_trainer, task_id,
                               args=[name, time_str, data_dir,
                                     config, overrides]):
            with open(os.path.join(self._sessions[name]["path"], time_str,
                                   "session_state")) as f:
                self._logd(f"Loading sessions state")
                self._sessions[name]["sessions"][time_str]["state"] = json.load(f)
            self._logd("Reinitialized session successfully")
        else:
            self._logd(f"Failed to reinitialize with task_id {task_id}")

    def _session_method_check(self, task_id, func_name, data):
        """Calls the appropriate helper based on the url route"""
        self._logd(f"Trying to {func_name.replace('_', ' ')}: {data['session_key']}")
        if "session_key" in data:
            try:
                session_name, timestamp = data["session_key"].split("/")
            except Exception as e:
                self._error_and_put(task_id, False, f"Invalid session key {e}" +
                                    traceback.format_exc())
                return
            valid, error_str = self._check_session_valid(session_name, timestamp)
            if not valid:
                error_str = f"Invalid session {data['session_key']}\n{error_str}"
                self._error_and_put(task_id, valid, error_str)
            else:
                try:
                    self._logd(f"Ok to {func_name.split('_')[0]} session {data['session_key']}")
                    if not func_name.startswith("_"):
                        func = getattr(self, "_" + func_name + "_helper")
                    else:
                        func = getattr(self, func_name + "_helper")
                    func(task_id, session_name, timestamp, data=data)
                except Exception as e:
                    self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())
        else:
            self._error_and_put(task_id, False, "Incorrect data format given")

    def _session_check_get(self, func_name):
        func = getattr(self, func_name)
        return _dump(func())

    def _session_check_post(self, func_name):
        if not flask_login.current_user.is_authenticated:
            return self.login_manager.unauthorized()
        if isinstance(request.json, dict):
            data = request.json
        else:
            data = json.loads(request.json)
        if "session_key" not in data:
            return _dump([False, f"Invalid data {data}"])
        else:
            task_id = self._get_task_id_launch_func(self._session_method_check, func_name, data)
            return _dump([True, {"task_id": task_id,
                                 "message": f"{func_name.split('_')[0]}ing {data}"}])

    def stop(self):
        # NOTE: clear the fwd_ports_event
        if self.fwd_ports_event is not None:
            self.fwd_ports_event.clear()
            self._logi("Waiting for the fwd_ports_thread to join. Can take upto 60 seconds")
        if self.fwd_ports_thread is not None:
            self.fwd_ports_thread.join()
            self._logi("Joined fwd_ports thread")
        # NOTE: kill the fwd ports
        if self.fwd_procs is not None:
            self._logi("Killing fwd_procs")
            for p in self.fwd_procs.values():
                p.kill()
        # NOTE: Unload the sessions
        self._logi("Unloading sessions")
        for k in self._sessions:
            self._unload_session_helper(-1, k)
        # NOTE: Kill the have_internet process if it exists
        self._logi("Killing have_internet process")
        if self._have_internet is not None:
            self._have_internet.kill()
        self._logi("Killed have_internet process")

    # NOTE: Main entry point for the server
    def start(self):
        # FIXME: Cache isn't used as of now
        self._cache = {}
        self._logi(f"Initializing Server on {self.hostname}:{self.port}")
        self.scan_sessions()
        self.load_unfinished_sessions()

        @self.login_manager.user_loader
        def load_user(userid):
            if not isinstance(userid, int):
                userid = int(userid)
            return self._users[self._ids[userid]]

        @self.app.route("/", methods=["GET"])
        def __index():
            return render_template("index.html")

        @self.app.route("/<filename>", methods=["GET"])
        def __files(filename=None):
            def read_file(mode):
                with open(os.path.join(self._template_dir, filename), mode) as f:
                    content = f.read()
                return content
            filename = escape(filename)
            # FIXME: Cache isn't used as of now
            # if filename in self._cache:
            #     content = self._cache[filename]["content"]
            #     mimetype = self._cache[filename]["mimetype"]
            #     return Response(content, mimetype=mimetype)
            if filename in os.listdir(self._template_dir):
                if filename.endswith("css"):
                    mode = "r"
                    mimetype = "text/css"
                elif filename.endswith("js"):
                    mode = "r"
                    mimetype = "text/css"
                elif filename.endswith("png"):
                    mode = "rb"
                    mimetype = "image/png"
                elif filename.endswith("jpg") or filename.endswith("jpeg"):
                    mode = "rb"
                    mimetype = "image/jpeg"
                else:
                    print(filename)
                    mode = "r"
                    mimetype = "text/html"
                content = read_file(mode)
                self._cache[filename] = {}
                self._cache[filename]["content"] = content
                self._cache[filename]["mimetype"] = mimetype
                return Response(content, mimetype=mimetype)
            else:
                return Response("Not found", status=404)

        # NOTE: Simplest way would be to proxy it
        #       Although a better way would be to get the function
        #       for the url rule and return value from it
        @self.app.route("/trainer/<int:port>/<endpoint>", methods=["GET", "POST"])
        @flask_login.login_required
        def __trainer(port=None, endpoint=None):
            sess_list = self._sessions_list
            if port not in [x["port"] for x in sess_list.values()]:
                return Response(_dump([False, f"Unloaded or invalid trainer {port}"]))
            session = [*filter(lambda x: x["port"] == port, sess_list.values())][0]
            if not session["loaded"]:
                return _dump([False, "Trainer is not loaded"])
            try:
                print(f"{request.json}, {request.data}, {request.form}")
                _json = _data = _files = None
                if request.json:
                    _json = request.json if isinstance(request.json, dict)\
                        else json.loads(request.json)
                if request.form:
                    _data = dict(request.form)
                if request.files:
                    _files = request.files
                response = requests.request(request.method, f"http://localhost:{port}/{endpoint}",
                                            files=_files, json=_json, data=_data)
                excluded_headers = ["content-encoding", "content-length",
                                    "transfer-encoding", "connection"]
                headers = [(name, value) for (name, value) in response.raw.headers.items()
                           if name.lower() not in excluded_headers]
                response = Response(response.content, response.status_code, headers)
                return response
            except Exception as e:
                return Response(_dump([False, f"Error occured {e}"]))

        @self.app.route("/trainer/<int:port>/<category>/<endpoint>", methods=["GET", "POST"])
        @flask_login.login_required
        def __trainer_one(port=None, category=None, endpoint=None):
            sess_list = self._sessions_list
            if port not in [x["port"] for x in sess_list.values()]:
                return Response(_dump([False, f"Unloaded or invalid trainer {port}"]))
            session = [*filter(lambda x: x["port"] == port, sess_list.values())][0]
            if not session["loaded"]:
                return _dump([False, "Trainer is not loaded"])
            try:
                _json = _data = _files = None
                if request.json:
                    _json = request.json if isinstance(request.json, dict)\
                        else json.loads(request.json)
                if request.form:
                    _data = dict(request.form)
                if request.files:
                    _files = request.files
                response = requests.request(request.method,
                                            f"http://localhost:{port}/{category}/{endpoint}",
                                            files=_files, json=_json, data=_data)
                excluded_headers = ["content-encoding", "content-length",
                                    "transfer-encoding", "connection"]
                headers = [(name, value) for (name, value) in response.raw.headers.items()
                           if name.lower() not in excluded_headers]
                response = Response(response.content, response.status_code, headers)
                return response
            except Exception as e:
                return Response(_dump([False, f"Error occured {e}"]))


        @self.app.route("/sessions", methods=["GET"])
        @flask_login.login_required
        def __list_sessions():
            """Returns a dictionary of sessions, their ports if they're alive and the
            state. Rest of the communication can be done with session

            With optional argument {name}, if the session name starts with
            {name} then all those sessions are returned.

            """
            try:
                name = request.args.get("name")
            except Exception:
                name = None
            sess_list = self._sessions_list
            if name:
                name = name.strip()
            if name:
                sessions = {k: v for k, v in sess_list.items()
                            if k.startswith(name)}
                if sessions:
                    return _dump([True, sessions])
                else:
                    return _dump([False, "No session found"])
            else:
                return _dump([True, self._sessions_list])

        @self.app.route("/current_user", methods=["GET"])
        @flask_login.login_required
        def __current_user():
            """Returns the name of the current user, in case we're logged in and username is
            not known to the client as they have refreshed and the store state
            is gone (LOL, FIXME)

            """
            return _dump([True, {"user": flask_login.current_user.name}])

        # CHECK: Why's this function so complicated?
        @self.app.route("/update_given_name", methods=["POST"])
        @flask_login.login_required
        def __update_given_name():
            # print("JSON?", request.json)
            # if hasattr(request, "data"):
            #     print("DATA", request.data)
            # print("FORM", [*request.form.keys()])
            if isinstance(request.json, dict):
                data = request.json
            else:
                data = json.loads(request.json)
            if "given_name" not in data:
                return _dump([False, "Name not in data"])
            if "trainer_url" not in data:
                return _dump([False, "Missing params"])
            if "session_key" not in data:
                return _dump([False, "Missing params"])
            else:
                url = data.pop("trainer_url")
                key = data.pop("session_key")
                # hack_param first, then force dump_state
                hp_data = {"given_name": {"type": "str", "value": data["given_name"]}}
                response = requests.request("POST", url + "_helpers/hack_param",
                                            headers={'Content-Type': 'application/json'},
                                            data=json.dumps(hp_data))
                if response.status_code == 200:
                    response = requests.request("POST", url + "_internals/_dump_state",
                                                headers={'Content-Type': 'application/json'},
                                                data=json.dumps({"secret": "_sxde#@_"}))
                    resp_str = json.loads(response.content)[0]
                    if resp_str[0]:
                        sess, ts = key.split("/")
                        # NOTE: Update state
                        with open(os.path.join(self._sessions[sess]["path"], ts, "session_state"),
                                  "r") as f:
                            self._sessions[sess]["sessions"][ts]["state"] = json.load(f)
                        return _dump([True, "Updated successfully"])
                    else:
                        return _dump([True, f"{resp_str[1]}"])
                else:
                    return _dump([False, f"Could not assign name"])

        # FIXME: no provision to load config as JSON? Everything can't be bytes
        #        can it?
        @self.app.route("/create_session", methods=["POST"])
        @flask_login.login_required
        def __new_session():
            if "name" not in request.form or ("name" in request.form
                                              and not len(request.form["name"])):
                return _dump([False, "Name not in request or empty name"])
            else:
                try:
                    data = json.loads(request.form["name"])
                    file_bytes = request.files["file"].read()
                except Exception as e:
                    return _dump([False, f"{e}" + "\n" + traceback.format_exc()])
            data = {"name": data, "config": file_bytes}
            task_id = self._get_task_id_launch_func(self.create_session, data)
            return _dump([True, {"task_id": task_id,
                                 "message": "Creating session with whatever data given"}])

        @self.app.route("/upload_session", methods=["POST"])
        @flask_login.login_required
        def __upload_session():
            form = request.form
            if form is None:
                return _dump([False, "Bad data in request"])
            elif "name" not in form or ("name" in form
                                        and not len(form["name"])):
                return _dump([False, "Name not in request or empty name"])
            if "config_file" not in request.files:
                return _dump([False, "Config file not in request"])
            try:
                data = {}
                for k, v in form.items():
                    data[k] = json.loads(v)
                name = data["name"]
                config_file = request.files["config_file"].read()
                saves = {}
                if "saves" in data:
                    print("FILES: ", request.files.keys())
                    for f in data["saves"]:
                        saves[f] = request.files[f].read()
                overrides = []
                if "overrides" in data:
                    overrides = data["overrides"]
            except Exception as e:
                return _dump([False, f"{e}" + "\n" + traceback.format_exc()])
            data = {"name": name, "config": config_file, "saves": saves,
                    "overrides": overrides}
            task_id = self._get_task_id_launch_func(self.create_session, data)
            return _dump([True, {"task_id": task_id,
                                 "message": "Creating session with whatever data given"}])

        @self.app.route("/docs", methods=["GET"])
        @flask_login.login_required
        def __docs():
            """Returns all the docs for all the endpoints in the server. The trainer docs
            can be fetched from the trainer itself.

            """
            endpoints = {x.rule: x.endpoint for x in self.app.url_map.iter_rules()
                         if "<" not in x.rule}
            docs = {e.replace("/", "", 1): self.app.view_functions[endpoints[e]].__doc__
                    for e in endpoints}
            preamble = """Some docs are missing for some endpoints and will be added soon.  Example
            calls and return values can also be added. In that sense the
            functions should become self documenting soon.

            :methods: GET
            :args: []
            :retval: json(list[status[bool], dict[{multi[docs]}]])
            """
            return _dump([preamble, docs])

        @self.app.route("/check_task", methods=["GET"])
        @flask_login.login_required
        def __check_task():
            """Check and return the status of a task submitted earlier.

            :methods: GET
            :args: [task_id]
            :retval: list[bool, dict[task_id, result, message]]
            """
            try:
                task_id = int(request.args.get("task_id").strip())
            except Exception as e:
                return _dump([False, f"Bad params {e}" + "\n" + traceback.format_exc()])
            if task_id not in self.__task_ids:
                return _dump([False, f"No such task: {task_id}"])
            else:
                result = self._check_result(task_id)
            if result is None:
                return _dump([True, {"task_id": task_id, "result": 0,
                                     "message": "Not yet processed"}])
            else:
                if len(result) == 2:
                    self._logw(f"Result of length 2 for check_task {result}")
                    return _dump([True, {"task_id": result[0], "result": True,
                                         "message": "Successful"}])
                elif len(result) == 3 and result[1]:
                    return _dump([True, {"task_id": result[0], "result": True,
                                         "message": result[2]}])
                elif len(result) == 3 and not result[1]:
                    return _dump([True, {"task_id": result[0], "result": False,
                                         "message": result[2]}])
                else:
                    return _dump([True, result])

        @self.app.route("/_version", methods=["GET"])
        def __version():
            """Return version of the current server.

            :methods: GET
            :args: []
            :retval: str
            """
            return self.__version__

        # FIXME: We're not checking overlap
        @self.app.route("/_devices", methods=["GET", "POST"])
        def __devices():
            if request.method == "POST":
                try:
                    data = request.json
                    if "action" == "reserve":
                        reserved = self.reserved_devices
                        available = [x for x in data["gpus"] if x not in reserved]
                        if data["port"] not in self._devices:
                            self._devices[data["port"]] = []
                        self._devices[data["port"]].extend(available)
                        return _dump([True, self._devices[data["port"]]])
                    elif "action" == "free":
                        self._devices[data["port"]] = list(set(self._devices[data["port"]]) - set(data["gpus"]))
                        return _dump([True, self._devices[data["port"]]])
                except Exception as e:
                    return _dump([False, f"{e}"])
            elif request.method == "GET":
                return _dump(self.reserved_devices)

        # @flask_login.login_required
        # @self.app.route("/api", methods=["GET"])
        # def __api():
        #     bleh = self
        #     import ipdb; ipdb.set_trace()
        #     return _dump("")

        # I think I have to update the user.id on each login
        @self.app.route("/login", methods=["POST"])
        def __login():
            """Return version of the current server.

            :methods: POST
            :args: dict[]
            :retval: str
            """
            if "username" not in request.form or "password" not in request.form:
                return _dump([False, "Username or Password not provided"])
            else:
                status, user = self._check_username_password(request.form)
                if status:
                    flask_login.login_user(user, remember=False)
                    return _dump([True, "Login Successful"])
                else:
                    return _dump([False, "Invalid Credentials"])

        # I think I have to update the user.id on each login
        @self.app.route("/logged_in", methods=["GET"])
        def __logged_in():
            """Check if the user is logged in"""
            if flask_login.current_user.is_authenticated:
                return _dump([True, "Logged in"])
            else:
                return _dump([False, "Could not Login"])

        @self.app.route("/logout", methods=["GET"])
        @flask_login.login_required
        def __logout():
            """Logout from the Server."""
            flask_login.logout_user()
            return _dump([True, "Logged Out"])

        @self.app.route("/list_session_modules", methods=["GET"])
        @flask_login.login_required
        def __list_session_modules():
            """Returns the list of module available with a given session.

            Requres param {session_key}

            """
            return _dump([False, "Not Implemented yet"])

        @self.app.route("/add_session_module", methods=["POST"])
        @flask_login.login_required
        def __add_session_module():
            """NOT IMPLEMENTED: This feature isn't implemented yet and should not be called.

            Load a module into a given session's load path. Parameters and
            methods are similar to `add_global_module`. Additional param
            {session_key} has to be provided.

            """
            return _dump([False, "Not implemented yet"])
            import ipdb; ipdb.set_trace()
            if "name" not in request.form or ("name" in request.form
                                              and not len(request.form["name"])):
                return _dump([False, "Name not in request or empty name"])
            else:
                try:
                    data = json.loads(request.form["name"])
                    file_bytes = request.files["file"].read()
                except Exception as e:
                    return _dump([False, f"{e}" + "\n" + traceback.format_exc()])
            data = {"name": data, "data_file": file_bytes}
            task_id = self._get_task_id_launch_func(self._load_module, data)
            return _dump([True, {"task_id": task_id,
                                 "message": "Adding global data"}])

        @self.app.route("/delete_session_module", methods=["POST"])
        @flask_login.login_required
        def __delete_session_module():
            """NOT IMPLEMENTED: This feature isn't implemented yet and should not be called.

            Delete the module in the given session. Expects args {module_name, session_key}

            NOTE: Make sure that the imports are reloaded if there's an update

            """
            return _dump([False, "Not implemented yet"])
            import ipdb; ipdb.set_trace()
            if "name" not in request.form or ("name" in request.form
                                              and not len(request.form["name"])):
                return _dump([False, "Name not in request or empty name"])
            else:
                try:
                    data = json.loads(request.form["name"])
                    file_bytes = request.files["file"].read()
                except Exception as e:
                    return _dump([False, f"{e}" + "\n" + traceback.format_exc()])
            data = {"name": data, "data_file": file_bytes}
            task_id = self._get_task_id_launch_func(self._load_module, data)
            return _dump([False, {"task_id": task_id,
                                  "message": "Adding global data"}])

        @self.app.route("/list_global_modules", methods=["GET"])
        @flask_login.login_required
        def __list_global_modules():
            """Returns the list of global modules available.
            """
            return _dump([True, self._modules])

        @self.app.route("/add_global_module", methods=["POST"])
        @flask_login.login_required
        def __add_global_module():
            """Can be python or zip file. Shows up in global modules and is immediately
            available for loading to all sessions. Will overwrite if a module
            with the same name already exists.

            NOTE: Make sure that the imports are reloaded if there's an update
            """
            if "name" not in request.form or ("name" in request.form
                                              and not len(request.form["name"])):
                return _dump([False, "Name not in request or empty name"])
            else:
                try:
                    data = json.loads(request.form["name"])
                    file_bytes = request.files["file"].read()
                except Exception as e:
                    return _dump([False, f"{e}" + "\n" + traceback.format_exc()])
            data = {"name": data, "data_file": file_bytes}
            task_id = self._get_task_id_launch_func(self._load_module, data)
            return _dump([True, {"task_id": task_id,
                                 "message": "Adding global module"}])

        @self.app.route("/delete_global_module", methods=["POST"])
        @flask_login.login_required
        def __delete_global_module():
            """Deletes the given module from the list of global modules. The module is
            immediately unavailable for all future running functions.

            NOTE: Make sure that the imports are reloaded if there's an update
            NOTE: What if a function relied on some deleted module? It should
                  no longer work. Not sure how to handle that.
            NOTE: Module names start with _module_ internally and the module name
                  itself shouldn't start with _
            """
            if "name" not in request.form or ("name" in request.form
                                              and not len(request.form["name"])):
                return _dump([False, "Name not in request or empty name"])
            else:
                mod_name = self._get_module_name(request.form["name"])
                if mod_name not in self._modules:
                    return _dump([False, f"No such module {request.form['name']}"])
            try:
                self._modules.pop(mod_name)
                mods_dir = self.modules_dir
                if os.path.exists(os.path.join(mods_dir, mod_name)):
                    if os.path.islink(os.path.join(mods_dir, mod_name)):
                        os.unlink(os.path.join(mods_dir, mod_name))
                    else:
                        shutil.rmtree(os.path.join(mods_dir, mod_name))
                    return _dump([True, f"Removed {mod_name}."])
                elif os.path.exists(os.path.join(mods_dir, mod_name + ".py")):
                    if os.path.islink(os.path.join(mods_dir, mod_name)):
                        os.unlink(os.path.join(mods_dir, mod_name + ".py"))
                    else:
                        os.remove(os.path.join(mods_dir, mod_name + ".py"))
                    return _dump([True, f"Removed {mod_name}."])
                else:
                    return _dump([False, f"Module {mod_name} was not on disk"])
            except Exception as e:
                return _dump([False, f"{e}" + "\n" + traceback.format_exc()])

        @self.app.route("/list_datasets", methods=["GET"])
        @flask_login.login_required
        def __list_datasets():
            """Returns the list of global modules available.
            """
            return _dump([True, self._datasets])

        @self.app.route("/upload_dataset", methods=["POST"])
        @flask_login.login_required
        def __upload_dataset():
            """Must be zip file. An __init__.py should be at the top of the zip file and
            should access the data with relative paths or through the
            network. No assumptions about absolute paths should be made.

            The dataset name should be given in form along with description and
            the dataset must implement __len__ and __getitem__.

            Type of data should also be mentioned.

            """
            if "name" not in request.form or ("name" in request.form
                                              and not len(request.form["name"])):
                return _dump([False, "Name not in request or empty name"])
            if "description" not in request.form or ("description" in request.form
                                                     and not len(request.form["description"])):
                return _dump([False, "Description not in request or empty description"])
            if "type" not in request.form or ("type" in request.form
                                              and not len(request.form["type"])):
                return _dump([False, "Dataset type not in request or empty"])
            try:
                data = {}
                for x in ["name", "description", "type"]:
                    data[x] = request.form[x]
                file_bytes = request.files["file"].read()
            except Exception as e:
                return _dump([False, f"{e}" + "\n" + traceback.format_exc()])
            data = {"data_file": file_bytes, **data}
            task_id = self._get_task_id_launch_func(self._load_dataset, data)
            return _dump([True, {"task_id": task_id,
                                 "message": "Adding global data"}])

        @self.app.route("/delete_dataset", methods=["POST"])
        @flask_login.login_required
        def __delete_dataset():
            if "name" not in request.form or ("name" in request.form
                                              and not len(request.form["name"])):
                return _dump([False, "Name not in request or empty name"])
            else:
                name = self._get_dataset_name(request.form["name"])
                if name not in self._datasets:
                    return _dump([False, f"No such dataset {request.form['name']}"])
            try:
                self._datasets.pop(name)
                data_dir = self.datasets_dir
                if os.path.exists(os.path.join(data_dir, name)):
                    shutil.rmtree(os.path.join(data_dir, name))
                    os.remove(os.path.join(data_dir, name) + ".json")
                    return _dump([True, f"Removed {name}."])
                elif os.path.exists(os.path.join(data_dir, name + ".py")):
                    os.remove(os.path.join(data_dir, name + ".py"))
                    os.remove(os.path.join(data_dir, name) + ".json")
                    return _dump([True, f"Removed {name}."])
                else:
                    return _dump([False, f"Dataset {name} was not on disk"])
            except Exception as e:
                return _dump([False, f"{e}" + "\n" + traceback.format_exc()])

        # @self.app.route("/fwd_ports", methods=["GET", "POST"])
        # @flask_login.login_required
        # def __fwd_ports():
        #     data = []
        #     if request.method == "POST":
        #         if request.json and isinstance(request.json, dict):
        #             data = request.json
        #         else:
        #             data = json.loads(request.json)
        #     self._fwd_ports(data)
        #     return _dump("Forwarded Ports again")

        # @self.app.route("/register_with_trackers", methods=["POST"])
        # @flask_login.login_required
        # def __register_with_trackers():
        #     data = []
        #     if request.json and isinstance(request.json, dict):
        #         data = request.json
        #     else:
        #         data = json.loads(request.json)
        #     self._register_with_trackers(data)
        #     return _dump("Registered with Trackers")

        # @self.app.before_request
        # def __before_request_func():
        #     self._logd(f"FWD ports thread is alive? {self.fwd_ports_thread.is_alive()}")

        @self.app.route("/_ping", methods=["GET"])
        def __ping():
            return "pong"

        @self.app.route("/_name", methods=["GET"])
        def __name():
            # only views the progress and parameters. No trainer is started
            print("DAEMON NAME", self.daemon_name)
            if self.daemon_name is not None:
                return self.daemon_name
            else:
                try:
                    with open("daemon_name", "r") as f:
                        daemon_name = f.read().split("\n")[0].strip()
                except Exception:
                    return "No Name"
                return daemon_name

        # NOTE: This should be disabled for now. Only if the number of sessions
        #       gets too large should I use this, as otherwise all the session
        #       data should be sent to the client.
        # @self.app.route("/view_session", methods=["POST"])
        # def __view_session():
        #     # only views the progress and parameters. No trainer is started
        #     return _dump([False, "Doesn't do anything"])

        # NOTE: Add session_methods.
        #       Routes are added by removing "_" prefix and "_helper" suffix
        #       from self._session_methods
        # FIXME: Check if login is required for these routes. I'm not sure
        for x in self._session_methods:
            self.app.add_url_rule("/" + x, x, partial(self._session_check_post, x),
                                  methods=["POST"])

        @atexit.register
        def cleanup():
            self.stop()

        @self.app.route("/_shutdown", methods=["GET"])
        @flask_login.login_required
        def __shutdown_server():
            self._logd("Shutdown called via HTTP. Shutting down.")
            Thread(target=cleanup).start()
            func = request.environ.get('werkzeug.server.shutdown')
            func()
            return "Shutting down"

        serving.run_simple(self.hostname, self.port, self.app, threaded=True,
                           ssl_context=self.context)


def create_daemon(test=False, params=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=23232,
                        help="The port on which to serve")
    parser.add_argument("--hostname", type=str, default="127.0.0.1",
                        help="The hostname on which to serve")
    parser.add_argument("--config-file", "-c", type=str,
                        default=os.path.expanduser("~/.training_server_daemon"),
                        help="The hostname on which to serve")
    parser.add_argument("--data-dir", "-d", type=str,
                        default=os.path.expanduser("~/.training_server_data"),
                        help="The directory which will contain all the sessions and model" +
                        " etc. files and directories")
    args = parser.parse_args()
    if not test:
        config = configparser.ConfigParser()
        if os.path.exists(args.config_file):
            config.read(args.config_file)
            for k in args.__dict__:
                if k in config["default"].keys():
                    args.__dict__[k] = config["default"][k]
        else:
            config.add_section("default")
            for k in args.__dict__:
                config["default"][k] = str(args.__dict__[k])
            with open(args.config_file, "w") as f:
                config.write(f)
        args.port = int(args.port)
    else:
        for p in params:
            args.__dict__[p] = params[p]
    daemon = Daemon(args.hostname, args.port, args.data_dir)
    return daemon


def _start_daemon(hostname, port, data_dir, production=False,
                  template_dir=None, static_dir=None, root_dir=None,
                  daemon_name=None, trackers=None, register=False):
    daemon = Daemon(hostname, port, data_dir, production, template_dir,
                    static_dir, root_dir, trackers, daemon_name, register)
    Thread(target=daemon.start).start()
    return daemon


if __name__ == "__main__":
    daemon = create_daemon()
    daemon.start()
