from typing import List, Dict, Any, Union, Callable, Optional, Tuple
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
import datetime
import logging
import hashlib
import traceback
import zipfile
import pathlib
from queue import Queue
from threading import Thread
import multiprocessing as mp
from subprocess import Popen, PIPE, TimeoutExpired
from markupsafe import escape
from functools import partial

import flask_login
from flask import Flask, render_template, request, Response, make_response
from flask_cors import CORS
from werkzeug import serving

from ..version import __version__
from ..mods import Modules
from ..helpers import Tag
from ..interfaces import FlaskInterface
from ..util import _dump, diff_as_sets, make_json
from .._log import Log

from .auth import __unti__, __inti__, User
from .util import load_json, create_module

from . import views
from . import models
from .sessions import Sessions
from .trainer_views import Trainer
from .check_task import CheckTask

session_method = Tag("session_method")

Path = Union[str, pathlib.Path]


class Daemon:
    __version__ = __version__

    def __init__(self, hostname: str, port: int, root_dir: Path,
                 daemon_name: str):
        self.ctx = mp.get_context("spawn")
        self._hostname = hostname
        self._port = port
        # self._trackers = trackers
        self.daemon_name = daemon_name
        # self.register = register
        # FIXME: have_internet shouldn't be here
        # NOTE: fwd_hosts and fwd_ports are hard coded
        # self.fwd_port_start = 8181    # starts with 8181
        # if "droid" not in get_hostname().lower():
        #     self._have_internet = mp.Process(target=have_internet)
        #     self._have_internet.start()
        # else:
        #     self._have_internet = None
        self._init_root_dir(root_dir)
        self._init_app()
        self._init_resources()
        self._init_logger()
        self._init_modules()
        self._init_auth()
        self._init_flags()
        self._logi("Initialized Daemon")
        # self._maybe_fwd_ports()

    def _init_root_dir(self, root_dir):
        """Initialize the directory structure.

        :attr:`data_dir` is the root directory where all the modules and
        training sessions are stored.

        """
        self._lib_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self._root_dir = os.path.abspath(root_dir)
        # NOTE: init data_dir
        if not os.path.exists(self._root_dir):
            os.mkdir(self._root_dir)
        self.tmp_dir = os.path.join(self._root_dir, ".tmp")
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)
        # FIXME: Code duplication here
        # NOTE: Append data_dir path
        self.env_str = f"""
import sys
if "{self._root_dir}" not in sys.path:
    sys.path.append("{self._root_dir}")
"""
        self.root_env_str = f"""
import sys
if "{self.root_dir}" not in sys.path:
    sys.path.append("{self.root_dir}")
"""
        # NOTE: init modules_dir
        self.modules_dir = os.path.join(self._root_dir, "global_modules")
        create_module(self.modules_dir,
                      [os.path.join(self._lib_dir, x)
                       for x in ["autoloads.py"]],
                      env_str=self.root_env_str)
        # NOTE: init datasets_dir
        self.datasets_dir = os.path.join(self._root_dir, "global_datasets")
        create_module(self.datasets_dir)
        # NOTE: Set exclude_dirs, dirs not to scan for sessions
        self._exclude_dirs = [*map(os.path.basename,
                                   [self.modules_dir, self.datasets_dir,
                                    self.tmp_dir])]
        self._session_exclude_dirs = ["modules", "datasets"]
        if not os.path.exists(self._root_dir):
            raise FileExistsError(f"FATAL ERROR! root dir {self._root_dir} doesn't exist")
        self.if_run_file = os.path.join(self._lib_dir, "if_run.py")
        if not os.path.exists(self.if_run_file):
            raise FileNotFoundError(f"FATAL ERROR! {self.if_run_file} doesn't exist")

    def _init_app(self):
        "Initialize the :class:`Flask` app"
        self.app = Flask(__name__)
        # NOTE: FIXME Fix for CSRF etc.
        #       see https://flask-cors.corydolphin.com/en/latest/api.html#using-cors-with-cookies
        CORS(self.app, supports_credentials=True)
        # self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        # NOTE: Not sure if this is really useful
        self.app.secret_key = __unti__("_sxde#@_")
        self.use_https = False
        self.verify_user = True
        self._last_free_port = self.port

    def _init_resources(self):
        self._threads = {}
        self._task_q = Queue()
        self._sessions = {}
        self._devices = {}
        self._modules = {}
        self._datasets: Dict[str, Dict] = {}
        self._init_context()
        self._task_id = 0
        self.__task_ids = []
        self._results = []

    def _init_logger(self):
        self._logger = logging.getLogger("daemon_logger")
        log_file = os.path.join(self._root_dir, "logs")
        formatter = logging.Formatter(datefmt='%Y/%m/%d %I:%M:%S %p', fmt='%(asctime)s %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.DEBUG)
        log = Log(self._logger)
        self._logd = log._logd
        self._loge = log._loge
        self._logi = log._logi
        self._logw = log._logw

    def _init_modules(self):
        "Initialize Module Loader, modules and datasets"
        self._module_loader = Modules(self._root_dir, self._logd, self._loge,
                                      self._logi, self._logw)
        self._load_available_global_modules()
        self._load_available_global_datasets()

    def _init_auth(self):
        "Initialize login manager and users"
        self.login_manager = flask_login.LoginManager()
        self.login_manager.init_app(self.app)
        try:
            self._ids = __ids__
            self._users = __users__
        except NameError:
            self._ids = {0: "admin", 1: "joe"}
            self._users = {"admin": User(0, "admin"),
                           "joe": User(1, "joe")}
        self._passwords = lambda x: __inti__(x)     # {"admin": "admin", "joe": "admin"}

    def _init_flags(self):
        self._already_scanned = False
        self._testing = False

    @property
    def root_dir(self) -> Path:
        """Directory for execution context of :class:`~subprocess.Popen` processes.

        Used only by :meth:`_create_trainer`"""
        return self._root_dir

    @property
    def datasets(self) -> Dict[str, Dict]:
        return self._datasets

    @property
    def hostname(self) -> str:
        "Hostname on which to serve"
        return self._hostname

    @property
    def port(self) -> int:
        "port on which to listen"
        return self._port

    @property
    def reserved_devices(self) -> List[int]:
        """A :class:`dict` mapping trainers to devices allocated to them.

        Used by trainers to manage GPUs among themselves.

        """
        devices = []
        for x in self._devices.values():
            devices.extend(x)
        return devices

    def _init_context(self):
        """Initialize the flask SSL context."""
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

    def _check_config(self, data_dir: Path, config: Dict[str, Any],
                      overrides: Dict[str, Any] = {}) -> Tuple[bool, str]:
        if os.path.exists(os.path.join(data_dir, "config.json")):
            status, result = True, "Config exists"
        else:
            try:
                self._logd(f"Checking config")
                iface = FlaskInterface(None, None, data_dir,
                                       config_overrides=overrides,
                                       no_start=True)
                status, result = iface.create_trainer(config)
                del iface
            except Exception as e:
                status, result = False, f"{e}" + "\n" + traceback.format_exc()
        return status, result

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
        # NOTE: 0 reserverd for instance
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
        """Call function `func` and wait until it finishes.

        The function upon completion will put the result with `task_id` on to
        the :attr:`_task_q`.

        """
        func(task_id, *args)
        result = self._check_result(task_id)
        while result is None:
            time.sleep(1)
            result = self._check_result(task_id)
        return result[1]

    def _update_init_file(self, init_file: Path, module_names: List[str]):
        """Update the `init_file` with the `module_names`.

        Each module is inserted as an import statement at the top of file.

        """
        lines = []
        for m in module_names:
            lines.append(f"from . import {m}\n")
        with open(init_file, "w") as f:
            f.writelines(lines)

    def _load_available_global_modules(self):
        "Scan the :attr:`modules_dir` and load all modules"
        mods_dir = self.modules_dir
        self._modules = self._module_loader.read_modules_from_dir(
            mods_dir, excludes=[lambda x: x.startswith("__")])
        self._update_init_file(os.path.join(self.modules_dir, "__init__.py"),
                               self._modules.keys())

    def _load_available_global_datasets(self):
        "Scan the :attr:`datasets_dir` and load all."
        json_filenames = [x for x in os.listdir(self.datasets_dir)
                          if x.endswith(".json")]
        for x in json_filenames:
            with open(os.path.join(self.datasets_dir, x)) as f:
                self._datasets[x.replace(".json", "")] = json.load(f)
        self._update_init_file(os.path.join(self.datasets_dir, "__init__.py"),
                               self._modules.keys())

    def _dataset_valid_p(self, data_dict):
        """Check a given dataset based on names that it exports."""
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

    def _load_dataset(self, task_id: int, data: Dict[str, Any]):
        """Load a dataset from a given data spec.

        The spec `data` should contain keys ['name', 'description', 'type',
        'data_file'].  `type` and `description` are informational while name
        should be unique.  A duplicate name overwrites the previous dataset.

        `data_file` should a readable binary IO stream.

        """
        name = self._get_dataset_name(data["name"])
        desc = data["description"]
        dtype = data["type"]
        file_bytes = data["data_file"]
        data_dir = self.datasets_dir
        result = self._module_loader.add_named_module(data_dir, file_bytes, name)
        check_duplicate = ""
        if name in self._datasets:
            check_duplicate = f"{name} already exists in datasets. Will be overwritten\n"
        if result[0]:
            status, message = self._dataset_valid_p(result[1])
            if status:
                self._datasets[name] = {}
                self._datasets[name]["members"] = result[1][name]
                self._datasets[name]["description"] = desc
                self._datasets[name]["type"] = dtype
                with open(os.path.join(data_dir, name + ".json"), "w") as f:
                    json.dump({name: self._datasets[name]}, f)
                self._debug_and_put(task_id, True, check_duplicate + message)
            else:
                self._error_and_put(task_id, status, message)
        else:
            self._error_and_put(task_id, False, f"Could not add dataset. {result}")

    def _load_module(self, task_id, data):
        """Load a module from a given data spec.

        The spec `data` should contain 'name' and an optional
        `description`. `name` should be unique and a duplicate name overwrites
        the previous module.

        `data_file` should a readable binary IO stream.

        """
        mod_name = self._get_module_name(data["name"])
        mod_file = data["data_file"]
        # FIXME: Not actually implemented
        if "description" in data:
            desc = data["description"]
        check_duplicate = ""
        if mod_name in self._modules:
            check_duplicate = f"{mod_name} already exists in modules. Will be overwritten\n"
        mods_dir = self.modules_dir
        status, result = self._module_loader.add_named_module(mods_dir, mod_file, mod_name)
        if status:
            self._modules.update(result)
            self._debug_and_put(task_id, True, check_duplicate + f"Added module {mod_name}")
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

    def scan_sessions(self):
        """Scan the :attr:`data_dir` for existing :class:`Trainer` sessions.

        scan_sessions should be called only ONCE at beginning and after that
        should raise error (or atleast warn) unless testing

        """
        if self._already_scanned and not self._testing:
            self._logw("Scanning sessions again!")
        self._logd("Scanning Sessions")
        session_names = [x for x in os.listdir(self._root_dir) if
                         os.path.isdir(os.path.join(self._root_dir, x))
                         and x not in self._exclude_dirs]
        for s in session_names:
            self._sessions[s] = {}
            self._sessions[s]["path"] = os.path.join(self._root_dir, s)
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
        self._already_scanned = True

    # FIXME: data should be pydantic type
    def create_session(self, task_id: int, data: dict):
        """Creates a new training session from given data

        Args:
            task_id: id for the task
            data: session properties and config

        Schemas:
            data:
                name: str
                config: trainer.config.Config
                overrides: Optional[dict]
                load: Optional[bool]
                saves: Optional[Dict[str, bytes]]

        A session has a structure `session[key]` where `key` in `{"path",
        "sessions", "modules"}` Modules are loaded from the module path which is
        appended to `sys.path` for that particular session. Each module has a
        separate namespace as such and since each trainer instance is separate,
        it should be easy to separate. The modules are shared among all the
        subsessions.

        For example, if the sessions directory is `sessions` then a session with
        the name of `funky_session` can have a directory structure like
        `/sessions/funky_session/2020-02-17T10:53:06.458827`.

        """
        session_name = data["name"]
        if session_name not in self._sessions:
            self._sessions[session_name] = {}
            self._sessions[session_name]["path"] = os.path.join(self._root_dir, session_name)
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
        if "load" in data:
            load = data["load"]
        else:
            load = False
        # FIXME: Why does wait_for_task succeed and then session_state is dead?
        if self._wait_for_task(self._create_trainer, task_id,
                               args=[session_name, time_str, data_dir,
                                     data["config"], overrides, load]):
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
    # FIXME: config is a pydantic type already
    def _create_trainer(self, task_id: int, name: str, time_str: str,
                        data_dir: Path, config: Optional[dict],
                        overrides: Dict[str, Any] = {}, load: bool = False):
        """Create a trainer.

        Args:
            task_id: Unique task_id assigned to the task
            name: Name for the trainer
            time_str: Timestamp of creation
            data_dir: Root data directory for the session
            config: Parameters for the trainer
            overrides: Changes made by the user after initial config
            load: Whether to load into memory on start

        A separate :class:`Flask` is created with :class:`~subprocess.Popen` which wraps
        around a :class:`Trainer` instance and communicates with the
        :class:`Daemon` via a proxy.

        """
        self._logd(f"Trying to create trainer with data_dir {data_dir}")
        if not os.path.isabs(data_dir):
            data_dir = os.path.abspath(data_dir)
        if config is None:
            try:
                with open(os.path.join(data_dir, "config.json")) as f:
                    config = json.load(f)
                status = True
            except Exception as e:
                self._error_and_put(task_id, False, f"Could not read config {e}" +
                                    "\n" + traceback.format_exc())
                return
        else:
            status, result = self._check_config(data_dir, config, overrides)
        if status:
            if load:
                try:
                    self._logd(f"Config valid")
                    port = self._find_open_port()
                    self._sessions[name]["sessions"][time_str]["config"] = config
                    self._sessions[name]["sessions"][time_str]["port"] = port
                    self._sessions[name]["sessions"][time_str]["data_dir"] = data_dir
                    self._sessions[name]["sessions"][time_str]["data_dir"] = data_dir
                    if overrides:
                        with open(os.path.join(data_dir, "config_overrides.json")) as f:
                            json.dump(overrides, f)
                    cmd = f"python {self.if_run_file} {self.hostname} {port} {data_dir} " +\
                        "--config-overrides=True"
                    cwd = os.path.dirname(self._lib_dir)
                    self._logd(f"Running command {cmd} in {cwd}")
                    p = Popen(shlex.split(cmd), env=os.environ, cwd=cwd)
                    self._sessions[name]["sessions"][time_str]["process"] = p
                    time.sleep(1)
                    if p.poll() is None:
                        self._info_and_put(task_id, True, "Created Trainer")
                    else:
                        self._error_and_put(task_id, False, "Trainer crashed")
                except Exception as e:
                    self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())
            else:
                self._info_and_put(task_id, True, result)
        else:
            self._error_and_put(task_id, False, result)

    def _refresh_state(self, session_key: str):
        """Helper function to refresh session state.

        Args:
            session_key: Session Key

        """
        name, time_str = session_key.split("/")
        with open(os.path.join(self._sessions[name]["path"], time_str,
                               "session_state"), "r") as f:
            self._sessions[name]["sessions"][time_str]["state"] = json.load(f)

    def _unload_finished_session(self, session_key: str):
        """Helper function to unload a session which has finished

        Args:
            session_key: Session Key

        """
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
        max_epochs = state["max_epochs"]
        iterations = state["iterations"]
        max_iterations = state["max_iterations"]
        if (not iterations and not max_iterations and epoch < max_epochs) or\
           (not epoch and not max_epochs and iterations < max_iterations):
            return False
        else:
            return True

    def _session_alive_p(self, session_name: str, timestamp: str) -> bool:
        """Check whether a session is alive.

        Args:
            session_name: Session Name
            timestemp: Session Timestamp

        """
        # FIXME: Fix this to enum
        status = self._check_session_valid(session_name, timestamp)
        if not status[0]:
            return status[0]
        else:
            retcode = self._sessions[session_name]["sessions"][timestamp]["process"].poll()
            if retcode is None:
                return True
            else:
                self._logd(f"Session died for {session_name}/{timestamp} with code {retcode}")
                return False

    def _check_session_valid(self, session_name: str, timestamp: str) -> Tuple[bool, str]:
        if session_name not in self._sessions:
            return False, f"Unknown session, {session_name}"
        elif (session_name in self._sessions and
              timestamp not in self._sessions[session_name]["sessions"]):
            return False, f"Given session instance not in sessions"
        elif not os.path.exists(os.path.join(self._root_dir, "/".join([session_name, timestamp]))):
            return False, f"Session has been deleted"
        else:
            return True, ""

    def compare_sessions(self):
        pass

    # FIXME: merge state and config here.
    #        What if the config changes in the middle? It should update automatically.
    @property
    def sessions_list(self) -> Dict[str, models.Session]:
        retval: Dict[str, Dict[str, Union[None, bool, int, Dict]]] = {}
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

    def _check_username_password(self, data: Dict[str, str]):
        if (data["username"] in self._users):
            __hash = hashlib.sha1(("2ads;fj4sak#)" + data["username"])
                                  .encode("utf-8")).hexdigest()
            if data["password"] == self._passwords(__hash):
                return True, self._users[data["username"]]
            else:
                return False, None
        else:
            return False, None

    def _get_config_file(self, name: str, time_str: str = None):
        """Get the config file for the session key.

        Args:
            name: Session Name
            time_str: Session Timestamp

        Session Key is reconstructed from `name` and `time_str`.  config_file is
        searched as either a ``session_config.py`` file or a ``session_config``
        directory with an ``__init__.py`` file at the top level.

        """
        if time_str is not None:
            data_dir = os.path.join(self._root_dir, name, time_str)
        else:
            data_dir = os.path.join(self._root_dir, name)
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
    def _session_methods(self) -> List[str]:
        """Return list of `session_method`s.

        A `session_method` is any method used for manipulating a method. It's
        created by :meth:`_session_check_post` and :meth:`_session_check_get`
        higher order functions.

        The method eventually calls `_ + methname + _session_helper` and the
        `methname_session` is exposed as an endpoint, e.g.,
        :meth:`_load_session_helper` is exported as `load_session`

        """
        return [x[1:].replace("_helper", "") for x in session_method.names]

    @session_method
    def _load_session_helper(self, task_id, name, time_str, data={}):
        """Load a session with given key `name`/`time_str`

        Args:
            task_id: The `task_id` for the task
            name: `name` of the session
            time_str: The time stamp of the session
            data: Any additional data passed from the request

        """
        key = name + "/" + time_str
        data_dir = os.path.join(self._root_dir, name, time_str)
        try:
            self._sessions[name]["sessions"][time_str] = {}
            self._create_trainer(task_id, name, time_str, data_dir,
                                 config=None, load=True, **data)
            with open(os.path.join(self._sessions[name]["path"], time_str,
                                   "session_state"), "r") as f:
                self._sessions[name]["sessions"][time_str]["state"] = json.load(f)
            self._debug_and_put(task_id, True, f"Loaded Session for {key}")
        except Exception as e:
            self._error_and_put(task_id, False, f"Could not load Session for {key}. {e}"
                                + "\n" + traceback.format_exc())

    @session_method
    def _unload_session_helper(self, task_id, name, time_str=None, data=None):
        """Unload a session with given key `name`/`time_str`

        Args:
            task_id: The `task_id` for the task
            name: `name` of the session
            time_str: The time stamp of the session
            data: Any additional data passed from the request

        """
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
        """Purge a session with given key `name`/`time_str`

        Args:
            task_id: The `task_id` for the task
            name: `name` of the session
            time_str: The time stamp of the session
            data: Any additional data passed from the request

        """
        self._logd(f"Purging {name}/{time_str}")
        try:
            sub_task_id = self._create_id()
            self._unload_session_helper(sub_task_id, name, time_str)
            result = self._check_result(sub_task_id)  # cannot be None
            if result[1]:
                shutil.rmtree(os.path.join(self._root_dir, name, time_str))
                dirlist = os.listdir(os.path.join(self._root_dir, name))
                # Remove the full directory if only modules remain
                if dirlist == ["modules"]:
                    shutil.rmtree(self._root_dir, name)
                self._sessions[name]["sessions"].pop(time_str)
                self._debug_and_put(task_id, True, f"Purged session {name}/{time_str}")
            else:
                self._error_and_put(*result)
        except Exception as e:
            self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())

    @session_method
    def _archive_session_helper(self, task_id: int, name: str,
                                time_str: str, data: Any = None):
        """Archive a session with given key `name`/`time_str`

        Args:
            task_id: The `task_id` for the task
            name: `name` of the session
            time_str: The time stamp of the session
            data: Any additional data passed from the request

        Schemas:
            class ArchiveSessionModel(BaseModel):
                saves: List[pathlib.Path]
                keep_checkpoint: bool
                notes: str

        """
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
        """Load a session with given key `name`/`time_str` to a given server

        Args:
            task_id: The `task_id` for the task
            name: `name` of the session
            time_str: The time stamp of the session
            data: CloneToServerModel

        Schemas:
            class CloneToServerModel(BaseModel):
                server: ipaddress.IPv4Address
                config: Dict
                saves: Optional[List[pathlib.Path]]
                modules: Optional[List[str]]

        """
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
                    savedir = os.path.join(self._root_dir, name, time_str, "savedir")
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
                for f in _files.values():
                    f.close()
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
        """Clone the session with `name`/`time_str` and optional given config differences

        Args:
            task_id: The `task_id` for the task
            name: `name` of the session
            time_str: The time stamp of the session
            data: Any additional data passed from the request

        Schemas:
            class CloneSessionModel(BaseModel):
                config: trainer.config.Config
                saves: Optional[List[pathlib.Path]]

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
        """Reinitialize a session with given key `name`/`time_str`

        Args:
            task_id: The `task_id` for the task
            name: `name` of the session
            time_str: The time stamp of the session
            data: ReinitSessionModel

        Schemas:
            class ReinitSessionModel(BaseModel):
                config: trainer.config.Config

        """
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
        data_dir = os.path.join(self._root_dir, name, time_str)
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
                    data.pop("session_key")
                    func(task_id, session_name, timestamp, data=data)
                except Exception as e:
                    self._error_and_put(task_id, False, f"{e}" + "\n" + traceback.format_exc())
        else:
            self._error_and_put(task_id, False, "Incorrect data format given")

    def _session_check_get(self, func_name):
        func = getattr(self, func_name)
        return _dump(func())

    def _session_check_post(self, func_name):
        """Handle a POST request for a `session_method`

        Args:
            func_name: The name of the helper method

        Schema:
            class Task(BaseModel):
                task_id: int
                message: str

        Request:
            content-type: MimeTypes.json
            body:
                session_key: str
                data: Union[:meth:`daemon.Daemon._archive_session_helper`: ArchiveSessionModel,
                            :meth:`_reinit_session_helper`: ReinitSessionModel,
                            :meth:`_clone_session_helper`: CloneSessionModel,
                            :meth:`_clone_to_helper`: CloneToServerModel,
                            Dict]

        Responses:
            Invalid data: ResponseSchema(405, "Invalid Data", MimeTypes.text,
                                        "Invalid data {some: json}")
            bad params: ResponseSchema(405, "Bad Params", MimeTypes.text,
                                       "Session key not in params")
            Success: ResponseSchema(200, "Initiated Task", MimeTypes.json, "Task")

        """
        # session_key: str
        # data: Union[ReinitSessionModel, CloneSessionModel,
        #             CloneToServerModel, ArchiveSessionModel]
        if not flask_login.current_user.is_authenticated:
            return self.login_manager.unauthorized()
        try:
            data = load_json(request.json)
            if data is None:
                return _dump([False, f"Invalid data {request.json}"])
        except Exception as e:
            return f"Invalid data {request.json}, {e}"
        if "session_key" not in data:
            return f"'session_key' must be in data.\nInvalid data {data}"
        else:
            task_id = self._get_task_id_launch_func(self._session_method_check, func_name, data)
            return _dump([True, {"task_id": task_id,
                                 "message": f"{func_name.split('_')[0]}ing {data}"}])

    def stop(self):
        # NOTE: clear the fwd_ports_event
        # if self.fwd_ports_event is not None:
        #     self.fwd_ports_event.clear()
        #     self._logi("Waiting for the fwd_ports_thread to join. Can take upto 60 seconds")
        # if self.fwd_ports_thread is not None:
        #     self.fwd_ports_thread.join()
        #     self._logi("Joined fwd_ports thread")
        # # NOTE: kill the fwd ports
        # if self.fwd_procs is not None:
        #     self._logi("Killing fwd_procs")
        #     for p in self.fwd_procs.values():
        #         p.kill()
        # NOTE: Unload the sessions
        self._logi("Unloading sessions")
        for k in self._sessions:
            self._unload_session_helper(-1, k)
        # NOTE: Kill the have_internet process if it exists
        # self._logi("Killing have_internet process")
        # if self._have_internet is not None:
        #     self._have_internet.kill()
        # self._logi("Killed have_internet process")

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
            """

            Responses:
                success: ResponseSchema(200, "Index page", MimeTypes.text, "")

            """
            return render_template("index.html")

        # if self._template_dir:
        #     @self.app.route("/<filename>", methods=["GET"])
        #     def __files(filename: Union[Path, None] = None):
        #         """
        #         Responses:
        #             success: ResponseSchema(200, "Index page", MimeTypes.text, "")
        #         """
        #         def read_file(mode):
        #             with open(os.path.join(self._template_dir, filename), mode) as f:
        #                 content = f.read()
        #             return content
        #         filename = escape(filename)
        #         # FIXME: Cache isn't used as of now
        #         # if filename in self._cache:
        #         #     content = self._cache[filename]["content"]
        #         #     mimetype = self._cache[filename]["mimetype"]
        #         #     return Response(content, mimetype=mimetype)
        #         if filename in os.listdir(self._template_dir):
        #             if filename.endswith("css"):
        #                 mode = "r"
        #                 mimetype = "text/css"
        #             elif filename.endswith("js"):
        #                 mode = "r"
        #                 mimetype = "text/css"
        #             elif filename.endswith("png"):
        #                 mode = "rb"
        #                 mimetype = "image/png"
        #             elif filename.endswith("jpg") or filename.endswith("jpeg"):
        #                 mode = "rb"
        #                 mimetype = "image/jpeg"
        #             else:
        #                 print(filename)
        #                 mode = "r"
        #                 mimetype = "text/html"
        #             content = read_file(mode)
        #             self._cache[filename] = {}
        #             self._cache[filename]["content"] = content
        #             self._cache[filename]["mimetype"] = mimetype
        #             return Response(content, mimetype=mimetype)
        #         else:
        #             return Response("Not found", status=404)

        # NOTE: Simplest way would be to proxy it
        #       Although a better way would be to get the function
        #       for the url rule and return value from it
        # @self.app.route("/trainer/<int:port>/<endpoint>", methods=["GET", "POST"])
        # @flask_login.login_required
        # def __trainer(port=None, endpoint=None):
        #     sess_list = self.sessions_list
        #     if port not in [x["port"] for x in sess_list.values()]:
        #         return Response(_dump([False, f"Unloaded or invalid trainer {port}"]))
        #     session = [*filter(lambda x: x["port"] == port, sess_list.values())][0]
        #     if not session["loaded"]:
        #         return _dump([False, "Trainer is not loaded"])
        #     try:
        #         print(f"{request.json}, {request.data}, {request.form}")
        #         _json = _data = _files = None
        #         if request.json:
        #             _json = request.json if isinstance(request.json, dict)\
        #                 else json.loads(request.json)
        #         if request.form:
        #             _data = dict(request.form)
        #         if request.files:
        #             _files = request.files
        #         response = requests.request(request.method, f"http://localhost:{port}/{endpoint}",
        #                                     files=_files, json=_json, data=_data)
        #         excluded_headers = ["content-encoding", "content-length",
        #                             "transfer-encoding", "connection"]
        #         headers = [(name, value) for (name, value) in response.raw.headers.items()
        #                    if name.lower() not in excluded_headers]
        #         response = Response(response.content, response.status_code, headers)
        #         return response
        #     except Exception as e:
        #         return Response(_dump([False, f"Error occured {e}"]))

        # @self.app.route("/trainer/<int:port>/<category>/<endpoint>", methods=["GET", "POST"])
        # @flask_login.login_required
        # def __trainer_one(port=None, category=None, endpoint=None):
        #     sess_list = self.sessions_list
        #     if port not in [x["port"] for x in sess_list.values()]:
        #         return Response(_dump([False, f"Unloaded or invalid trainer {port}"]))
        #     session = [*filter(lambda x: x["port"] == port, sess_list.values())][0]
        #     if not session["loaded"]:
        #         return _dump([False, "Trainer is not loaded"])
        #     try:
        #         _json = _data = _files = None
        #         if request.json:
        #             _json = request.json if isinstance(request.json, dict)\
        #                 else json.loads(request.json)
        #         if request.form:
        #             _data = dict(request.form)
        #         if request.files:
        #             _files = request.files
        #         response = requests.request(request.method,
        #                                     f"http://localhost:{port}/{category}/{endpoint}",
        #                                     files=_files, json=_json, data=_data)
        #         excluded_headers = ["content-encoding", "content-length",
        #                             "transfer-encoding", "connection"]
        #         headers = [(name, value) for (name, value) in response.raw.headers.items()
        #                    if name.lower() not in excluded_headers]
        #         response = Response(response.content, response.status_code, headers)
        #         return response
        #     except Exception as e:
        #         return Response(_dump([False, f"Error occured {e}"]))

        # @self.app.route("/sessions", methods=["GET"])
        # @flask_login.login_required
        # def __list_sessions():
        #     """Returns a dictionary of sessions, their ports if they're alive and the
        #     state. Rest of the communication can be done with session

        #     With optional argument {name}, if the session name starts with
        #     {name} then all those sessions are returned.

        #     """
        #     try:
        #         name = request.args.get("name")
        #     except Exception:
        #         name = None
        #     sess_list = self.sessions_list
        #     if name:
        #         name = name.strip()
        #     if name:
        #         sessions = {k: v for k, v in sess_list.items()
        #                     if k.startswith(name)}
        #         if sessions:
        #             return _dump([True, sessions])
        #         else:
        #             return _dump([False, "No session found"])
        #     else:
        #         return _dump([True, self.sessions_list])

        @self.app.route("/current_user", methods=["GET"])
        @flask_login.login_required
        def __current_user():
            """Return the name of the current user.

            Tags:
                daemon, user

            This is in case we're logged in and username is not known to the
            client as they have refreshed and the store state is gone (LOL,
            FIXME)

            Schemas:
                class Success(BaseModel): user: str

            Responses:
                Success: ResponseSchema(200, "Current logged in user", MimeTypes.json, "Success")

            """
            return _dump([True, {"user": flask_login.current_user.name}])

        # CHECK: Why's this function so complicated?
        @self.app.route("/update_given_name", methods=["POST"])
        @flask_login.login_required
        def __update_given_name():
            """Update the name of a given trainer.

            Tags:
                daemon, maintenance

            Requests:
                content-type: MimeTypes.multipart
                body:
                    given_name: str
                    trainer_url: str
                    session_key: str

            Responses:
                bad params: ResponseSchema(400, "Bad params", MimeTypes.text, "given_name not in params")
                Success: ResponseSchema(200, "Current logged in user", MimeTypes.text, "Successfully assigned name")

            """
            data = load_json(request.json)
            if data is None:
                return _dump([False, f"Invalid data {request.json}"])
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
                    return _dump([False, "Could not assign name"])

        @self.app.route("/create_session", methods=["POST"])
        @flask_login.login_required
        def __create_session():
            """Create a new session.

            Tags:
                daemon, session

            Requests:
                content-type: MimeTypes.json
                body:
                    data: daemon.models.CreateSessionModel

            Schemas:
                class Task(BaseModel):
                    task_id: int
                    message: str

            Responses:
                Success: ResponseSchema(200, "Current logged in user", MimeTypes.json, "Task")

            """
            data = load_json(request.json)
            if data is None:
                return _dump([False, f"Invalid data {request.json}"])
            # except Exception as e:
            #     return _dump([False, f"{e}" + "\n" + traceback.format_exc()])
            task_id = self._get_task_id_launch_func(self.create_session, data)
            return _dump([True, {"task_id": task_id,
                                 "message": "Creating session with whatever data given"}])

        # FIXME: How's this different from create_session?
        @self.app.route("/upload_session", methods=["POST"])
        @flask_login.login_required
        def __upload_session():
            """Upload an archived session and create it.

            Tags:
                daemon, session

            Requests:
                params:
                    name: str

            Schemas:
                class Task(BaseModel):
                    task_id: int
                    message: str

            Responses:
                Success: ResponseSchema(200, "Current logged in user", MimeTypes.json, "Task")

            """
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
            """Returns all the docs for all the endpoints in the server.

            The trainer docs can be fetched from the trainer itself.

            Tags:
                daemon, docs

            Schemas:
                class Docs(BaseModel):
                    docs: Dict[str, str]

            Responses:
                Success: ResponseSchema(200, "Docs for the server", MimeTypes.json, "Docs")

            """
            endpoints = {x.rule: x.endpoint for x in self.app.url_map.iter_rules()
                         if "<" not in x.rule}
            docs = {e.replace("/", "", 1): self.app.view_functions[endpoints[e]].__doc__
                    for e in endpoints}
            preamble = """Some docs are missing for some endpoints and will be added soon.  Example
            calls and return values can also be added. In that sense the
            functions should become self documenting soon.
            """
            return _dump({"preamble": preamble, **docs})

        # @self.app.route("/check_task", methods=["GET"])
        # @flask_login.login_required
        # def __check_task():
        #     """Check and return the status of a task submitted earlier.

        #     :methods: GET
        #     :args: [task_id]
        #     :retval: list[bool, dict[task_id, result, message]]
        #     """
        #     try:
        #         task_id = int(request.args.get("task_id").strip())
        #     except Exception as e:
        #         return _dump([False, f"Bad params {e}" + "\n" + traceback.format_exc()])
        #     if task_id not in self.__task_ids:
        #         return _dump([False, f"No such task: {task_id}"])
        #     else:
        #         result = self._check_result(task_id)
        #     if result is None:
        #         return _dump([True, {"task_id": task_id, "result": 0,
        #                              "message": "Not yet processed"}])
        #     else:
        #         if len(result) == 2:
        #             self._logw(f"Result of length 2 for check_task {result}")
        #             return _dump([True, {"task_id": result[0], "result": True,
        #                                  "message": "Successful"}])
        #         elif len(result) == 3 and result[1]:
        #             return _dump([True, {"task_id": result[0], "result": True,
        #                                  "message": result[2]}])
        #         elif len(result) == 3 and not result[1]:
        #             return _dump([True, {"task_id": result[0], "result": False,
        #                                  "message": result[2]}])
        #         else:
        #             return _dump([True, result])

        @self.app.route("/_version", methods=["GET"])
        def __version():
            """Return version of the current server.

            Tags:
                daemon, status

            Responses:
                success: ResponseSchema(200, "Server Version", MimeTypes.text, "0.3.0")
            """
            return self.__version__

        # FIXME: We're not checking overlap
        # FIXME: Should only be available to interfaces
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
                        self._devices[data["port"]] = list(set(self._devices[data["port"]])
                                                           - set(data["gpus"]))
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
            """Login to the server

            Tags:
                daemon, user

            Requests:
                body:
                    username: str
                    password: str

            Responses:
                logged_in: ResponseSchema(200, "Logged in", MimeTypes.text, "Logged in")
                not logged in: ResponseSchema(400, "not logged in", MimeTypes.text, "Could not log in")

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
            """Check if the user is logged in

            Tags:
                daemon, user

            Responses:
                logged_in: ResponseSchema(200, "Logged in", MimeTypes.text, "Logged in")
                not logged in: ResponseSchema(400, "not logged in", MimeTypes.text, "Could not log in")

            """
            if flask_login.current_user.is_authenticated:
                return "Logged in"
            else:
                return "Could not Login"

        @self.app.route("/logout", methods=["GET"])
        @flask_login.login_required
        def __logout():
            """Logout from the server.

            Tags:
                daemon, user

            Responses:
                logged out: ResponseSchema(400, "Logged out", MimeTypes.text, "Logged out")
            """
            flask_login.logout_user()
            return "Logged Out"

        @self.app.route("/list_session_modules", methods=["GET"])
        @flask_login.login_required
        def __list_session_modules():
            """NOT IMPLEMENTED: This feature isn't implemented yet and should not be called.

            Returns the list of modules available with a given session.

            Tags:
                daemon, session

            Responses:
                not implemented: ResponseSchema(400, "Not Implemented", MimeTypes.text, "Not Implemented")

            """
            return "Not Implemented yet"

        @self.app.route("/add_session_module", methods=["POST"])
        @flask_login.login_required
        def __add_session_module():
            """NOT IMPLEMENTED: This feature isn't implemented yet and should not be called.

            Load a module into a given session's load path. Parameters and
            methods are similar to `add_global_module`. Additional param
            {session_key} has to be provided.

            Responses:
                not implemented: ResponseSchema(400, "Not Implemented", MimeTypes.text, "Not Implemented")

            """
            return "Not implemented yet"
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

            Tags:
                daemon, session

            Responses:
                not implemented: ResponseSchema(400, "Not Implemented", MimeTypes.text, "Not Implemented")

            """
            return "Not implemented yet"
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

            Tags:
                daemon, modules

            Schemas:
                class Success(BaseModel): default: Dict[str, Any]

            Responses:
                Success: ResponseSchema(200, "Sucess", MimeTypes.json, "Success")

            """
            return _dump([True, self._modules])

        @self.app.route("/add_global_module", methods=["POST"])
        @flask_login.login_required
        def __add_global_module():
            """Add a module to the global modules.

            Can be python or zip file. Shows up in global modules and is immediately
            available for loading to all sessions. Will overwrite if a module
            with the same name already exists.

            NOTE: Make sure that the imports are reloaded if there's an update

            Tags:
                daemon, modules

            Requests:
                params:
                    name: str

            Schemas:
                class Task(BaseModel):
                    task_id: int
                    message: str

            Responses:
                Success: ResponseSchema(200, "Current logged in user", MimeTypes.json, "Task")

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
            """Delete the given module from the list of global modules.

            The module is immediately unavailable for all future running functions.

            NOTE: Make sure that the imports are reloaded if there's an update
            NOTE: What if a function relied on some deleted module? It should
                  no longer work. Not sure how to handle that.
            NOTE: Module names start with _module_ internally and the module name
                  itself shouldn't start with _

            Tags:
                daemon, modules

            Requests:
                params:
                    name: str

            Responses:
                not such module: ResponseSchema(404, "No such module", MimeTypes.text, "No such module mode_name")
                bad params: ResponseSchema(400, "Name not in params", MimeTypes.text, "Name of module required")
                Success: ResponseSchema(200, "Deleted Dataset", MimeTypes.text, "Deleted module MNIST")

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
            """Return the list of global datasets available.

            Tags:
                daemon, datasets

            Schemas:
                class Dataset(BaseModel):
                    default: :attr:`datasets`.returns

            Responses:
                Success: ResponseSchema(200, "Datasets", MimeTypes.json, "Dataset")

            """
            return make_json(self.datasets)

        @self.app.route("/upload_dataset", methods=["POST"])
        @flask_login.login_required
        def __upload_dataset():
            """Upload a dataset which would be globally available to the server.

            Must be zip file. An __init__.py should be at the top of the zip file and
            should access the data with relative paths or through the
            network. No assumptions about absolute paths should be made.

            The dataset name should be given in form along with description and
            the dataset must implement __len__ and __getitem__.

            Type of data should also be mentioned.

            Tags:
                daemon, datasets

            Schemas:
                class Dataset(BaseModel):
                    default: :attr:`datasets`.returns

            Requests
                params:
                    name: str
                    description: str
                    type: str

            Responses:
                bad params: ResponseSchema(405, "Bad Params", MimeTypes.text,
                            "name not in request")
                Error: ResponseSchema(405, "Error Occurred", MimeTypes.text,
                            "Some error occured: error while parsing file")
                Success: ResponseSchema(200, "Uploaded Successfully", MimeTypes.text,
                            "Uploaded Dataset MNIST successfully")

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
            """Delete a given dataset

            Tags:
                daemon, datasets

            Requests:
                params:
                    name: str

            Responses:
                not such dataset: ResponseSchema(404, "No such dataset", MimeTypes.text,
                                           "No such dataset mode_name")
                bad params: ResponseSchema(400, "Name not in params", MimeTypes.text,
                                           "Name of dataset required")
                Success: ResponseSchema(200, "Deleted Dataset", MimeTypes.text, "Deleted dataset MNIST")

            """
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

        @self.app.route("/_ping", methods=["GET"])
        def __ping():
            """Return Pong.

            Tags:
                daemon, status

            Schemas:
                class Pong(BaseModel):
                    pong: str = "pong"

            Responses:
                success: ResponseSchema(200, "Pong", MimeTypes.text, "Pong")

            """
            return "pong"

        @self.app.route("/_name", methods=["GET"])
        def __name():
            """Return Daemon Name.

            Tags:
                daemon, status

            Responses:
                success: ResponseSchema(200, "Daemon Name", MimeTypes.text, "SomeDaemon")

            """
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
        #
        # @self.app.route("/view_session", methods=["POST"])
        # def __view_session():
        #     # only views the progress and parameters. No trainer is started
        #     return _dump([False, "Doesn't do anything"])

        # NOTE: Add session_methods.  Routes are added by removing "_" prefix
        #       and "_helper" suffix from self._session_methods
        for x in self._session_methods:
            self.app.add_url_rule("/" + x, x, partial(self._session_check_post, x),
                                  methods=["POST"])

        trainer_view = Trainer.as_view("trainer", self)
        self.app.add_url_rule("/trainer/<int:port>/<endpoint>",
                              view_func=trainer_view)
        self.app.add_url_rule("/trainer/<int:port>/<category>/<endpoint>",
                              view_func=trainer_view)

        check_task = CheckTask.as_view("check_task", self)
        self.app.add_url_rule("/check_task",
                              view_func=check_task)

        sessions = Sessions.as_view("sessions", self)
        self.app.add_url_rule("/sessions",
                              view_func=sessions)

        @atexit.register
        def cleanup():
            self.stop()

        @self.app.route("/_shutdown", methods=["GET"])
        @flask_login.login_required
        def __shutdown_server():
            """Shutdown the machine

            Tags:
                daemon, maintenance

            Responses:
                Success: ResponseSchema(200, "Shutting Down", MimeTypes.text, "Shutting Down")

            """
            self._logd("Shutdown called via HTTP. Shutting down.")
            Thread(target=cleanup).start()
            func = request.environ.get('werkzeug.server.shutdown')
            func()
            return "Shutting down"

        serving.run_simple(self.hostname, self.port, self.app, threaded=True,
                           ssl_context=self.context)
