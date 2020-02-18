import os
import sys
import ssl
import json
import socket
import shutil
import atexit
import argparse
import datetime
import configparser
from queue import Queue
from threading import Thread
from multiprocessing import Process
from flask import Flask, render_template, request, Response
from flask_cors import CORS
from werkzeug import serving

from .mods import Modules
from .trainer import Trainer
from .interfaces import FlaskInterface
from .util import _dump


class Daemon:
    def __init__(self, hostname, port, data_dir):
        self.hostname = hostname
        self.port = port
        self.data_dir = data_dir
        self.logger = None
        self.app = Flask(__name__)
        CORS(self.app)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        self.use_https = False
        self.verify_user = True
        self._last_free_port = self.port
        self._threads = {}
        self._task_q = Queue()
        self._sessions = {}
        self._init_context()
        self._task_id = 0
        self.__task_ids = []
        self._results = []

        # FIXME: all the log functions should be created here
        def print_func(x):
            print(x)
        self._modules = Modules(self.data_dir, print_func, print_func,
                                print_func, print_func)

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
                         f"Missing cert or key. Details: {e}")
        else:
            self.context = None

    def _update_results(self):
        while not self._task_q.empty():
            self._results.append(self._task_q.get())

    def _check_config(self, config):
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

    def _create_id(self):
        # 0 reserverd for instance
        self._task_id += 1
        self.__task_ids.append(self._task_id)
        return self._task_id

    def _get_task_id_launch_func(self, func, *args):
        task_id = self._create_id()
        self._threads[task_id] = Thread(target=func, args=[task_id, *args])
        self._threads[task_id].start()
        return task_id

    def _check_result(self, task_id):
        self._update_results()
        for x in self._results:
            if task_id == x[0]:
                return x
        else:
            return None

    # TODO: scan_sessions should be called at beginning and after that should
    #       raise error unless testing
    def scan_sessions(self):
        session_names = [x for x in os.listdir(self.data_dir) if
                         os.path.isdir(os.path.join(self.data_dir, x))]
        for s in session_names:
            self._sessions[s] = {}
            self._sessions[s]["path"] = os.path.join(self.data_dir, s)
            self._sessions[s]["sessions"] = {}
            for d in os.listdir(self._sessions[s]["path"]):
                try:
                    data_dir = os.path.join(self._sessions[s]["path"], d)
                    self._sessions[s]["sessions"][d] = {}
                    self._sessions[s]["sessions"][d]["data_dir"] = data_dir
                    with open(os.path.join(self._sessions[s]["path"], d, "session_state"),
                              "r") as f:
                        self._sessions[s]["sessions"][d]["state"] = json.load(f)
                except Exception as e:
                    self._sessions[s]["sessions"][d] = "Error " + str(e)

    def create_session(self, task_id, data):
        """Creates a new training session from given data"""
        session_name = data["name"]
        if session_name not in self._sessions:
            self._sessions[session_name] = {}
            self._sessions[session_name]["path"] = os.path.join(self.data_dir, session_name)
            self._sessions[session_name]["sessions"] = {}
            if not os.path.exists(self._sessions[session_name]["path"]):
                os.mkdir(self._sessions[session_name]["path"])  # /sessions/funky_session
        time_str = datetime.datetime.now().isoformat()
        # Like: /sessions/funky_session/2020-02-17T10:53:06.458827
        data_dir = os.path.join(self._sessions[session_name]["path"], time_str)
        os.mkdir(data_dir)
        self._sessions[session_name]["sessions"][time_str] = {}
        self._create_trainer(task_id, session_name, time_str, data_dir, data["config"])
        with open(os.path.join(self._sessions[session_name]["path"], time_str,
                               "session_state")) as f:
            self._sessions[session_name]["sessions"][time_str]["state"] = json.load(f)

    def _create_trainer(self, task_id, name, time_str, data_dir, config, add=True):
        if add and self._check_config(config):
            try:
                status, result = self._modules.add_config(data_dir, config)
                if status:
                    trainer = Trainer(**{"data_dir": data_dir, **result})
                    trainer._init_all()
                else:
                    self._task_q.put((task_id, False, f"Could not read config. {result}"))
            except Exception as e:
                self._task_q.put((task_id, False, f"{e}"))
        elif not add and self._check_config(config):
            try:
                trainer = Trainer(**{"data_dir": data_dir, **config})
                trainer._init_all()
                port = self._find_open_port()
                iface = FlaskInterface(self.hostname, port, trainer)
                self._sessions[name]["sessions"][time_str]["config"] = config
                self._sessions[name]["sessions"][time_str]["trainer"] = trainer
                self._sessions[name]["sessions"][time_str]["port"] = port
                self._sessions[name]["sessions"][time_str]["iface"] = iface
                self._sessions[name]["sessions"][time_str]["data_dir"] = data_dir
                p = Process(target=iface.start)
                self._sessions[name]["sessions"][time_str]["process"] = p
                p.start()
                self._task_q.put((task_id, True))
            except Exception as e:
                self._task_q.put((task_id, False, f"{e}"))
        else:
            self._task_q.put((task_id, False, "Check failed on config"))

    def _refresh_state(self, session_key):
        name, time_str = session_key.split("/")
        with open(os.path.join(self._sessions[name]["path"], time_str,
                               "session_state"), "r") as f:
            self._sessions[name]["sessions"][time_str]["state"] = json.load(f)

    def _unload_finished_session(self, session_key):
        self._refresh_state(session_key)
        name, time_str = session_key.split("/")
        state = self._sessions[name]["sessions"][time_str]["state"]
        if self._check_session_finished(state):
            self._unload_helper(name, time_str)

    def _refresh_all_loaded_sessions(self):
        for name in self._sessions:
            for time_str, sess in self._sessions[name]["sessions"].items():
                if "process" in sess:
                    self._refresh_state("/".join([name, time_str]))

    def load_unfinished_sessions(self):
        for name, session in self._sessions.items():
            for sub_name, sub_sess in session["sessions"].items():
                state = sub_sess["state"]
                if self._check_session_finished(state):
                    self.load_session(0, {"session_key": "/".join([name, sub_name])})

    def _check_session_finished(self, state) -> bool:
        epoch = state["epoch"]
        max_epochs = state["trainer_params"]["max_epochs"]
        iterations = state["iterations"]
        max_iterations = state["trainer_params"]["max_iterations"]
        if (not iterations and not max_iterations and epoch < max_epochs) or\
           (not epoch and not max_epochs and iterations < max_iterations):
            return True
        else:
            return False

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

    def load_session(self, task_id, data):
        """Loads a training session into memory"""
        key = data["session_key"]
        session_name, timestamp = key.split("/")
        valid, error_str = self._check_session_valid(session_name, timestamp)
        print(f"LOAD {data}")
        if not valid:
            self._task_q.put((task_id, valid, error_str))
        else:
            data_dir = os.path.join(self.data_dir, key)
            config_candidates = [x for x in os.listdir(data_dir)
                                 if "session_config" in x]
            if not len(config_candidates) == 1:
                self._task_q.put((task_id, False, f"Error. More than one config detected"))
            else:
                # CHECK: How to resolve multiple entries in sys.path having same
                #        module names Currently I remove dir from sys.path after
                #        loading.  Not sure if it'll work correctly.
                # NOTE: I think it's not really causing an issue.
                if data_dir not in sys.path:
                    sys.path.append(data_dir)
                from session_config import config
                sys.path.remove(data_dir)
            self._sessions[session_name]["sessions"][timestamp] = {}
            self._create_trainer(task_id, session_name, timestamp, data_dir, config, False)
            with open(os.path.join(self._sessions[session_name]["path"], timestamp,
                                   "session_state"), "r") as f:
                self._sessions[session_name]["sessions"][timestamp]["state"] = json.load(f)
            print([(k, v.keys()) for k, v in self._sessions["meh_session"]["sessions"].items()])

    # TODO: log unloaded sessions
    def _unload_helper(self, name, time_str=None):
        def _unload(name, time_str):
            if "process" in self._sessions[name]["sessions"][time_str]:
                self._sessions[name]["sessions"][time_str]["process"].terminate()
                self._sessions[name]["sessions"][time_str].pop("process")
            self._sessions[name]["sessions"][time_str]["trainer"] = None
            self._sessions[name]["sessions"][time_str]["config"] = None
            self._sessions[name]["sessions"][time_str].pop("trainer")
            self._sessions[name]["sessions"][time_str].pop("config")
            if "port" in self._sessions[name]["sessions"][time_str]:
                self._sessions[name]["sessions"][time_str].pop("port")
            if "iface" in self._sessions[name]["sessions"][time_str]:
                self._sessions[name]["sessions"][time_str].pop("iface")
        if time_str is None:
            for k in self._sessions[name]["sessions"].keys():
                _unload(name, k)
        else:
            _unload(name, time_str)

    def _purge_helper(self, task_id, name, time_str):
        try:
            self._unload_helper(name, time_str)
            shutil.rmtree(os.path.join(self.data_dir, name, time_str))
            self._sessions[name]["sessions"].pop(time_str)
            self._task_q.put((task_id, True, f"Purged session {name}/{time_str}"))
        except Exception as e:
            self._task_q.put((task_id, False, f"{e}"))

    def unload_session(self, task_id, data):
        """Unloads a training session from memory"""
        print("DATA Unload session", data)
        if "session_key" in data:
            session_name, timestamp = data["session_key"].split("/")
            valid, error_str = self._check_session_valid(session_name, timestamp)
            if not valid:
                self._task_q.put((task_id, valid, error_str))
            else:
                try:
                    self._unload_helper(session_name, timestamp)
                    self._task_q.put((task_id, True, f"Unloaded session {data['session_key']}"))
                except Exception as e:
                    self._task_q.put((task_id, False, f"{e}"))
        elif "session_name" in data:
            session = data["session_name"]
            try:
                self._unload_helper(session)
                self._task_q.put((task_id, True, f"Unloaded all sessions for {session_name}"))
            except Exception as e:
                self._task_q.put((task_id, False, f"{e}"))
        else:
            self._task_q.put((task_id, False, "Incorrect data format"))

    def compare_sessions(self):
        pass

    def purge_session(self, task_id, data):
        """Unloads the session and removes it from disk"""
        if "session_key" in data:
            session_name, timestamp = data["session_key"].split("/")
            valid, error_str = self._check_session_valid(session_name, timestamp)
            if not valid:
                self._task_q.put((task_id, valid, error_str))
            else:
                try:
                    self._purge_helper(task_id, session_name, timestamp)
                except Exception as e:
                    self._task_q.put((task_id, False, f"{e}"))
        else:
            self._task_q.put((task_id, False, "Incorrect data format"))

    @property
    def _sessions_list(self):
        retval = {}
        for k, v in self._sessions.items():
            session_stamps = v["sessions"].keys()
            for st in session_stamps:
                key = k + "/" + st
                session = v["sessions"][st]
                retval[key] = {}
                retval[key]["loaded"] = "process" in session
                retval[key]["port"] = session["port"] if retval[key]["loaded"] else None
                retval[key]["state"] = session["state"]
                retval[key]["finished"] = session["state"]
        return _dump(retval)

    def start(self):
        self.scan_sessions()
        self.load_unfinished_sessions()

        @self.app.route("/sessions", methods=["GET"])
        def __list_sessions():
            """Returns a dictionary of sessions, their ports if they're alive and the
            state. Rest of the communication can be done with session"""
            return self._sessions_list

        @self.app.route("/new_session", methods=["POST"])
        def __new_session():
            # TODO:
            # - Creates a new session with given data
            # - Displays config editor for the user
            # - Or the client should display?
            if "name" not in request.form:
                return _dump([False, "Name not in request"])
            else:
                try:
                    data = request.form["name"]
                    file_bytes = request.files["file"].read()
                except Exception as e:
                    return _dump([False, f"{e}"])
            data = {"name": data, "config": file_bytes}
            task_id = self._get_task_id_launch_func(self.create_session, data)
            return _dump({"task_id": task_id,
                          "message": "Creating session with whatever data given"})

        @self.app.route("/load_session", methods=["POST"])
        def __load_session():
            data = json.loads(request.json)
            if "session_key" not in data:
                return _dump(f"Invalid data {data}")
            else:
                task_id = self._get_task_id_launch_func(self.load_session, data)
                return _dump({"task_id": task_id, "message": f"Loading session with {data}"})

        @self.app.route("/unload_session", methods=["POST"])
        def __unload_session():
            data = json.loads(request.json)
            if not ("session_key" in data or "session_name" in data):
                return _dump(f"Invalid data {data}")
            else:
                task_id = self._get_task_id_launch_func(self.unload_session, data)
                return _dump({"task_id": task_id,
                              "message": f"Unloading session: {data['session_key']}"})

        # TODO: Fix design issues. Should only return return json, not "False, json"
        @self.app.route("/purge_session", methods=["POST"])
        def __purge_session():
            data = json.loads(request.json)
            if "session_key" not in data:
                return _dump(f"Invalid data {data}")
            else:
                task_id = self._get_task_id_launch_func(self.purge_session, data)
                return _dump({"task_id": task_id, "message": f"Purging session {data}"})

        @self.app.route("/check_task", methods=["GET"])
        def __check_task():
            try:
                task_id = int(request.args.get("task_id").strip())
            except Exception as e:
                return _dump(f"Bad params")
            if task_id not in self.__task_ids:
                return _dump(f"No such task: {task_id}")
            else:
                result = self._check_result(task_id)
            if result is None:
                return _dump({"task_id": task_id, "message": "Not yet processed"})
            else:
                if len(result) == 2:
                    return _dump({"task_id": result[0], "message": "Successful"})
                elif len(result) == 3:
                    return _dump({"task_id": result[0], "message": result[2]})
                else:
                    return _dump(result)

        @self.app.route("/_shutdown", methods=["GET"])
        def __shutdown_server():
            func = request.environ.get('werkzeug.server.shutdown')
            func()
            return "Shutting down"

        # NOTE: This should be disabled for now. Only if the number of sessions
        #       gets too large should I use this, as otherwise all the session
        #       data should be sent to the client.
        @self.app.route("/view_session", methods=["POST"])
        def __view_session():
            # only views the progress and parameters. No trainer is started
            return _dump("Doesn't do anything")

        @atexit.register
        def unload_all():
            print("SELF RESULTS", self._results)
            for k in self._sessions:
                self._unload_helper(k)

        serving.run_simple(self.hostname, self.port, self.app, ssl_context=self.context)


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
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    daemon = Daemon(args.hostname, args.port, args.data_dir)
    return daemon


def _start_daemon(hostname, port, data_dir):
    daemon = Daemon(hostname, port, data_dir)
    Thread(target=daemon.start).start()
    return daemon


if __name__ == "__main__":
    daemon = create_daemon()
    daemon.start()
