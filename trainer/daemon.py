import os
import sys
import ssl
import json
import argparse
import datetime
import configparser
from threading import Thread
from flask import Flask, render_template, request, Response
from flask_cors import CORS
from werkzeug import serving

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
        self._sessions = {}
        self._init_context()

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
                         "Missing cert or key. Details: {}".format(e))
        else:
            self.context = None

    # CHECK: How's the state decided? State will be updated by the Trainer
    #        * One way can be for trainer to write to both to ".pth" files
    #          and to the session_state file (with a backup)
    #        * session_params for trainer?
    def scan_sessions(self):
        session_names = [x for x in os.listdir(self.data_dir) if
                         os.path.isdir(os.path.join(self.data_dir, x))]
        for s in session_names:
            self._sessions[s] = {}
            self._sessions[s]["session_dir"] = os.path.join(self.data_dir, s)
            self._sessions[s]["sessions"] = {}
            for d in os.listdir(self._sessions[s]["session_dir"]):
                try:
                    self._sessions[s]["sessions"][d] =\
                        json.load(open(os.path.join(d, "session_state")))
                except Exception as e:
                    self._sessions[s]["sessions"][d] = "Error " + str(e)

    def _check_params(self, params):
        # FIXME
        pass

    def _find_open_port(self):
        pass

    def _create_id(self):
        return 0

    def _get_task_id(self, func, *args):
        task_id = self._create_id()
        self._threads[task_id] = Thread(target=func, args=[task_id, *args])
        self._threads[task_id].start()
        return task_id

    def _check_result(self, task_id):
        if task_id in self._results:
            return self._results[task_id]
        else:
            return None

    def create_session(self, task_id, data):
        # TODO: Put the result in the queue with the task_id
        name = data["name"]
        if name not in self._sessions:
            self._sessions[name] = {}
        time_str = datetime.datetime.now().isoformat()
        os.mkdir(os.path.join(self._sessions[name]["session_dir"], time_str))
        session_name = name + "_" + time_str
        if self._check_params(data["params"]):
            # TODO: load modules
            data_dir = os.path.join(self._sessions[name]["session_dir"],
                                    time_str)
            trainer = Trainer({"data_dir": data_dir, **data["params"]})
            port = self._find_open_port()
            iface = FlaskInterface(self.hostname, port, trainer)
            self._sessions[name]["sessions"][session_name]["trainer"] = trainer
            self._sessions[name]["sessions"][session_name]["port"] = port
            self._sessions[name]["sessions"][session_name]["iface"] = iface
            self._sessions[name]["sessions"][session_name]["data_dir"] = data_dir
            # CHECK: Should it be a separate process? Probably yes.
            Thread(target=iface.start).start()
            return True
        else:
            return False

    def load_unfinished_sessions(self):
        for name, session in self._sessions:
            if session:
                pass
        pass

    def start(self):
        self.scan_sessions()
        self.load_unfinished_sessions()

        @self.app.route("/sessions", methods=["GET"])
        def __sessions():
            return _dump(self._sessions)

        @self.app.route("/new_session", methods=["POST"])
        def __new_session():
            # Creates a new session with given data
            # Displays config editor
            data = request.data
            task_id = self._get_task_id(self.create_session, data)
            return _dump({task_id, "Creating session with whatever data given"})

        @self.app.route("/unload_session", methods=["POST"])
        def __unload_session():
            pass

        # FIXME
        @self.app.route("/purge_session", methods=["POST"])
        def __purge_session():
            data = request.data
            try:
                self.unload_session(data)
                self.purge_session(data)
            except Exception as e:
                return False, _dump(e)
            return True, f"Purged session {data}"

        @self.app.route("/load_session", methods=["POST"])
        def __load_session():
            # Loads a previously created a session and starts a trainer for it
            # Steps:
            # 1. Check if all required modules are present
            #    The data structure will be simple and they can be found in session["modules"]
            # 2. Start the trainer with the params and modules
            # 3. Can't be done async as that'll hold up the client. I can have a
            #    simple timer with the client so that it keeps checking back with the server.

            # the client can check for the task if it's completed or not.
            task_id = self._get_task_id(self._load_session)
            return _dump({task_id, "Loading session with whatever data given"})

        @self.app.route("/check_task", methods=["GET"])
        def __check_task():
            data = request.data
            # FIXME: data should be parsed correctly
            return self._check_result(data)

        # NOTE: This should be disabled for now. Only if the number of sessions
        #       gets too large should I use this, as otherwise all the session
        #       data should be sent to the client.
        @self.app.route("/view_session", methods=["POST"])
        def __view_session():
            # only views the progress and parameters. No trainer is started
            return _dump("Doesn't do anything")

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
                        help="The directory which will contain all the session and model" +
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


if __name__ == "__main__":
    daemon = create_daemon()
