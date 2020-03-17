from typing import Dict, Iterable, Union
import os
import sys
import ssl
import json
import atexit
import shutil
import logging
import traceback
from functools import partial

from flask import Flask, render_template, request, Response
from flask_cors import CORS
from werkzeug import serving

from .util import _dump
from .trainer import Trainer
from .mods import Modules
from ._log import Log


def __ifaceunti__(_n):
    if _n == "_sxde#@_":
        return True
    else:
        return False


class FlaskInterface:
    """Flask Interface to the trainer, to create, destroy and control the
    trainer. Everything's communicated as JSON.
    """

    def __init__(self, hostname, port, data_dir, bare=True, production=False, no_start=False,
                 config_overrides=None):
        """
        :param hostname: :class:`str` host over which to serve
        :param port: :class:`int` port over which to serve
        :param trainer: :class:`trainer.Trainer` instance
        :param bare: `deprecated` whether to server html files or not
        :returns: None
        :rtype: None

        """
        self.api_host = hostname
        self.api_port = port
        self.logger = None
        self.data_dir = data_dir
        self.bare = bare
        self.production = production
        self.config_overrides = config_overrides
        self.app = Flask(__name__)
        CORS(self.app, supports_credentials=True)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        self.use_https = False
        self.verify_user = True
        self._init_context()
        self._logger = logging.getLogger("trainer_logger")
        log_file = os.path.join(self.data_dir, "trainer_logger")
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
        self._modules = Modules(self.data_dir, self._logd, self._loge,
                                self._logi, self._logw)
        self._current_config = None
        self._current_overrides = None
        if (self.api_host and self.api_port and self.config_exists):
            self._logi("Creating Trainer")
            status, message = self.create_trainer()
            if status and no_start:
                self._logi("no_start given. Not starting")
            elif status and not no_start:
                self._logi("Starting server")
                self.start()
            else:
                self._loge(f"Error creating trainer {message}")
        elif (not self.state_exists and self.config_exists and
                not self.api_host and not self.api_port):
            self._logi(f"Initializing Trainer State")
            status, message = self.create_trainer()
            self._logd(f"{status}, {message}")
            self.trainer = None
            del self.trainer
        elif not self.state_exists and not self.config_exists:
            self._logd(f"Config doesn't exist. Cannot create trainer")

    @property
    def config_exists(self):
        return (os.path.exists(os.path.join(self.data_dir, "session_config"))
                or os.path.exists(os.path.join(self.data_dir, "session_config.py")))

    @property
    def state_exists(self):
        return os.path.exists(os.path.join(self.data_dir, "session_state"))

    def _update_config(self, config: Dict, overrides: Iterable[Union[int, float, str]]):
        def _check(conf, seq):
            status = True
            inner = conf.copy()
            for s in seq:
                status = s in inner
                if status:
                    inner = inner.__getitem__(s)
                else:
                    return False
            return status

        def _set(conf, seq):
            c = conf
            for x in seq[:-2]:
                print(c, x)
                c = c[x]
            c[seq[-2]] = seq[-1]

        for o in overrides:
            if _check(config, o[:-1]):
                _set(config, o)

    def _create_trainer_helper(self):
        try:
            if self.data_dir not in sys.path:
                sys.path.append(self.data_dir)
                from session_config import config
                sys.path.remove(self.data_dir)
            else:
                from session_config import config
            overrides_file = os.path.join(self.data_dir, "config_overrides.json")
            if self.config_overrides is not None:
                self._logd(f"Config Overrides given: \n{self.config_overrides}" +
                           "\nWill write to file")
                self._update_config(config, self.config_overrides)
                with open(overrides_file, "w") as f:
                    json.dump(self.config_overrides, f)
            if os.path.exists(overrides_file):
                self._logd(f"Config Overrides File exists. Loading")
                with open(overrides_file, "r") as f:
                    config_overrides = json.load(f)
                    self._update_config(config, config_overrides)
            else:
                config_overrides = None
            self._current_config = config
            self._current_overrides = config_overrides
            self.trainer = Trainer(**{"data_dir": self.data_dir, "production": self.production,
                                      **config})
            self.trainer._init_all()
            return True, "Created Trainer"
        except Exception as e:
            return False, f"{e}" + "\n" + traceback.format_exc()

    def create_trainer(self, config=None):
        if self.config_exists:
            return self._create_trainer_helper()
        elif not self.config_exists and config is None:
            return False, "No existing config"
        elif not self.config_exists and config is not None:
            status, result = self.check_config(config)
            if status:
                return self._create_trainer_helper()
            else:
                return status, result

    def check_config(self, config, env=None):
        status, result = self._modules.add_config(self.data_dir, config, env=env)
        if status:
            return status, None
        else:
            return status, result

    # TODO: I might have to give the names for various thingies while generating
    #       certificates for it to work correctly.
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
                         f"Missing cert or key. Details: {e}"
                         + "\n" + traceback.format_exc())
        else:
            self.context = None

    def trainer_control(self, func_name):
        retval = getattr(self.trainer, func_name)()
        if retval:
            return _dump(retval)
        else:
            return _dump("Performing %s\n" % func_name)

    def trainer_props(self, prop_name):
        return _dump(self.trainer.__class__.__dict__[prop_name].fget(self.trainer))

    def trainer_get(self, func_name):
        status, response = getattr(self.trainer, func_name)()
        if status:
            return _dump({"success": response})
        else:
            return _dump({"error": response})

    def trainer_post_form(self, func_name):
        status, response = getattr(self.trainer, func_name)(request)
        if status:
            response = _dump({"success": response})
            return Response(response, status=200, mimetype='application/json')
        else:
            response = _dump({"error": response})
            return Response(response, status=500, mimetype='application/json')

    def trainer_post(self, func_name):
        if hasattr(request, "json"):
            data = request.json
            status, response = getattr(self.trainer, func_name)(data)
            response = _dump(response)
            if not status:
                return Response(response, status=400, mimetype='application/json')
            else:
                return Response(response, status=200, mimetype='application/json')
        else:
            response = _dump({"error": "No data given"})
            return Response(response, status=400, mimetype='application/json')

    def trainer_route(self, func_name):
        if request.method == "POST":
            if getattr(getattr(self.trainer, func_name), "content_type", "json") == "form":
                return self.trainer_post_form(func_name)
            else:
                return self.trainer_post(func_name)
        elif request.method == "GET":
            return self.trainer_get(func_name)
        else:
            return Response(status=405)

    def trainer_internals(self, func_name):
        if hasattr(request, "json"):
            data = request.json
            if "secret" not in data:
                return _dump([False, None])
            elif "secret" in data and not __ifaceunti__(data["secret"]):
                return _dump([False, None])
            else:
                data.pop("secret")
                status, response = getattr(self.trainer, func_name)(**data)
            response = _dump(response)
            if not status:
                return Response(response, status=400, mimetype='application/json')
            else:
                return Response(response, status=200, mimetype='application/json')
        else:
            response = _dump({"error": "No data given"})
            return Response(response, status=400, mimetype='application/json')

    def start(self):
        @atexit.register
        def kill_trainer():
            print("AT EXIT, kill trainer process")

        if not self.bare:
            @self.app.route('/')
            def __index():
                mobile = any([x in str(request.user_agent).lower()
                              for x in ["mobile", "android"]])
                return render_template("index_2.html",
                                       props=[p for p in self.trainer.props if p != "all_params"],
                                       controls=self.trainer.controls,
                                       mobile=mobile)

        @self.app.route('/props')
        def __props():
            return _dump(self.trainer.props)

        @self.app.route("/_shutdown", methods=["GET"])
        def __shutdown_server():
            func = request.environ.get('werkzeug.server.shutdown')
            func()
            return "Shutting down"

        # # TODO: This should be a loop over add_rule like controls
        # # TODO: Type check, with types allowed in {"bool", "int", "float", "string", "list[type]"}
        # #       one level depth check
        @self.app.route("/destroy", methods=["GET"])
        def __destroy():
            self._logi("Destroying")
            self._logi("Does nothing for now")

        @self.app.route("/config", methods=["GET"])
        def __config():
            return _dump({"config": self._current_config,
                          "overrides": self._current_overrides})

        # TODO: Should restart self also, but we can't restart self, only daemon can. LOL
        @self.app.route("/config_file", methods=["GET", "POST"])
        def __config_file():
            config_file = os.path.join(self.data_dir, "session_config.py")
            if not os.path.exists(config_file):
                config_file = os.path.join(config_file[:-3], "__init__.py")
            if request.method == "GET":
                with open(config_file, "rb") as f:
                    config = f.read()
                return config
            elif request.method == "POST":
                try:
                    file_bytes = request.files["file"].read()
                    shutil.copy(config_file, config_file + ".bak")
                    with open(config_file, "w") as f:
                        f.write(file_bytes.decode("utf-8"))
                    return self._logi("Updated Config")
                except Exception as e:
                    shutil.copy(config_file + ".bak", config_file)
                    return self._loge(f"Exception Occured Config {e}. Reverted to earlier." +
                                      "\n" + traceback.format_exc())

        @self.app.route("/update_restart", methods=["POST"])
        def __update_restart():
            return _dump("Does nothing for now")

        # NOTE: Props
        for x in self.trainer.props:
            self.app.add_url_rule("/" + "props/" + x, x, partial(self.trainer_props, x))

        # NOTE: Controls
        for x, y in self.trainer.controls.items():
            self.app.add_url_rule("/" + x, x, partial(self.trainer_control, x))

        # NOTE: Adding extras
        for x, y in self.trainer._extras.items():
            if "GET" in y.__http_methods__:
                self.app.add_url_rule("/_extras/" + x, x, partial(self.trainer_get, x),
                                      methods=["GET"])
            elif "POST" in y.__http_methods__:
                self.app.add_url_rule("/_extras/" + x, x, partial(self.trainer_post, x),
                                      methods=["POST"])

        # NOTE: Adding helpers
        for x, y in self.trainer._helpers.items():
            methods = []
            if "POST" in y.__http_methods__:
                methods.append("POST")
            if "GET" in y.__http_methods__:
                methods.append("GET")
            self.app.add_url_rule("/_helpers/" + x, x, partial(self.trainer_route, x),
                                  methods=methods)

        for x, y in self.trainer._internals.items():
            self.app.add_url_rule("/_internals/" + x, x, partial(self.trainer_internals, x),
                                  methods=["POST"])
        serving.run_simple(self.api_host, self.api_port, self.app, ssl_context=self.context)
