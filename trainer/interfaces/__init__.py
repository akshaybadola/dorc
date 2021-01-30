from typing import Any, Optional, Dict, Union, Any, Iterable, List, Callable
import os
import re
import sys
import ssl
import glob
import json
import atexit
import shutil
import logging
import requests
import traceback
from functools import partial
from pathlib import Path
from trainer.spec.models import BaseModel

from flask import Flask, render_template, request, Response
from flask_cors import CORS
from werkzeug import serving

from ..util import _dump, deprecated
from ..trainer import Trainer
from ..trainer.models import Return
from ..mods import Modules
from .._log import Log

from .views import ConfigFile


def make_return(status: bool, message: str) -> Return:
    return Return(status=status, message=message)


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
        self._daemon_url = "http://localhost:20202/"
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
        self._orig_config = None
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
    def config_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.data_dir, "session_config")) or
                os.path.exists(os.path.join(self.data_dir, "session_config.py")))

    @property
    def state_exists(self) -> bool:
        return os.path.exists(os.path.join(self.data_dir, "session_state"))

    @property
    def reserved_gpus(self) -> List[int]:
        return requests.get(self._daemon_url + "_devices").json()

    def reserve_gpus(self, gpus: List[int]) -> List[Union[bool, None, str]]:
        response = requests.post(self._daemon_url + "_devices",
                                 json={"action": "reserve",
                                       "gpus": gpus,
                                       "port": self.api_port})
        result = json.loads(response.content)
        return result

    def free_gpus(self, gpus: List[int]) -> Dict:
        response = requests.post(self._daemon_url + "_devices",
                                 json={"action": "free",
                                       "gpus": gpus,
                                       "port": self.api_port})
        result = json.loads(response.content)
        return result

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
            self._orig_config = json.loads(_dump(config))  # self._orig_config is now serializable
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
            self.trainer.reserved_gpus = lambda: self.reserved_gpus
            self.trainer.reserve_gpus = self.reserve_gpus
            self.trainer._init_all()
            return True, "Created Trainer"
        except Exception as e:
            return False, f"{e}" + "\n" + traceback.format_exc()

    def create_trainer(self, config: Union[Path, str, None] = None):
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
                         f"Missing cert or key. Details: {e}" +
                         "\n" + traceback.format_exc())
        else:
            self.context = None

    def trainer_control(self, func_name: str) -> Any:
        """Call a trainer `control`

        Tags:
            trainer, controls

        Map:
            /<control>: :class:`~trainer.Trainer`.<control>

        Responses:
            not found: ResponseSchema(404, "Not found", MimeTypes.text, "No such control")
            success: ResponseSchema(200, "Success", :class:`trainer.Trainer`.<control>,
                                    :class:`trainer.Trainer`.<control>)

        """
        retval = getattr(self.trainer, func_name)()
        if retval:
            return _dump(retval)
        else:
            return _dump("Performing %s\n" % func_name)

    def trainer_props(self, prop_name: str) -> Any:
        """Return a property `prop_name` from the trainer

        Tags:
            trainer, properties

        Map:
            /props/<prop_name>: :class:`~trainer.Trainer`.<prop_name>

        Responses:
            not found: ResponseSchema(404, "Not found", MimeTypes.text, "Property not found")
            success: ResponseSchema(200, "Success", :class:`trainer.Trainer`.<prop_name>,
                                    :class:`trainer.Trainer`.<prop_name>)

        """
        return _dump(self.trainer.__class__.__dict__[prop_name].fget(self.trainer))

    def trainer_get(self, func_name: str) -> Response:
        """Call a trainer `func_name` via HTTP GET, after some checks.

        Args:
            func_name: Name of the function to call

        Returns:
            An instance of :class:`Return`

        """
        retval = getattr(self.trainer, func_name)()
        status = 200 if retval.status else 400
        return Response(retval.json(), status=status, mimetype='application/json')

    @deprecated
    def trainer_post_form(self, func_name: str) -> Response:
        status, response = getattr(self.trainer, func_name)(request)
        if status:
            response = _dump([True, response])
            return Response(response, status=200, mimetype='application/json')
        else:
            response = _dump([False, response])
            return Response(response, status=500, mimetype='application/json')

    def remove_class(self, v: Any) -> str:
        return re.sub(r'<class \'?\"?([a-zA-Z\._]+)\'?\"?>', r'\1', str(v))

    def get_model_from_args(self, func) -> BaseModel:
        annotations = func.__annotations__.copy()
        annotations.pop("return", None)
        lines = [f"    {k}: {self.remove_class(v)}" for k, v in annotations.items()]
        ldict: Dict[str, Any] = {}
        exec("\n".join(["class Annot(BaseModel):", *lines]), globals(), ldict)
        return ldict["Annot"]

    def check_data_trainer_method(self, func_name: str, data: Dict):
        func = getattr(self.trainer, func_name)
        if func is None:
            return Return(status=False, message=f"No such method {func_name}")
        else:
            Model = self.get_model_from_args(func)
            try:
                model = Model(**data)
            except Exception as e:
                return make_return(False, f"{e}")
            return make_return(True, "")

    def trainer_post(self, func_name: str) -> Response:
        """Call a trainer `func_name` via HTTP POST, after some checks.

        Args:
            func_name: Name of the function to call

        Returns:
            An instance of :class:`Return`

        """
        if hasattr(request, "json"):
            data = request.json
        elif hasattr(request, "form") and request.form:
            data = {}
            for k, v in request.form.items():
                data[k] = request.form[k]
        else:
            retval = Return(status=False, message="No data given")
            return Response(retval.json(), status=400, mimetype='application/json')
        retval = self.check_data_trainer_method(func_name, data)
        if retval.status:
            retval = getattr(self.trainer, func_name)(data)
        status = 200 if retval.status else 400
        return Response(retval.json(), status=status, mimetype='application/json')

    def trainer_route(self, func_name: str) -> Response:
        """Return a property `prop_name` from the trainer

        Tags:
            trainer, methods

        Map:
            /methods/<method_name>: :class:`~trainer.Trainer`.<method_name>
            /extras/<method_name>: :class:`~trainer.Trainer`.<method_name>

        Responses:
            not found: ResponseSchema(404, "Not found", MimeTypes.text, "Method not found")
            bad params: ResponseSchema(405, "Bad Params", MimeTypes.text,
                                            "Required param weights not in params")
            invalid data: ResponseSchema(405, "Invalid Data", MimeTypes.text,
                                              "Invalid data {some: json}")
            success: ResponseSchema(200, "Success", :class:`trainer.Trainer`.<method_name>,
                                                    :class:`trainer.Trainer`.<method_name>)

        """
        if request.method == "POST":
            return self.trainer_post(func_name)
        elif request.method == "GET":
            return self.trainer_get(func_name)
        else:
            return Response(status=405)

    def trainer_internals(self, func_name: str) -> Response:
        if hasattr(request, "json"):
            data = request.json
            if "secret" not in data:
                return make_return(False, "Secret not in data")
            elif "secret" in data and not __ifaceunti__(data["secret"]):
                return make_return(False, "Bleh")
            else:
                data.pop("secret")
                status, response = getattr(self.trainer, func_name)(**data)
            response = _dump(response)
            if not status:
                return Response(response, status=400, mimetype='application/json')
            else:
                return Response(response, status=200, mimetype='application/json')
        else:
            response = make_return(False, "No data given")
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

        @self.app.route('/_ping')
        def __ping():
            """Return Pong.

            Tags:
                iface, status

            Schemas:
                class Pong(BaseModel):
                    pong: str = "pong"

            Responses:
                success: ResponseSchema(200, "Pong", MimeTypes.text, "Pong")

            """
            return "pong"

        @self.app.route('/props')
        def __props():
            """Return list of available trainer properties

            Tags:
                trainer, properties

            Responses:
                success: ResponseSchema(200, "List of trainer properties", MimeTypes.json,
                                        "See :attr:`~trainer.Trainer.props`")

            """
            return _dump(self.trainer.props)

        @self.app.route('/docs')
        def __docs():
            """Return documentation for trainer and interface

            Currently only docs for `~trainer.Trainer.props` are sent. Rest will
            be added soon.

            Tags:
                trainer, docs

            Responses:
                success: ResponseSchema(200, "Success", MimeTypes.json,
                                        "See :attr:`~trainer.Trainer.props`")

            """
            return self.trainer.docs()

        @self.app.route("/batch_props", methods=["POST"])
        def __batch_props():
            """Return trainer properties which are requested.

            Instead of retrieving a single prop at a time, this helps reducing
            the network overhead as one can request multiple props directly.

            Tags:
                trainer, properties

            Responses:
                success: ResponseSchema(200, "Success", MimeTypes.json,
                                        "See :attr:`~trainer.Trainer.props`")

            """
            print(f"{request.json}, {request.data}, {request.form}")
            if hasattr(request, "json"):
                props = request.json
                if not props or "props_list" not in props:
                    return _dump([False, "Need dict with key \"props_list\""])
                try:
                    result = [True, dict(map(lambda x: (x, json.loads(self.trainer_props(x))),
                                             props["props_list"]))]
                except Exception as e:
                    result = [False, f"Error occured {e}"]
                response = _dump(result)
            else:
                response = _dump([False, f"Error: No data given"])
            return Response(response, status=200, mimetype='application/json')

        @self.app.route("/_shutdown", methods=["GET"])
        def __shutdown_server():
            """Shutdown the machine

            Tags:
                interface, maintenance

            Responses:
                Success: ResponseSchema(200, "Shutting Down", MimeTypes.text, "Shutting Down")

            """
            func = request.environ.get('werkzeug.server.shutdown')
            func()
            return "Shutting down"

        # # TODO: This should be a loop over add_rule like controls
        # # TODO: Type check, with types allowed in {"bool", "int", "float", "string", "list[type]"}
        # #       one level depth check
        @self.app.route("/destroy", methods=["GET"])
        def __destroy():
            """Not Implemented

            Tags:
                interface, maintenance

            Responses:
                Success: ResponseSchema(200, "Destroying", MimeTypes.text, "Destroying")

            """
            self._logi("Destroying")
            self._logi("Does nothing for now")

        @self.app.route("/config", methods=["GET"])
        def __config():
            """Retrieve the current config, original config and overrides (if any) as a json object

            Tags:
                trainer, maintenance

            Schema:
                class Success(BaseModel):
                    config: trainer.config.Config
                    orig_config: trainer.config.Config
                    overrides: Dict

            Responses:
                Success: ResponseSchema(200, "Success", MimeTypes.json, "Success")

            """
            return _dump({"config": self._current_config,
                          "orig_config": self._orig_config,
                          "overrides": self._current_overrides})

        @self.app.route("/list_files", methods=["GET"])
        def __list_files():
            """Retrieve the list of files (config files?)

            Tags:
                trainer, maintenance

            Schema:
                class Success(BaseModel):
                    default: List[pathlib.Path]

            Responses:
                Success: ResponseSchema(200, "Success", MimeTypes.json, "Success")

            """
            if os.path.exists(os.path.join(self.data_dir, "session_config.py")):
                files = ["/session_config.py"]
            else:
                files = [f.replace(self.data_dir, "") for f in
                         glob.glob(os.path.join(self.data_dir, "session_config", "**/*.py"),
                                   recursive=True)]
            return _dump([True, {"files": files}])

        @self.app.route("/get_file", methods=["POST"])
        def __get_file():
            """Return a file from the given path

            Tags:
                trainer, maintenance

            Schema:
                class Success(BaseModel):
                    default: List[pathlib.Path]

            Responses:
                Not found: ResponseSchema(404, "File not found", MimeTypes.text, "File not found")
                Success: ResponseSchema(200, "Success", MimeTypes.binary, "Success")

            """
            filepath = request.json["filepath"].strip()
            if filepath.startswith("/"):
                filepath = filepath[1:]
            filepath = os.path.join(self.data_dir, filepath)
            if os.path.exists(filepath):
                self._logd(f"Sending file {filepath}")
                with open(filepath, "rb") as f:
                    data = f.read()
                return data
            else:
                return Response(self._loge(f"File not Found {filepath}"), status=404)

        @self.app.route("/put_file", methods=["POST"])
        def __put_file():
            """Put a given file to the path

            Tags:
                trainer, maintenance

            Requests:
                content-type: MimeTypes.multipart
                body:
                    filepath: str
                    file: bytes

            Responses:
                Bad params: ResponseSchema(405, "Bad Params", MimeTypes.text,
                                          "File not sent with request")
                Error: ResponseSchema(400, "Error occured", MimeTypes.text,
                                      "Error occured: directory not writable")
                Success: ResponseSchema(200, "Success", MimeTypes.text, "Written successfully")

            """
            if not request.files:
                return Response(self._loge("File not sent with request"), status=405)
            elif "filepath" not in request.form or not request.form["filepath"]:
                return Response(self._loge("Filepath not sent with request"), status=405)
            else:
                try:
                    filepath = request.form["filepath"]
                    file_bytes = request.files["file"].read()
                    if filepath.startswith("/"):
                        filepath = filepath[1:]
                    filepath = os.path.join(self.data_dir, filepath)
                    with open(filepath, "w") as f:
                        f.write(file_bytes.decode("utf-8"))
                    return self._logi(f"Updated file {filepath}")
                except Exception as e:
                    return self._loge(f"Exception Occured Config {e}. Reverted to earlier." +
                                      "\n" + traceback.format_exc())

        # # TODO: Should restart self also, but we can't restart self, only daemon can. LOL
        # @self.app.route("/config_file", methods=["GET", "POST"])
        # def __config_file():
        #     config_file = os.path.join(self.data_dir, "session_config.py")
        #     if not os.path.exists(config_file):
        #         config_file = os.path.join(config_file[:-3], "__init__.py")
        #     if request.method == "GET":
        #         with open(config_file, "rb") as f:
        #             config = f.read()
        #         return config
        #     elif request.method == "POST":
        #         try:
        #             if not request.files:
        #                 return Response("File not sent with request", status=405)
        #             else:
        #                 file_bytes = request.files["file"].read()
        #                 shutil.copy(config_file, config_file + ".bak")
        #                 with open(config_file, "w") as f:
        #                     f.write(file_bytes.decode("utf-8"))
        #                 return self._logi("Updated Config")
        #         except Exception as e:
        #             shutil.copy(config_file + ".bak", config_file)
        #             return self._loge(f"Exception Occured Config {e}. Reverted to earlier." +
        #                               "\n" + traceback.format_exc())

        config_file_rule = ConfigFile.as_view("config_file", self)
        self.app.add_url_rule("/config_file",
                              view_func=config_file_rule)


        @self.app.route("/update_restart", methods=["POST"])
        def __update_restart():
            """Not Implemented

            Tags:
                interface, maintenance

            Responses:
                Success: ResponseSchema(200, "Update and restart", MimeTypes.text, "Update and restart")

            """
            return _dump("Does nothing for now")

        # NOTE: Add rule for each property `prop` of `Trainer`"
        for x in self.trainer.props:
            self.app.add_url_rule("/" + "props/" + x, x, partial(self.trainer_props, x))

        # NOTE: Controls
        for x, y in self.trainer.controls.items():
            self.app.add_url_rule("/" + x, x, partial(self.trainer_control, x))

        # NOTE: Adding helpers
        for x, y in self.trainer.methods.items():
            http_methods = []
            if "POST" in y.__http_methods__:
                http_methods.append("POST")
            if "GET" in y.__http_methods__:
                http_methods.append("GET")
            self.app.add_url_rule("/methods/" + x, x, partial(self.trainer_route, x),
                                  methods=http_methods)

        # NOTE: Adding extras
        for x, y in self.trainer.extras.items():
            if "GET" in y.__http_methods__:
                self.app.add_url_rule("/extras/" + x, x, partial(self.trainer_get, x),
                                      methods=["GET"])
            elif "POST" in y.__http_methods__:
                self.app.add_url_rule("/extras/" + x, x, partial(self.trainer_post, x),
                                      methods=["POST"])

        for x, y in self.trainer._internals.items():
            self.app.add_url_rule("/_internals/" + x, x, partial(self.trainer_internals, x),
                                  methods=["POST"])
        # reference to url_map for generating schemas
        self.endpoints = self.app.url_map
        serving.run_simple(self.api_host, self.api_port, self.app, threaded=True,
                           ssl_context=self.context)
