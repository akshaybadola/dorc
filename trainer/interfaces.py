import sys
import ssl
import json
from threading import Thread
from functools import partial

from time import sleep
from flask import Flask, render_template, request, Response
from flask_cors import CORS
from werkzeug import serving

from .util import _dump


class AWSInterface:
    def __init__(self):
        self.counter = 0

    def fetch_from_sqs(self, sqs_queue, shared_queue):
        self.logger.info("Initializing polling thread")
        message = True
        while True:
            message = sqs_queue.receive_messages()
            sleep(1.01)
            if message:
                self.logger.info("Received %s" % str(message[0].body))
                if isinstance(message[0].body, str):
                    shared_queue.put(json.loads(message[0].body.replace("'", '"')))
                elif isinstance(message[0].body, dict):
                    shared_queue.put(message[0].body)
                message[0].delete()  # easier maybe
            else:
                self.logger.info("Received nothing")

    def push_to_sqs(self, sqs_queue, message, group=None):
        self.counter += 1
        response = sqs_queue.send_message(MessageBody=json.dumps(message),
                                          MessageDeduplicationId=str(self.counter),
                                          MessageGroupId=str(group))
        self.logger.info("Pushed to queue with result %s" % response)

    def fetch_from_s3(self, bucket, key, filename):
        try:
            bucket.download_file(Key=key, Filename=filename)
            self.logger.debug("Downloaded file from %s to %s" % (key, filename))
            return True
        except Exception as e:
            self.logger.error(str(e))
            return False

    def push_to_s3(self, bucket, data):
        pass


class HTTPInterface:
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port

    def push_to_server(self):
        pass

    def poll_from_server(self):
        pass


class FlaskInterface:
    # CHECK: This Func is not used
    class Func:
        def __init__(self, name, caller):
            self.name = name
            self.caller = caller

        def __call__(self):
            Thread(target=self.caller).start()
            print("Performing %s" % self.name)

    def __init__(self, hostname, port, trainer, bare=False):
        self.api_host = hostname
        self.api_port = port
        self.logger = None
        self.trainer = trainer
        self.bare = bare
        self.app = Flask(__name__)
        CORS(self.app)
        self.app.config['TEMPLATES_AUTO_RELOAD'] = True
        self.use_https = False
        self.verify_user = True
        self._init_context()

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
                         "Missing cert or key. Details: {}".format(e))
        else:
            self.context = None

    def trainer_control(self, func_name):
        self.trainer.__class__.__dict__[func_name](self.trainer)
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
            # if not func_name == "fetch_image":
            #     response = _dump(response)
            #     print(response)
            # else:
            #     response = json.dumps(response)
            response = _dump(response)
            if not status:
                return Response(response, status=400, mimetype='application/json')
            else:
                return Response(response, status=200, mimetype='application/json')
        else:
            response = _dump({"error": "No data given"})
            print(response)
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

    def start(self):
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

        # # TODO: This should be a loop over add_rule like controls
        # # TODO: Type check, with types allowed in {"bool", "int", "float", "string", "list[type]"}
        # #       one level depth check
        # # @self.app.route('/_extras/call_adhoc_run', methods=["POST"])
        # def __call_adhoc_run():
        #     error_dict = {"required_atleast_[split]": ["train", "val", "test"],
        #                   "required_for_[split]": {"metrics": "[list[string]]_which_metrics",
        #                                            "epoch": "[int|string]_which_epoch",
        #                                            "fraction": "[float]_fraction_of_dataset"}}
        @self.app.route("/destroy", methods=["GET"])
        def __destroy():
            self.logger.info("Destroying")
            self.logger.info("Does nothing for now")

        for x, y in self.trainer.controls.items():
            self.app.add_url_rule("/" + x, x, partial(self.trainer_control, x))
        for x in self.trainer.props:
            self.app.add_url_rule("/" + "props/" + x, x, partial(self.trainer_props, x))

        # Adding extras
        for x, y in self.trainer._extras.items():
            if "GET" in y.__http_methods__:
                self.app.add_url_rule("/_extras/" + x, x, partial(self.trainer_get, x),
                                      methods=["GET"])
            elif "POST" in y.__http_methods__:
                self.app.add_url_rule("/_extras/" + x, x, partial(self.trainer_post, x),
                                      methods=["POST"])

        # Adding helpers
        for x, y in self.trainer._helpers.items():
            methods = []
            if "POST" in y.__http_methods__:
                methods.append("POST")
            if "GET" in y.__http_methods__:
                methods.append("GET")
            self.app.add_url_rule("/_helpers/" + x, x, partial(self.trainer_route, x),
                                  methods=methods)
        serving.run_simple(self.api_host, self.api_port, self.app, ssl_context=self.context)
