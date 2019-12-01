import ipdb
import sys
import ssl
import json
from threading import Thread
from functools import partial

from time import sleep
from flask import Flask, render_template, request, Response
from flask_cors import CORS
from werkzeug import serving


def _dump(x):
    return json.dumps(x, default=lambda o: f"<<{type(o).__qualname__}>>")
    # return json.dumps(x, default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")


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

    def wrapper_control(self, func_name):
        self.trainer.__class__.__dict__[func_name](self.trainer)
        return _dump("Performing %s\n" % func_name)

    def wrapper_props(self, prop_name):
        return _dump(self.trainer.__class__.__dict__[prop_name].fget(self.trainer))

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

        @self.app.route('/_extras/call_adhoc_run', methods=["POST"])
        def __call_adhoc_run():
            data = request.json()
            ipdb.set_trace()
            # What does this return?
            response = self.trainer.call_adhoc(data)

        @self.app.route('/_extras/get_adhoc_run_output')
        def __get_adhoc_run_output():
            response = self.trainer.report_adhoc_run()

        @self.app.route("/update", methods=["POST"])
        def __update():
            data = json.loads(request.data.decode("utf-8"))
            if not all(x in data for x in ["model_params", "trainer_params", "dataloader_params"]):
                return Response(json.dumps("Invalid data format in update request"), status=412,
                                mimetype="application/json")
            status = self.trainer.try_update(data)
            if status:
                return json.dumps(status)
            else:
                return json.dumps(False)

        for x, y in self.trainer.controls.items():
            self.app.add_url_rule("/" + x, x, partial(self.wrapper_control, x))
        for x in self.trainer.props:
            self.app.add_url_rule("/" + "props/" + x, x, partial(self.wrapper_props, x))
        serving.run_simple(self.api_host, self.api_port, self.app, ssl_context=self.context)
