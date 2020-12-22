import json
import requests
from flask import request, Response
from flask.views import MethodView
from flask_login import login_required

from ..util import _dump


def pull_from_function():
    return None


class Trainer(MethodView):
    decorators = [login_required]

    def __init__(self, daemon):
        self.daemon = daemon
        self.responses = {"invalid_port": {405: {"description": "Invalid port given",
                                                 "content": {"application/text":
                                                             "Unloaded or invalid trainer {port}"}}},
                          "not_loaded": {405: {"description": "Trainer not loaded",
                                               "content": {"application/json":
                                                           "Trainer not loaded"}}},
                          "success": {200: {"description": "Successful request",
                                            "content": {"applicatoin/json":
                                                        {"schema": "Trainer not loaded"}}}}}

    def schema(self):
        """In case it's props then the schema description and summary can be pulled from
        props. The return value can be pulled from function return types."""
        return {"description": "something",
                "summary": "some summary",
                "requestBody": {"description": "Whatever's required",
                                "content": {"application/json":
                                            {"schema": pull_from_function()}}}}

    def get(self, port: int, endpoint: str, category: str = None):
        return self.check_and_dispatch(request, port, category, endpoint)

    def post(self, port: int, endpoint: str, category: str = None):
        return self.check_and_dispatch(request, port, category, endpoint)

    def check_and_dispatch(self, request, port, category, endpoint):
        sess_list = self.daemon._sessions_list
        if port not in [x["port"] for x in sess_list.values()]:
            # return make_response(f"Unloaded or invalid trainer {port}", 405)
            return Response(_dump([False, f"Unloaded or invalid trainer {port}"]))
        else:
            session = [*filter(lambda x: x["port"] == port, sess_list.values())][0]
            if not session["loaded"]:
                return _dump([False, "Trainer is not loaded"])
                # return make_response("Trainer is not loaded", 405)
            else:
                return self.call_trainer(request, port, category, endpoint)

    def call_trainer(self, request, port, category, endpoint):
        try:
            _json = _data = _files = None
            if request.json:
                _json = request.json if isinstance(request.json, dict)\
                    else json.loads(request.json)
            if request.form:
                _data = dict(request.form)
            if request.files:
                _files = request.files
            if category is None:
                response = requests.request(request.method, f"http://localhost:{port}/{endpoint}",
                                            files=_files, json=_json, data=_data)
            else:
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
