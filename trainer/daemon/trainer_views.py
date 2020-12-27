from typing import Optional
import json
import requests
from flask import Request, request, Response
from flask.views import MethodView
from flask_login import login_required

from ..util import _dump


class Trainer(MethodView):
    decorators = [login_required]

    def __init__(self, daemon):
        self.daemon = daemon

    def get(self, port: int, endpoint: str, category: Optional[str] = None) -> Response:
        """GET response from trainer method `endpoint`

        Responses:
            See :meth:`check_and_dispatch`

        """
        return self.check_and_dispatch(request, port, category, endpoint)

    def post(self, port: int, endpoint: str, category: Optional[str] = None) -> Response:
        """POST response from trainer method `endpoint`

        Responses:
            See :meth:`check_and_dispatch`

        """
        return self.check_and_dispatch(request, port, category, endpoint)

    def check_and_dispatch(self, request, port: int, category: Optional[str],
                           endpoint: str) -> Response:
        """Fetch the result from trainer.

        Args:
            port: port of the trainer
            category: category of request
            endpoint: request endpoint

        Responses:
            bad params: ResponseSchema(405, "Bad Params Given", "text", "Invalid trainer {port}")
            not loaded: ResponseSchema(405, "Trainer not loaded", "text", "Trainer not loaded")
            success: ResponseSchema(200, "Successful request", "json",
                                    "See :meth:`call_trainer`: Success")

        Returns:
            See :meth:`call_trainer`

        """
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

    def call_trainer(self, request: Request, port: int, category: Optional[str] = None,
                     endpoint: str = "") -> Response:
        """Call the trainer with the request `Request` and return the response

        Args:
            port: port of the trainer
            category: category of request
            endpoint: request endpoint

        Schemas:
            class Success(BaseModel): default: Optional[Dict]

        Returns:
            Response from the :class:`Trainer` or error message if something is wrong
            with the parameters. In fact :class:`FlaskInterface` handles the HTTP layer
            wrapping around the :class:`Trainer`'s functions.

        """
        try:
            _json = _data = _files = None
            if request.json:
                _json = request.json if isinstance(request.json, dict) else json.loads(request.json)
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
            return Response(response.content, response.status_code, headers)
        except Exception as e:
            return Response(_dump([False, f"Error occured {e}"]))
