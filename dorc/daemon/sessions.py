from typing import Any, Optional, Dict, Union
import traceback
from flask import request, Response
from flask.views import MethodView
from flask_login import login_required

from ..util import make_json


class Sessions(MethodView):
    "Get the sessions list"

    decorators = [login_required]

    def __init__(self, daemon):
        self.daemon = daemon

    def get(self) -> Response:
        """GET the sessions list.

        See :attr:`~daemon.Daemon.sessions_list`

        Tags:
            daemon

        Requests:
            params:
                name: Optional[str]

        Schemas:
            class Success(BaseModel):
                default: :attr:`~daemon.Daemon.sessions_list`.returns

        Responses:
            failure: ResponseSchema(404, "No such session", MimeTypes.text,
                                    "No Session with key: some_key")
            Success: ResponseSchema(200, "Check Successful", MimeTypes.json, "Success")

        """
        try:
            name = request.args.get("name")
        except Exception:
            name = None
        sess_list = self.daemon.sessions_list
        if name:
            name = name.strip()
            sessions = {k: v for k, v in sess_list.items()
                        if k.startswith(name)}
            if sessions:
                return make_json(sessions)
            else:
                return Response(f"No session found with name: {name}", 404)
        else:
            return sess_list
