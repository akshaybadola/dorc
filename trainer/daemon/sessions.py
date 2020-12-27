from typing import Any, Optional, Dict, Union
import traceback
from flask import request, Response
from flask.views import MethodView
from flask_login import login_required


class Sessions(MethodView):
    decorators = [login_required]

    def __init__(self, daemon):
        self.daemon = daemon

    def get(self):
        """GET the sessions list

        Schemas:
            class Success(BaseModel):
                default: daemon.Daemon._sessions_list.fget.__annotations__["return"]

        Responses:
            failure: ResponseSchema(404, "No session", "text", "No Session with key: some_key")
            Success: ResponseSchema(200, "Check Successful", "json", "Success")

        """
        try:
            name = request.args.get("name")
        except Exception:
            name = None
        sess_list = self._sessions_list
        if name:
            name = name.strip()
            sessions = {k: v for k, v in sess_list.items()
                        if k.startswith(name)}
            if sessions:
                return sessions
            else:
                return f"No session found with name: {name}"
        else:
            return sess_list
