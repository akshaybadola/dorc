from typing import Any, Optional, Dict, Union
from flask.views import MethodView
import flask_login
from flask_login import login_required


class CurrentUser(MethodView):
    decorators = [login_required]

    def __init__(self, daemon):
        self.daemon = daemon

    def get(self):
        """GET the name of the current user

        Request:
            RequestSchema()

        Schemas:
            class Success(BaseModel): user: str

        Responses:
            Success: [200, "Current logged in user", "json", "Success"]

        """
        return self.current_user

    @property
    def current_user(self):
        """Returns the name of the current user, in case we're logged in and username is
        not known to the client as they have refreshed and the store state is
        gone (LOL, FIXME)

        """
        return {"user": flask_login.current_user.name}
