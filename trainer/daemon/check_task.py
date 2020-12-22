import json
import requests
from flask import request, Response
from flask.views import MethodView
from flask_login import login_required

from ..util import _dump


class CheckTask(MethodView):
    decorators = [login_required]

    def __init__(self, daemon):
        self.daemon = daemon
        self.requests = {"bad params": {405: {"description": "bad params",
                                              "content": {"application/text":
                                                          "Unloaded or invalid trainer {port}"}}},
                         "success": {200: {"description": "Successful request",
                                           "content": {"application/json": {"schema": "bleh"}}}}}

    def get(self):
        try:
            task_id = int(request.args.get("task_id").strip())
            return check_task(task_id)
        except Exception as e:
            return bad_params(f"Bad params {e}" + "\n" + traceback.format_exc())
            # return bad_params(f"Bad params {e}" + "\n" + traceback.format_exc())

    def bad_params(self, message):
            return _dump([False, message])

    def check_task(self, task_id: int):
        """Check and return the status of a task submitted earlier.

        Args:
            args: task_id

        Returns:
            class Type(BaseModel): task_id: int; result: bool; message: str
        Requests:
            bad_params: [405, "bad params", "text", "Unloaded or invalid trainer {port}"]

        """
        try:
            task_id = int(request.args.get("task_id").strip())
        except Exception as e:
            return _dump([False, f"Bad params {e}" + "\n" + traceback.format_exc()])
        if task_id not in self.__task_ids:
            return _dump([False, f"No such task: {task_id}"])
        else:
            result = self._check_result(task_id)
        if result is None:
            return _dump([True, {"task_id": task_id, "result": 0,
                                 "message": "Not yet processed"}])
        else:
            if len(result) == 2:
                self._logw(f"Result of length 2 for check_task {result}")
                return _dump([True, {"task_id": result[0], "result": True,
                                     "message": "Successful"}])
            elif len(result) == 3 and result[1]:
                return _dump([True, {"task_id": result[0], "result": True,
                                     "message": result[2]}])
            elif len(result) == 3 and not result[1]:
                return _dump([True, {"task_id": result[0], "result": False,
                                     "message": result[2]}])
            else:
                return _dump([True, result])
