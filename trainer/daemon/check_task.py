from typing import Any, Optional, Dict, Union
import traceback
from flask import request, Response
from flask.views import MethodView
from flask_login import login_required


class CheckTask(MethodView):
    decorators = [login_required]

    def __init__(self, daemon):
        self.daemon = daemon

    def get(self):
        """GET the status of a `task_id`

        Requests:
            params:
                task_id: int

        Schemas:
            class Success(BaseModel):
                task_id: int
                result: bool
                message: str

        Responses:
            bad_params: ResponseSchema(405, "Bad Params", MimeTypes.text, "task_id not given")
            No such task: ResponseSchema(404, "No such Task", MimeTypes.text, "No such task 4")
            Success: ResponseSchema(200, "Check Successful", MimeTypes.json, "Success")

        """
        try:
            task_id = int(request.args.get("task_id").strip())
            return self.check_task(task_id)
        except Exception as e:
            return self.bad_params(f"Bad params {e}" + "\n" + traceback.format_exc())
        if task_id not in self.__task_ids:
            return Response(f"No such task: {task_id}", 404)

    def check_task(self, task_id: int) -> Dict[str, Union[str, bool, int]]:
        """Check and return the status of a task submitted earlier.

        Args:
            task_id: ID of the the task to check

        Returns:
            The status of the task with `task_id` or error message if it doesn't exist.

        """
        result = self.daemon._check_result(task_id)
        if result is None:
            return {"task_id": task_id, "result": 0, "message": "Not yet processed"}
        else:
            if len(result) == 2:
                return {"task_id": result[0], "result": True, "message": "Successful"}
            elif len(result) == 3 and result[1]:
                return {"task_id": result[0], "result": True, "message": result[2]}
            elif len(result) == 3 and not result[1]:
                return {"task_id": result[0], "result": False, "message": result[2]}
            else:
                return result
