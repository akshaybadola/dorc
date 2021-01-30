import os
import shutil
import traceback

from flask.views import MethodView
from flask import request, Response


class ConfigFile(MethodView):
    def __init__(self, iface):
        self.iface = iface

    def get(self):
        """GET the config file for the trainer.

        Schemas:
            class Success(BaseModel):
                default: bytes

        Responses:
            not found: ResponseSchema(404, "Not Found", MimeTypes.text, "File not found")
            error: ResponseSchema(400, "Error occured", MimeTypes.text,
                                      "Error reading file on disk")
            Success: ResponseSchema(200, "Successful", MimeTypes.binary, "Success")

        """
        config_file = os.path.join(self.iface.data_dir, "session_config.py")
        if not os.path.exists(config_file):
            config_file = os.path.join(config_file[:-3], "__init__.py")
        if os.path.exists(config_file):
            try:
                with open(config_file, "rb") as f:
                    config = f.read()
                return config
            except Exception as e:
                return Response(f"Error occured {e}" + "\n" + traceback.format_exc(), 400)
        else:
            return Response(f"File not found", 404)

    def post(self):
        """Update the config file on disk.

        Requests:
            content-type: MimeTypes.multipart
            body:
                file: bytes

        Responses:
            bad params: ResponseSchema(405, "File not sent", MimeTypes.text,
                                       "File not sent in request")
            error: ResponseSchema(400, "Error occured", MimeTypes.text,
                                       "Error occured while writing file")
            Success: ResponseSchema(200, "Successful", MimeTypes.text, "Updated config")

        """
        config_file = os.path.join(self.iface.data_dir, "session_config.py")
        if not os.path.exists(config_file):
            config_file = os.path.join(config_file[:-3], "__init__.py")
        try:
            if not request.files:
                return Response("File not sent with request", status=405)
            else:
                file_bytes = request.files["file"].read()
                shutil.copy(config_file, config_file + ".bak")
                with open(config_file, "w") as f:
                    f.write(file_bytes.decode("utf-8"))
                return self.iface._logi("Updated Config")
        except Exception as e:
            shutil.copy(config_file + ".bak", config_file)
            return self.iface._loge(f"Error updating config {e}. Reverted to earlier." +
                                    "\n" + traceback.format_exc())
