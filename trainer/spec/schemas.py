from typing import Union, List, Callable, Dict, Tuple, Optional, Any
import sys
import pathlib
from enum import Enum


class MimeTypes(str, Enum):
    text = "text/plain"
    html = "text/html"
    form = "application/x-www-form-urlencoded"
    multipart = "multipart/form-data"
    json = "application/json"
    binary = "binary"


FlaskTypes = {"string": "str",
              "int": "integer",
              "integer": "integer",
              "float": "float",
              "uuid": "uuid",
              "path": "path"}


SwaggerTypes = {bool: {"type": "boolean"},
                int: {"type": "integer"},
                float: {"type": "float"},
                pathlib.Path: {"type": "string"},
                bytes: {"type": "string", "format": "binary"}}


# class BaseSchema:
#     def __init__(self, description: str, mimetype: Union[str, MimeTypes],
#                  example: Optional[str] = None):
#         self.description = description
#         self.mimetype = mimetype
#         self.example = example
#         if self.mimetype in {MimeTypes.json, MimeTypes.multipart, MimeTypes.form}:
#             self.schema_field = self.example
#         else:
#             self.schema_field = None

# class RequestSchema:
#     def __init__(self, req_type, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.req_type = req_type


class ResponseSchema:
    def __init__(self, status_code: int, description: str,
                 mimetype: Union[str, MimeTypes],
                 example: Optional[str] = None,
                 spec: Optional[Dict] = None):
        self.status_code = status_code
        self.description = description
        self.mimetype = mimetype
        self.example = example
        if self.mimetype == MimeTypes.json:
            self.schema_field = self.example
        else:
            self.schema_field = None
        self.spec = spec

    def schema(self, spec: Optional[Dict[str, Any]] = None) ->\
            Dict[int, Dict[str, Union[str, Dict]]]:
        if self.mimetype == MimeTypes.text:
            content = self.content_text()
        elif self.mimetype == MimeTypes.json:
            if spec:
                content = self.content_json(spec)  # type: ignore
            elif self.spec:
                content = self.content_json(self.spec)  # type: ignore
            else:
                content = self.content_json({"type": "object"})
        return {self.status_code: {"description": self.description,
                                   'content': content}}

    def content_text(self) -> Dict[str, Dict]:
        if self.example:
            return {"text/plain": {"schema": {"type": "string", "example": self.example}}}
        else:
            return {"text/plain": {"schema": {"type": "string"}}}

    def content_json(self, spec: Dict[str, Any]) -> Dict[str, Dict]:
        if "type" in spec and spec["type"] == "object" and\
           "default" in spec["properties"]:
            title = spec["title"]
            spec = spec["properties"]["default"]
            spec["title"] = title
        return {"application/json": {"schema": spec}}
