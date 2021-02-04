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


def recurse_dict(jdict: Dict[str, Any], subs: Tuple[str, str, str]) -> Dict[str, Any]:
    key, sub, repl = subs
    for k, v in jdict.items():
        if k == key and v == sub:
            jdict[k] = repl
        if isinstance(v, dict):
            jdict[k] = recurse_dict(v, subs)
    return jdict


class ResponseSchema:
    def __init__(self, status_code: int, description: str,
                 mimetype: Union[str, MimeTypes],
                 example: Optional[str] = None,
                 spec: Optional[Dict] = None):
        self.status_code = status_code
        self.description = description
        self.mimetype = mimetype
        self.example = example
        if self.mimetype == MimeTypes.json or self.mimetype == MimeTypes.binary:
            self.schema_field = self.example
        else:
            self.schema_field = None
        self.spec = spec

    def schema(self, spec: Optional[Dict[str, Any]] = None) ->\
            Dict[int, Dict[str, Union[str, Dict]]]:
        """Return schema for self.

        Args:
            spec: The OpenAPI specification :class:`dict`

        `spec` for mimetype `text/plain` is generated with :meth:`content_text`.
        For form ans JSON objects, the spec field must be provided.

        """
        if self.mimetype == MimeTypes.text:
            content = self.content_text()
        elif self.mimetype == MimeTypes.json:
            if spec:
                content = self.content_json(spec)  # type: ignore
            elif self.spec:
                content = self.content_json(self.spec)  # type: ignore
            else:
                content = self.content_json({"type": "object"})
        elif self.mimetype == MimeTypes.binary:
            spec = spec or self.spec
            if spec:
                content = self.content_binary(spec)
        return {self.status_code: {"description": self.description,
                                   'content': content}}

    def content_binary(self, spec):
        if len(spec["properties"]) == 1:
            if set(spec["properties"].keys()) == {"default"}:
                return {"application/octet-stream":
                        {"schema":
                         spec["properties"]["default"]}}
            else:
                key, val = [*spec["properties"].items()][0]
                return {key.replace("_", "/"): {"schema": val}}
        else:
            spec = recurse_dict(spec, ("format", "binary", "byte"))
            return self.content_json(spec)

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
