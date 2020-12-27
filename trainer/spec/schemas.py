from typing import Union, List, Callable, Dict, Tuple, Optional, Any


class RequestSchema:
    pass


class ResponseSchema:
    def __init__(self, status_code: int, description: str, mimetype: str,
                 example: Optional[str] = None):
        self.status_code = status_code
        self.description = description
        self.mimetype = mimetype
        self.example = example
        if self.mimetype in {"json", "application/json"}:
            self.schema_field = self.example
        else:
            self.schema_field = None

    def schema(self, spec: Optional[Dict[str, Any]] = None) ->\
            Dict[int, Dict[str, Union[str, Dict]]]:
        if self.mimetype in {"text", "text/plain"}:
            content = self.content_text()
        elif self.mimetype in {"json", "application/json"}:
            content = self.content_json(spec)  # type: ignore
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
