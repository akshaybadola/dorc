from typing import Optional, Any, Dict, Callable
from pydantic import BaseModel as PydanticBaseModel
from pydantic.fields import ModelField


def add_nullable(schema: Dict[str, Any], model: PydanticBaseModel) -> None:
    """Add the property `nullable` to the schema.

    For the pydantic :class:`~pydantic.BaseModel`, patch the schema generation to
    include nullable for attributes which can be null.

    Used in :func:`schema_extra` of :class:`~pydantic.BaseModel.Config`

    Args:
        schema: A dictionary of kind :class:`pydantic.BaseModel.schema`
        model: The pydantic model

    """
    def add_nullable_subroutine(field: ModelField, value: Any,
                                process_all: bool = False) -> None:
        if "type" in value and value["type"] == "null":
            value.pop("type")
        if field.allow_none:
            if "$ref" in value:
                if issubclass(field.type_, PydanticBaseModel):
                    value['title'] = field.type_.__config__.title or field.type_.__name__
                value['anyOf'] = [{'$ref': value.pop('$ref')}]
            value["nullable"] = True
        if field.sub_fields:
            if "type" in value and value["type"] == "object" and\
               "additionalProperties" in value:
                if process_all:
                    for i, s in enumerate(field.sub_fields):
                        add_nullable_subroutine(s, value["additionalProperties"][i], process_all)
                else:
                    # NOTE: we only check the first one for the dict as it has only one
                    add_nullable_subroutine(field.sub_fields[0], value["additionalProperties"],
                                            process_all)
            elif "type" in value and value["type"] == "object" and\
                 "properties" in value:
                if process_all:
                    for i, s in enumerate(field.sub_fields):
                        add_nullable_subroutine(s, value["properties"][i], process_all)
                else:
                    # NOTE: we only check the first one for the dict as it has only one
                    add_nullable_subroutine(field.sub_fields[0], value["properties"],
                                            process_all)
            elif "anyOf" in value:
                for i, sub_field in enumerate(field.sub_fields):
                    add_nullable_subroutine(sub_field, value["anyOf"][i],
                                            process_all)
            elif "items" in value:
                add_nullable_subroutine(field.sub_fields[0], value["items"],
                                        process_all)
        # No args, i.e., []
        if field.args_field and isinstance(field.args_field.sub_fields, tuple) and\
           not field.args_field.sub_fields:  # empty args list
            value["properties"].pop("args")
        # Ellipses or [Any]
        elif field.args_field and field.args_field.sub_fields is None:
            value["properties"]["args"] = {"type": "object"}
        elif field.args_field is not None:
            add_nullable_subroutine(field.args_field, value["properties"]["args"], True)
        if field.ret_field is not None:
            add_nullable_subroutine(field.ret_field, value["properties"]["retval"], True)
    for prop, value in schema.get('properties', {}).items():
        field = [x for x in model.__fields__.values() if x.alias == prop][0]
        pa = 'x-type' in value and value['x-type'] == "function"
        add_nullable_subroutine(field, value, pa)


def add_required(schema, model):
    """Add the property `required` to the schema.

    For the pydantic :class:`~pydantic.BaseModel`, patch the schema generation to
    include a `required` field in case a value is not `Optional`.

    Used in :func:`schema_extra` of :class:`~pydantic.BaseModel.Config` for
    :class:`ParamsModel`

    Args:
        schema: A dictionary of kind :class:`pydantic.BaseModel.schema`
        model: The pydantic model

    """
    def add_required_subroutine(field, value):
        if field.allow_none:
            if "$ref" in value:
                if issubclass(field.type_, PydanticBaseModel):
                    value['title'] = field.type_.__config__.title or field.type_.__name__
                value['anyOf'] = [{'$ref': value.pop('$ref')}]
            value["required"] = False
        else:
            value["required"] = True
        if field.sub_fields:
            if "type" in value and value["type"] == "object" and\
               "additionalProperties" in value:
                add_required_subroutine(field.sub_fields[0], value["additionalProperties"])
            elif "anyOf" in value:
                for i, sub_field in enumerate(field.sub_fields):
                    add_required_subroutine(sub_field, value["anyOf"][i])
    for prop, value in schema.get('properties', {}).items():
        field = [x for x in model.__fields__.values() if x.alias == prop][0]
        add_required_subroutine(field, value)


def remove_attr(schema: dict, model: PydanticBaseModel, attr: str):
    """Remove the specified attribute `attr` from the schema.

    Args:
        schema: A dictionary of kind :class:`pydantic.BaseModel.schema`
        model: The pydantic model
        attr: The attribute to remove

    Example:
        remove_attr(schema, model, "title")

    """
    if attr in schema:
        schema.pop(attr)


def remove_prop_titles(schema, model):
    """Remove the `title` from properties in the objects inside schema

    Args:
        schema: A dictionary of kind :class:`pydantic.BaseModel.schema`
        model: The pydantic model
        attr: The attribute to remove

    Example:
        remove_attr(schema, model, "title")

    """

    for prop in schema.get('properties', {}).values():
        prop.pop('title', None)


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: PydanticBaseModel) -> None:
            add_nullable(schema, model)


class ParamsModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: PydanticBaseModel) -> None:
            add_required(schema, model)


class ModelNoTitleNoRequiredNoPropTitle(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

        @staticmethod
        def schema_extra(schema: Dict[str, Any], model: PydanticBaseModel) -> None:
            add_nullable(schema, model)
            remove_prop_titles(schema, model)
            remove_attr(schema, model, "title")
            remove_attr(schema, model, "required")


class TextModel(BaseModel):
    default: str


class DefaultModel(BaseModel):
    default: Optional[Any]


class FunctionSignature(BaseModel):
    args: Dict[str, Any]
    retval: Optional[Any]


class FunctionSpec(BaseModel):
    name: str
    signature: FunctionSignature


class Function(BaseModel):
    name: str
    path: Optional[str]
    params: Optional[Dict[str, Any]]
    source: Optional[str]
    doc: Optional[str]
    callable: Callable

    def get_function(self):
        return self.callable(**self.params)


class ModuleModel(BaseModel):
    name: str
    path: Optional[str]
    source: Optional[str]
    doc: Optional[str]


class Expression(BaseModel):
    name: str
    expr: str
