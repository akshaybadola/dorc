from typing import Optional, Any, Dict
from pydantic import BaseModel as PydanticBaseModel
from pydantic.fields import ModelField


def add_nullable(schema, model):
    """Add the property `nullable` to OpenAPI spec.

    For the pydantic :class:`~pydantic.BaseModel`, patch the schema generation to
    include nullable for attributes which can be null.

    Used in :func:`schema_extra` of :class:`~pydantic.BaseModel.Config`

    Args:
        schema: A dictionary of kind :class:`pydantic.BaseModel.schema`
        model: The pydantic model

    """

    def add_nullable_subroutine(field: ModelField, value: Any):
        if field.allow_none:
            if "$ref" in value:
                if issubclass(field.type_, PydanticBaseModel):
                    value['title'] = field.type_.__config__.title or field.type_.__name__
                value['anyOf'] = [{'$ref': value.pop('$ref')}]
            value["nullable"] = True
        if field.sub_fields:
            if "type" in value and value["type"] == "object" and\
               "additionalProperties" in value:
                add_nullable_subroutine(field.sub_fields[0], value["additionalProperties"])
            elif "anyOf" in value:
                for i, sub_field in enumerate(field.sub_fields):
                    add_nullable_subroutine(sub_field, value["anyOf"][i])
    for prop, value in schema.get('properties', {}).items():
        field = [x for x in model.__fields__.values() if x.alias == prop][0]
        add_nullable_subroutine(field, value)


def add_required(schema, model):
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


# class BaseModel(PydanticBaseModel):
#     """An extension of :class:`pydantic.BaseModel` with correct `nullable` property.

#     There's still a bug here where `nullable` doesn't appear inside anyOf
#     fields. Need to check.

#     """
#     class Config:
#         title = None
#         arbitrary_types_allowed = True

#         @staticmethod
#         def schema_extra(schema, model):
#             def subroutine(field, value):
#                 if field.allow_none:
#                     if "$ref" in value:
#                         if issubclass(field.type_, PydanticBaseModel):
#                             value['title'] = field.type_.__config__.title or field.type_.__name__
#                         value['anyOf'] = [{'$ref': value.pop('$ref')}]
#                     value["nullable"] = True
#                 if field.sub_fields:
#                     if "type" in value and value["type"] == "object" and\
#                        "additionalProperties" in value:
#                         subroutine(field.sub_fields[0], value["additionalProperties"])
#                     elif "anyOf" in value:
#                         for i, sub_field in enumerate(field.sub_fields):
#                             subroutine(sub_field, value["anyOf"][i])
#             for prop, value in schema.get('properties', {}).items():
#                 field = [x for x in model.__fields__.values() if x.alias == prop][0]
#                 subroutine(field, value)


# class ParamsModel(PydanticBaseModel):
#     """This is actually a ParamsModel.

#     Perhaps ResponseModel can be changed like this.

#     """
#     class Config:
#         @staticmethod
#         def schema_extra(schema, model):
#             def subroutine(field, value):
#                 if field.allow_none:
#                     if "$ref" in value:
#                         if issubclass(field.type_, PydanticBaseModel):
#                             value['title'] = field.type_.__config__.title or field.type_.__name__
#                         value['anyOf'] = [{'$ref': value.pop('$ref')}]
#                     value["required"] = False
#                 else:
#                     value["required"] = True
#                 if field.sub_fields:
#                     if "type" in value and value["type"] == "object" and\
#                        "additionalProperties" in value:
#                         subroutine(field.sub_fields[0], value["additionalProperties"])
#                     elif "anyOf" in value:
#                         for i, sub_field in enumerate(field.sub_fields):
#                             subroutine(sub_field, value["anyOf"][i])
#             for prop, value in schema.get('properties', {}).items():
#                 field = [x for x in model.__fields__.values() if x.alias == prop][0]
#                 subroutine(field, value)


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
