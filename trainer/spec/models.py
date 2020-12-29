from typing import Optional, Any
from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
    """An extension of :class:`pydantic.BaseModel` with correct `nullable` property.

    There's still a bug here where `nullable` doesn't appear inside anyOf
    fields. Need to check.

    """
    class Config:
        @staticmethod
        def schema_extra(schema, model):
            def subroutine(field, value):
                if field.allow_none:
                    if "$ref" in value:
                        if issubclass(field.type_, PydanticBaseModel):
                            value['title'] = field.type_.__config__.title or field.type_.__name__
                        value['anyOf'] = [{'$ref': value.pop('$ref')}]
                    value["nullable"] = True
                if field.sub_fields:
                    if "type" in value and value["type"] == "object" and\
                       "additionalProperties" in value:
                        subroutine(field.sub_fields[0], value["additionalProperties"])
                    elif "anyOf" in value:
                        for i, sub_field in enumerate(field.sub_fields):
                            subroutine(sub_field, value["anyOf"][i])
            for prop, value in schema.get('properties', {}).items():
                field = [x for x in model.__fields__.values() if x.alias == prop][0]
                subroutine(field, value)


class ParamsModel(PydanticBaseModel):
    """This is actually a ParamsModel.

    Perhaps ResponseModel can be changed like this.

    """
    class Config:
        @staticmethod
        def schema_extra(schema, model):
            def subroutine(field, value):
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
                        subroutine(field.sub_fields[0], value["additionalProperties"])
                    elif "anyOf" in value:
                        for i, sub_field in enumerate(field.sub_fields):
                            subroutine(sub_field, value["anyOf"][i])
            for prop, value in schema.get('properties', {}).items():
                field = [x for x in model.__fields__.values() if x.alias == prop][0]
                subroutine(field, value)


class DefaultModel(BaseModel):
    default: Optional[Any]
