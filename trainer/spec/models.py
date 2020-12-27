from pydantic import BaseModel as PydanticBaseModel


class BaseModel(PydanticBaseModel):
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
                        value["properties"] = value["additionalProperties"]
                        value.pop("additionalProperties")
                        subroutine(field.sub_fields[0], value["properties"])
                    elif "anyOf" in value:
                        for i, sub_field in enumerate(field.sub_fields):
                            subroutine(sub_field, value["anyOf"][i])
            for prop, value in schema.get('properties', {}).items():
                field = [x for x in model.__fields__.values() if x.alias == prop][0]
                subroutine(field, value)
