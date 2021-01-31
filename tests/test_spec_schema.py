import pytest
from typing import Union, List, Callable, Dict, Tuple, Optional, Any
from dorc.spec.models import BaseModel


def dget(obj, *args):
    if args:
        return dget(obj.get(args[0]), *args[1:])
    else:
        return obj


def test_callable_no_args():
    class NoArgs(BaseModel):
        meh: int
        func: Callable
    schema = NoArgs.schema()
    assert dget(schema, "properties", "func", "additionalProperties", "args", "type") == "object"
    assert dget(schema, "properties", "func", "additionalProperties", "retval", "nullable")


def test_callable_empty_args():
    class EmptyArgs(BaseModel):
        func: Callable[[], None]
    schema = EmptyArgs.schema()
    assert dget(schema, "properties", "func", "additionalProperties", "args", "type") == "array"
    assert dget(schema, "properties", "func", "additionalProperties", "retval", "nullable")


def test_callable_any_args():
    class AnyArgs(BaseModel):
        func: Callable[..., None]
    schema = AnyArgs.schema()
    assert dget(schema, "properties", "func", "additionalProperties", "args", "type") == "object"
    assert dget(schema, "properties", "func", "additionalProperties", "retval", "nullable")


def test_callable_simple_args():
    class SimpleArgs(BaseModel):
        func: Callable[[int, int], None]
    schema = SimpleArgs.schema()
    assert dget(schema, "properties", "func", "additionalProperties", "args", "type") == "object"
    assert dget(schema, "properties", "func",
                "additionalProperties", "args", "additionalProperties").keys() == {0, 1}
    assert dget(schema, "properties", "func", "additionalProperties", "retval", "nullable")


def test_callable_recurse_args():
    class RecurseArgs(BaseModel):
        func: Callable[[int, int, Callable[[int], None]], None]
    schema = RecurseArgs.schema()
    assert dget(schema, "properties", "func", "additionalProperties", "args", "type") == "object"
    assert dget(schema, "properties", "func", "additionalProperties", "retval", "nullable")
    assert dget(schema, "properties", "func",
                "additionalProperties", "args", "additionalProperties").keys() == {0, 1, 2}
    func_arg = dget(schema, "properties", "func",
                    "additionalProperties", "args", "additionalProperties")[2]
    assert "args" in func_arg["additionalProperties"]
    assert dget(func_arg, "additionalProperties", "args", "additionalProperties",
                0, "type") == "integer"
    assert dget(func_arg, "additionalProperties", "retval", "nullable")


def test_nullable_in_object():
    class NullableInObject(BaseModel):
        func: Dict[str, Dict[str, Union[List[Optional[int]], int, bool, None]]]
    schema = NullableInObject.schema()
    assert "properties" in schema
    union = dget(schema, "properties", "func", "additionalProperties",
                 "additionalProperties", "anyOf")
    assert "nullable" in [*filter(lambda x: x["type"] == "array", union)][0]["items"]


def test_nullable_in_array():
    class NullableInArray(BaseModel):
        func: Dict[str, Union[List[Union[int, str, None]], int, bool]]
    schema = NullableInArray.schema()
    assert len(dget(schema, "properties", "func", "additionalProperties", "anyOf")) == 3
    bleh = dget(schema, "properties", "func", "additionalProperties", "anyOf")
    assert [*filter(lambda x: x["type"] == "array", bleh)][0]["items"]["nullable"]
