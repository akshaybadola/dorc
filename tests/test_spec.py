import pytest
from typing import Union, List, Callable, Dict, Tuple, Optional, Any
from pydantic import BaseModel as PydanticBaseModel, Field
from dorc.spec.models import (BaseModel, ModelNoTitleNoRequiredNoPropTitle,
                              add_nullable, remove_attr, remove_prop_titles)
from dorc.spec.schemas import ResponseSchema
from dorc import daemon
from dorc import spec
from dorc.spec import docstring


class FooModel(BaseModel):
    foo: str


class BarModel(BaseModel):
    bar: Optional[str]


class YourModel(BaseModel):
    meow: int
    bleh: Union[Dict[str, Dict[str, Union[None, int, Dict]]], Dict[str, Union[None, int, Dict]]]
    model_foo: Optional[FooModel]


class BigModel(BaseModel):
    int_foo: int
    str_foo: str
    model_foo: FooModel
    opt_str_foo: Optional[str] = Field(alias='opt_str_foo_alias')
    opt_model_foo: Optional[FooModel]
    union_foo: Union[int, str]
    union_model_foo_str: Union[FooModel, str]
    union_model_foo_bar: Union[FooModel, BarModel]
    opt_union_foo: Union[None, int, str]
    opt_union_foo2: Optional[Union[int, str]]
    opt_union_model_foo_bar: Optional[Union[FooModel, BarModel]]


check_task = {'/check_task': {'get': {'tags': 'check_task',
                                      'operationId': 'check_task_check_task',
                                      'requestBody': {'description': None,
                                                      'content': {'application/json': {'schema': None}}},
                                      'responses': [{405: {'description': 'Bad Params',
                                                           'content':
                                                           {'application/text':
                                                            {'schema': {'properties':
                                                                        {'type': 'string'}}}}}},
                                                    {404: {'description': 'No such Task',
                                                           'content':
                                                           {'application/text':
                                                            {'schema': {'properties':
                                                                        {'type': 'string'}}}}}},
                                                    {200: {'description': 'Check Successful',
                                                           'content':
                                                           {'application/json':
                                                            {'type': 'object',
                                                             'properties':
                                                             {'task_id':
                                                              {'title': 'Task Id', 'type': 'integer'},
                                                              'result': {'title': 'Result', 'type': 'Boolean'},
                                                              'Message': {'title': 'Message', 'type': 'string'}},
                                                             'required': ['task_id', 'result', 'message']}}}}]}}}

example = {'/pet': {'put': {'tags': ['pet'],
                            'summary': 'Update an existing pet',
                            'operationId': 'updatePet',
                            'requestBody': {'description': 'Pet object that needs to be added to the store',
                                            'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Pet'}},
                                                        'application/xml': {'schema': {'$ref': '#/components/schemas/Pet'}}},
                                            'required': True},
                            'responses': {400: {'description': 'Invalid ID supplied', 'content': {}},
                                          404: {'description': 'Pet not found', 'content': {}},
                                          405: {'description': 'Validation exception', 'content': {}}},
                            'security': [{'petstore_auth': ['write:pets', 'read:pets']}],
                            'x-codegen-request-body-name': 'body'},
                    'post': {'tags': ['pet'],
                             'summary': 'Add a new pet to the store',
                             'operationId': 'addPet',
                             'requestBody': {'description': 'Pet object that needs to be added to the store',
                                             'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Pet'}},
                                                         'application/xml': {'schema': {'$ref': '#/components/schemas/Pet'}}},
                                             'required': True},
                             'responses': {405: {'description': 'Invalid input', 'content': {}}},
                             'security': [{'petstore_auth': ['write:pets', 'read:pets']}],
                             'x-codegen-request-body-name': 'body'}}}


best_save = """
/props/best_save:
    get:
      description: Return a property `best_save` from the trainer
      operationId: FlaskInterface__props__GET
      responses:
        200:
          content:
            application/json:
              schema:
                format: path
                nullable: true
                title: default
                type: string
          description: Success
        404:
          content:
            text/plain:
              schema:
                example: Property not found
                type: string
          description: Not found
"""


def bleh():
    """Handle a POST request for a `session_method`

    Args:
        func_name: The name of the helper method

    Schema:
        class Task(BaseModel):
            task_id: int
            message: str

    Request:
        content-type: MimeTypes.json
        body:
            session_key: str
            data: Union[:meth:`daemon.Daemon._archive_session_helper`: ArchiveSessionModel,
                        :meth:`daemon.Daemon._reinit_session_helper`: ReinitSessionModel,
                        :meth:`daemon.Daemon._clone_session_helper`: CloneSessionModel,
                        :meth:`daemon.Daemon._clone_to_helper`: CloneToServerModel,
                        Dict]
            some_other_shit: Dict

    Responses:
        invalid data: ResponseSchema(405, "Invalid Data", MimeTypes.text,
                                    "Invalid data {some: json}")
        bad params: ResponseSchema(405, "Bad Params", MimeTypes.text,
                                   "Session key not in params")
        Success: ResponseSchema(200, "Initiated Task", MimeTypes.json, "Task")
    """
    pass


def bleh_redirect() -> Union[Dict[str, Optional[List[str]]]]:
    "Doesn't have any docstring"
    pass


def bleh_annot():
    """Some random shit

    Request:
        content-type: MimeTypes.json
        body:
            some_attr: :func:`bleh_redirect`
    """
    pass


@pytest.mark.spec
def test_get_opId():
    assert spec.get_opId("/path/get_stuff", bleh_annot, None,
                         ["task_id", "op_id"], "GET",
                         "[__%M%r%n]__[_%p]__%H", {}) == 'TestSpec__get_stuff__task_id_op_id__GET'
    assert spec.get_opId("/path/get_stuff", bleh_annot, None,
                         ["task_id", "op_id"], "GET",
                         "[__%m%r%n]__[_%p]__%H", {}) == 'testSpec__get_stuff__task_id_op_id__GET'
    assert spec.get_opId("/path/get_stuff", bleh_annot, None,
                         ["task_id", "op_id"], "GET",
                         "[__%C%r%n]__[_%p]__%H", {}) == 'get_stuff__task_id_op_id__GET'
    assert spec.get_opId("/path/get_stuff", bleh_annot, None,
                         ["task_id", "op_id"], "GET",
                         "[__%C%f%r%n]__[_%p]__%H", {}) ==\
                         'bleh_annot__get_stuff__task_id_op_id__GET'
    assert spec.get_opId("/path/get_stuff", bleh_annot, None,
                         ["task_id", "op_id"], "GET",
                         "[__%C%F%r%n]__[_%p]__%H", {}) ==\
                         'Bleh_annot__get_stuff__task_id_op_id__GET'
    assert spec.get_opId("/path/create_session", daemon.Daemon.create_session, None,
                         [], "GET", "[__%M%C%f%r%n]__[_%p]__%H", {}) ==\
                         'Dorc.Daemon__Daemon__create_session____GET'
    assert spec.get_opId("/path/create_session", daemon.Daemon.create_session, None,
                         [], "GET", "[_%C%f%r]%H", {}) ==\
                         'Daemon_create_sessionGET'


@pytest.mark.spec
def test_types():
    class AnyModel(BaseModel):
        any: Any
        list_any: List[Any]
        object_str_any: Dict[str, Any]
        object_any_any: Dict[Any, Any]
        union_any: Union[str, Any]
        object: Dict
        optional_object: Optional[Dict]
        list_object: List[Dict]

    schema = AnyModel.schema()
    assert schema["properties"]["any"]["type"] == "object"
    assert schema["properties"]["any"]["nullable"]
    assert schema["properties"]["list_any"]["type"] == "array"
    assert schema["properties"]["list_any"]["items"]["type"] == "object"
    assert schema["properties"]["list_any"]["items"]["nullable"]
    assert schema["properties"]["object_str_any"]["type"] == "object"
    assert "nullable" not in schema["properties"]["object_str_any"]
    assert schema["properties"]["object_any_any"]["type"] == "object"
    assert "nullable" not in schema["properties"]["object_any_any"]
    assert schema["properties"]["object"]["type"] == "object"
    assert "nullable" not in schema["properties"]["object"]
    assert schema["properties"]["optional_object"]["type"] == "object"
    assert schema["properties"]["optional_object"]["nullable"]
    assert schema["properties"]["list_object"]["type"] == "array"
    assert schema["properties"]["list_object"]["items"]["type"] == "object"
    assert "nullable" not in schema["properties"]["list_object"]["items"]


@pytest.mark.spec
def test_get_requests():
    doc = docstring.GoogleDocstring(bleh.__doc__)
    assert not hasattr(doc, "params")
    assert hasattr(doc, "requests")
    assert hasattr(doc, "schemas")


@pytest.mark.spec
def test_check_for_redirects():
    check_str = ':meth:`daemon.Daemon._reinit_session_helper`: ReinitSessionModel'
    func, attr = spec.check_for_redirects(check_str, bleh)
    assert func == daemon.Daemon._reinit_session_helper
    assert attr == "ReinitSessionModel"
    check_str = ':func:`test_check_for_redirects`'
    func, attr = spec.check_for_redirects(check_str, bleh)
    assert func == test_check_for_redirects
    assert attr == "return"


@pytest.mark.spec
def test_get_request_body():
    request = spec.get_requests(bleh, "GET", "")
    assert request['content-type'] == 'MimeTypes.json'
    assert "body" in request
    body = spec.join_subsection(request["body"])
    assert len(body) == 3
    splits = [x.split(":", 1)[0].strip() for x in body]
    assert 'session_key' in splits
    assert 'data' in splits
    assert 'some_other_shit' in splits
    body = spec.get_request_body(request["body"], bleh)
    props = body['properties']
    assert 'session_key' in props
    assert 'data' in props
    assert 'some_other_shit' in props
    models_list = ["ArchiveSessionModel",
                   "ReinitSessionModel",
                   "CloneSessionModel",
                   "CloneToServerModel"]
    assert 'anyOf' in props['data']
    models_in_props = [[*x.values()][0] for x in props['data']['anyOf']]
    for model_name in models_list:
        any(model_name in x for x in models_in_props)


@pytest.mark.spec
def test_get_request_body_from_annotations():
    request = spec.get_requests(bleh_annot, "GET", "")
    assert request['content-type'] == 'MimeTypes.json'
    assert "body" in request
    body = spec.get_request_body(request["body"], bleh_annot)
    assert "some_attr" in body["properties"]
    # TODO: Check for attributes in body by redirect to '$ref'


@pytest.mark.spec
def test_fix_nullable():
    class YourModel(BaseModel):
        a_int: int
        n_a_int: Optional[int]
        str_int_union: Union[int, Optional[str]]
        obj_int_union: Union[int, Dict[str, Optional[int]]]

    schema = YourModel.schema()
    props = schema['properties']
    assert "a_int" in props
    assert props["a_int"]["type"] == "integer"
    assert "nullable" in props["n_a_int"]
    assert isinstance(props["str_int_union"]["anyOf"], list)
    assert "nullable" in props["str_int_union"]
    assert isinstance(props["obj_int_union"]["anyOf"], list)
    assert "nullable" not in props["obj_int_union"]
    assert "nullable" not in props["obj_int_union"]["anyOf"][-1]
    assert "nullable" in props["obj_int_union"]["anyOf"][-1]["additionalProperties"]


@pytest.mark.spec
def test_remove_prop_titles():
    class YourModel(PydanticBaseModel):
        a_int: int
        n_a_int: Optional[int]
        str_int_union: Union[int, Optional[str]]
        obj_int_union: Union[int, Dict[str, Optional[int]]]
    assert YourModel.schema()["type"] == "object"
    props = YourModel.schema()["properties"]
    for v in props.values():
        assert "title" in v

    class YourModel(BaseModel):
        a_int: int
        n_a_int: Optional[int]
        str_int_union: Union[int, Optional[str]]
        obj_int_union: Union[int, Dict[str, Optional[int]]]

        class Config:
            arbitrary_types_allowed = True

            @staticmethod
            def schema_extra(schema, model):
                remove_prop_titles(schema, model)

    assert YourModel.schema()["type"] == "object"
    props = YourModel.schema()["properties"]
    for v in props.values():
        assert "title" not in v


@pytest.mark.spec
def test_remove_attr():
    class Simple(PydanticBaseModel):
        pass
    assert "description" not in Simple.schema()

    class Simple(PydanticBaseModel):
        "Simple description"
        pass
    assert "description" in Simple.schema()

    class Simple(PydanticBaseModel):
        "Simple description"
        class Config:
            arbitrary_types_allowed = True

            @staticmethod
            def schema_extra(schema, model):
                remove_attr(schema, model, "description")
    assert "description" not in Simple.schema()
    assert "title" in Simple.schema()

    class Simple(PydanticBaseModel):
        "Simple description"

        class Config:
            arbitrary_types_allowed = True

            @staticmethod
            def schema_extra(schema, model):
                remove_attr(schema, model, "title")
    assert "description" in Simple.schema()
    assert "title" not in Simple.schema()


@pytest.mark.spec
def test_response_schema():
    class Response(ModelNoTitleNoRequiredNoPropTitle):
        "Simple Description"
        default: bytes
    schema = Response.schema()
    assert "description" in schema
    assert "title" not in schema
    assert schema["type"] == "object"
    assert all("title" not in x for x in schema["properties"].values())


@pytest.mark.spec
def test_response_model_binary():
    class Response(ModelNoTitleNoRequiredNoPropTitle):
        "Simple Description"
        default: bytes
    schema = ResponseSchema(200, "Successful", "binary", "Success")
    schema.spec = Response.schema()
    assert schema.schema() == {200: {"description": "Successful",
                                     "content": {'application/octet-stream':
                                                 {'schema':
                                                  {'type': 'string', 'format': 'binary'}}}}}

    class Response(ModelNoTitleNoRequiredNoPropTitle):
        "Simple Description"
        application_pdf: bytes
    schema.spec = Response.schema()
    assert schema.schema() == {200: {"description": "Successful",
                                     "content": {'application/pdf':
                                                 {'schema':
                                                  {'type': 'string', 'format': 'binary'}}}}}

    class Response(ModelNoTitleNoRequiredNoPropTitle):
        "Simple Description"
        name: str
        stuff: List[str]
        file: bytes
    schema.spec = Response.schema()
    assert schema.schema() == {200: {"description": "Successful",
                                     "content": {'application/json':
                                                 {'schema':
                                                  {'description': 'Simple Description',
                                                   'type': 'object',
                                                   'properties':
                                                   {'name': {'type': 'string'},
                                                    'stuff': {'type': 'array', 'items':
                                                              {'type': 'string'}},
                                                    'file': {'type': 'string', 'format': 'byte'}}}}}}}
