from typing import Union, List, Callable, Dict, Tuple, Optional, Any
import os
import sys
import time
import shutil
import requests
import flask
from threading import Thread
from pydantic import BaseModel

from .docstring import GoogleDocstring
from .daemon import _start_daemon


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


def create_module(module_dir, module_files=[]):
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    if not os.path.exists(os.path.join(module_dir, "__init__.py")):
        with open(os.path.join(module_dir, "__init__.py"), "w") as f:
            f.write("")
    for f in module_files:
        shutil.copy(f, module_dir)


def make_daemon():
    data_dir = ".test_dir"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    port = 23232
    hostname = "127.0.0.1"
    daemon = _start_daemon(hostname, port, ".test_dir")
    host = "http://" + ":".join([hostname, str(port) + "/"])
    time.sleep(.5)
    cookies = requests.request("POST", host + "login",
                               data={"username": "admin",
                                     "password": "AdminAdmin_33"}).cookies
    return daemon, cookies


def make_interface():
    hostname = "127.0.0.1"
    port = 12321
    data_dir = ".test_dir"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    from trainer.interfaces import FlaskInterface
    iface = FlaskInterface(hostname, port, data_dir, no_start=True)
    with open("_setup.py", "rb") as f:
        f_bytes = f.read()
        status, message = iface.create_trainer(f_bytes)
    iface_thread = Thread(target=iface.start)
    create_module(os.path.abspath(os.path.join(data_dir, "global_modules")),
                  [os.path.abspath("../trainer/autoloads.py")])
    sys.path.append(os.path.abspath(data_dir))
    status, message = iface.create_trainer()
    iface_thread = Thread(target=iface.start)
    iface_thread.start()
    time.sleep(1)
    return iface


def generate_respones(func: Callable) -> Optional[List[Dict[str, Dict]]]:
    if func.__doc__ is None:
        return None
    doc = GoogleDocstring(func.__doc__)
    responses = {}
    for name, response_str in doc.responses.items():
        response_args = process_response_string(response_str)
        # TODO: name and define type for response_args to silence mypy
        responses[name] = gen_response(name, *response_args)
        if response_args[-2] in {"json", "application/json"}:
            if doc.schemas is None:
                raise AttributeError(f"Schema cannot be none for json {func}")
            else:
                schema = get_schema_var(doc.schemas[1], response_args[-1])
            if schema is None:
                raise AttributeError(f"{responses[name]} should be in schemas")
            else:
                schema_dict = schema.schema()
                schema_dict.pop("title")
                code = [*responses[name].keys()][0]
                responses[name][code]['content']['application/json'] = schema_dict
    return [*responses.values()]


def process_response_string(response_str: str) -> List[Union[int, str]]:
    ldict: Dict[str, Any] = {}
    exec("testvar = " + response_str, globals(), ldict)
    retval = ldict['testvar']
    return retval


def gen_response(name: str, code: int, desc: str, mimetype: str,
                 content, **kwargs) -> Dict[int, Dict[str, Any]]:
    mt_dict = {"json": "application/json",
               "text": "application/text",
               "html": "text/html"}
    if mimetype in mt_dict:
        mimetype = mt_dict[mimetype]
    if mimetype in {"text", "application/text"}:
        content = {"properties": {"type": "string"}}
    return {code: {"description": desc,
                   "content": {mimetype: {"schema": content}}}}


def get_doc_for_func(func: Callable):
    return None


def start_app(app_str: str) -> flask.Flask:
    ldict = {}
    exec("import " + app_str + " as app", globals(), ldict)
    


def get_redirect(redir_str: str):
    ldict: Dict[str, Any] = {}
    exec("retval = " + redir_str, globals(), ldict)
    rdict: Dict[str, str] = ldict["retval"]
    app: flask.Flask = start_app(rdict["app"])
    rdict["endpoint"]


def get_schema_var(schemas: List[str], var: str) -> Optional[BaseModel]:
    splits = "\n".join(var).split(":", 1)
    if splits[0] == "$schema":
        ldict: Dict[str, Any] = {}
        exec("\n".join(schemas), globals(), ldict)
        if var not in ldict:
            return None
        else:
            return ldict[var]
    elif splits[0] == "$returns":
        get_doc_for_func(splits[1].strip())
    elif splits[0] == "$redirect":
        get_redirect(splits[1])


def get_request_schema(endpoint: Callable) -> Optional[str]:
    return None


def get_endpoint_description(endpoint: Callable) -> Optional[str]:
    return None


def make_paths(daemon, cookies):
    paths = {}
    for rule in daemon.app.url_map.iter_rules():
        name = rule.rule
        if name in {"/check_task", "/trainer/<port>/<endpoint>"}:
            endpoint = daemon.app.view_functions[rule.endpoint]
            paths[name] = {}
            for method in rule.methods:
                if method in ["GET", "POST"]:
                    method_func = getattr(endpoint.view_class, method.lower())
                    paths[name][method.lower()] = {}
                    paths[name][method.lower()]["tags"] = name.split("/")[1]
                    # FIXME: Find a better way to generate operationId
                    paths[name][method.lower()]["operationId"] = name.split("/")[1] + "_" +\
                        rule.endpoint.strip("_")
                    request_body = {"description": get_endpoint_description(method_func),
                                    "content": {"application/json":
                                                {"schema": get_request_schema(method_func)}}}
                    paths[name][method.lower()]["requestBody"] = request_body
                    paths[name][method.lower()]['x-codegen-request-body-name'] = 'body'
                    responses = generate_respones(method_func)
                    paths[name][method.lower()]["responses"] = responses
                    # security = [{'petstore_auth': ['write:pets', 'read:pets']}]
    return paths
