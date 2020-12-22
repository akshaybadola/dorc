import os
import time
import shutil
import requests
import apispec

from .daemon import _start_daemon


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


def get_endpoint_description(endpoint):
    return None


def get_schema(endpoint):
    return None


def make_schemas():
    return None


def make_paths():
    daemon, cookies = make_daemon()
    paths = {}
    for rule in daemon.app.url_map.iter_rules():
        name = rule.rule
        endpoint = rule.endpoint.strip("_")
        paths[name] = {}
        for method in rule.methods:
            paths[name][method.lower()] = {}
            paths[name][method.lower()]["tags"] = name.split("/")[1]
            paths[name][method.lower()]["operationId"] = name.split("/")[1] + "_" + endpoint
            request_body = {"description": get_endpoint_description(endpoint),
                            "content": {"application/json": {"schema": {"$ref": get_schema(endpoint)}}}}
            paths[name][method.lower()]["requestBody"] = request_body
            paths[name][method.lower()]['x-codegen-request-body-name'] = 'body'
            responses = {}
            security = [{'petstore_auth': ['write:pets', 'read:pets']}]

