from typing import Union, List, Callable, Dict, Tuple, Optional, Any
import flask
import re
import sys

from .. import daemon
from .. import trainer
from .. import interfaces

from . import docstring
from .schemas import RequestSchema, ResponseSchema
from .models import BaseModel


file_content = {'content': {'multipart/form-data':
                            {'schema':
                             {'properties':
                              {'additionalMetadata':
                               {'type': 'string',
                                'description': 'Additional data to pass to server'},
                               'file': {'type': 'string',
                                        'description': 'file to upload',
                                        'format': 'binary'}}}}}}


ref_regex = re.compile(r'(.+)(:[a-zA-Z0-9]+[\-_+:.])`(.+?)`')


def exec_and_return(response_str: str) -> Any:
    ldict: Dict[str, Any] = {}
    exec("testvar = " + response_str, globals(), ldict)
    retval = ldict['testvar']
    return retval


def get_func_for_redirect(func_name: str, redirect_from: Callable) -> Optional[Callable]:
    func_class = getattr(sys.modules[redirect_from.__module__],
                         redirect_from.__qualname__.split(".")[0])
    func = getattr(func_class, func_name)
    return func


def ref_repl(x):
    return re.sub(ref_regex, r'\3', x)


def check_for_redirects(var: str, redirect_from: Callable) ->\
        Optional[Tuple[str, List[str]]]:
    if re.match(ref_regex, var):
        var = ref_repl(var)
    func_name, attr = [x.strip() for x in var.split(":")]
    func = get_func_for_redirect(func_name, redirect_from)
    if func is None:
        return None
    else:
        if func.__doc__ is None:
            return None
        else:
            doc = docstring.GoogleDocstring(func.__doc__)
            if hasattr(doc, "schemas"):
                return doc.schemas
            else:
                return None


def get_redirects(func_name: str, attr: str,
                  redirect_from: Callable) -> Optional[BaseModel]:
    func = get_func_for_redirect(func_name, redirect_from)
    if func is None:
        return None
    else:
        if func.__doc__ is None:
            return None
        else:
            doc = docstring.GoogleDocstring(func.__doc__)
            # attr = splits[0].replace("$", "", 1)
            return getattr(doc, attr)


def generate_responses(func: Callable) -> Optional[List[Dict[str, Dict]]]:
    if func.__doc__ is None:
        return None
    doc = docstring.GoogleDocstring(func.__doc__)
    responses = {}

    def response_subroutine(name, response_str):
        response_args = exec_and_return(response_str)
        if isinstance(response_args, ResponseSchema):
            if response_args.mimetype in {"text", "text/plain"}:
                content = response_args.schema()
            elif response_args.mimetype in {"json", "application/json"}:
                sf = response_args.schema_field
                if not hasattr(doc, "schemas") or doc.schemas is None:
                    spec = check_for_redirects(sf, func)
                    var = sf.split(":")[-1].strip()
                    spec = get_schema_var(spec[1], var)
                else:
                    var = sf.split(":")[-1].strip()
                    spec = get_schema_var(doc.schemas[1], var)
                content = response_args.schema(spec.schema())
            responses[name] = content
        else:
            responses[name] = gen_response(name, *response_args)
            if response_args[-2] in {"json", "application/json"}:
                if not hasattr(doc, "schemas") or doc.schemas is None:
                    schema = check_for_redirects(response_args[-1], func)
                    var = response_args[-1].split(":")[-1]
                    schema = get_schema_var(schema[1], var)
                else:
                    var = response_args[-1].split(":")[-1]
                    schema = get_schema_var(doc.schemas[1], var)
                if schema is None:
                    raise AttributeError(f"{responses[name]} should be in schemas")
                schema_dict = schema.schema()
                schema_dict.pop("title")
                code = [*responses[name].keys()][0]
                responses[name][code]['content']['application/json'] = schema_dict

    for name, response_str in doc.responses.items():
        if name == "responses":
            response_dict = get_redirects(response_str, name, func)
            if response_dict is None:
                raise ValueError(f"Check Redirect failed for {name} {func}")
            else:
                for name, response_str in response_dict.items():
                    response_subroutine(name, response_str)
        elif name == "returns":
            # print(name, func.__qualname__)
            response_str = exec_and_return(response_str)
            response_subroutine(name, response_str)
        else:
            # print(name, func.__qualname__)
            response_subroutine(name, response_str)
    return [*responses.values()]


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


def get_schema_var(schemas: List[str], var: str) -> Optional[BaseModel]:
    ldict: Dict[str, Any] = {}
    exec("\n".join(schemas), globals(), ldict)
    if var not in ldict:
        return None
    else:
        return ldict[var]


def get_request_schema(endpoint: Callable) -> Optional[str]:
    return None


def get_endpoint_description(endpoint: Callable) -> Optional[str]:
    return None


def make_paths(app: flask.Flask) -> Dict:
    paths: Dict = {}
    for rule in app.url_map.iter_rules():
        name = rule.rule
        endpoint = app.view_functions[rule.endpoint]
        # name.split("/")[1] in {"trainer", "check_task"}:  # in {"/trainer/<port>/<endpoint>"}:
        if hasattr(endpoint, "view_class"):
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
                    responses = generate_responses(method_func)
                    paths[name][method.lower()]["responses"] = responses
                    # security = [{'petstore_auth': ['write:pets', 'read:pets']}]
    return paths



# def alt_paths():
#     val = "\n".join(getattr(doc, attr))
#     if val.startswith("$returns") or val.startswith("$redirect"):
#         return check_for_redirects(val, func)
#     else:
#         # GET SCHEMA VAR
#         import ipdb; ipdb.set_trace()
#     elif splits[0] == "$responses":
#         redirect = exec_and_return(splits[1])
#         app = apps[redirect["app"]]
#         attr = redirect["attr"]
#         matches = []
#         endpoints = getattr(app, attr)
#         import ipdb; ipdb.set_trace()
#         for ep in endpoints:
#             test = [x for x in app.url_map.iter_rules()
#                     if re.match(x._regex, "|/" + ep)]
#             matches.extend(test)
#         func_name = splits[1].strip()
#         func = get_func_for_redirect(func_name, redirect_from)
#         if func is None:
#             return None
#         else:
#             doc = docstring.GoogleDocstring(func.__doc__)
#             val = "\n".join(doc.returns)
#             if val.startswith("$returns") or val.startswith("$redirect"):
#                 return check_for_redirects(val, func, rule)
#             else:
#                 return None
#     elif splits[0] == "$schemas":
#         redirect = exec_and_return(splits[1])
#         app = apps[redirect["app"]]
#         attr = redirect["attr"]
#         matches = []
#         endpoints = getattr(app, attr)
#         import ipdb; ipdb.set_trace()
#         for ep in endpoints:
#             test = [x for x in app.url_map.iter_rules()
#                     if re.match(x._regex, "|/" + ep)]
#             matches.extend(test)
#         func_name = splits[1].strip()
#         func = get_func_for_redirect(func_name, redirect_from)
#         if func is None:
#             return None
#         else:
#             doc = docstring.GoogleDocstring(func.__doc__)
#             val = "\n".join(doc.returns)
#             if val.startswith("$returns") or val.startswith("$redirect"):
#                 return check_for_redirects(val, func, rule)
#             else:
#                 return None

# def start_app(app_str: str) -> flask.Flask:
#     ldict = {}
#     exec("import " + app_str + " as app", globals(), ldict)


# def get_redirect(redir_str: str):
#     ldict: Dict[str, Any] = {}
#     exec("retval = " + redir_str, globals(), ldict)
#     rdict: Dict[str, str] = ldict["retval"]
#     return rdict


# import importlib
# from trainer import util, docstring, api

# daemon = util.make_test_daemon()
