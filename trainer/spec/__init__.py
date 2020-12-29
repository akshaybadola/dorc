import typing
from typing import Union, List, Callable, Dict, Tuple, Optional, Any
import werkzeug
import functools
import flask
import re
import sys


from .. import daemon
from .. import trainer
from .. import interfaces

from . import docstring
from .schemas import ResponseSchema, MimeTypes, MimeTypes as mt
from .models import BaseModel, ParamsModel, DefaultModel


try:
    from types import NoneType
except Exception:
    NoneType = type(None)


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


def ref_repl(x: str) -> str:
    """Replace any reference markup from docstring with empty string.
    Uses :attr:`ref_regex`

    Args:
        x: String on which to do replacement

    Returns:
        The replaced string

    """
    return re.sub(ref_regex, r'\3', x)


def exec_and_return(exec_str: str) -> Any:
    """Execute the exec_str with :meth:`exec` and return the value

    Args:
        exec_str: The string to execute

    Returns:
        The value in the `exec_str`

    """
    ldict: Dict[str, Any] = {}
    exec("testvar = " + exec_str, globals(), ldict)
    retval = ldict['testvar']
    return retval


def resolve_partials(func: Callable) -> Callable:
    """Resolve partial indirections to get to the first function.
    Useful when the docstring of original function is required.

    Args:
        func: A :class:`functools.partial` function

    Returns:
        The original function if it's a partial function, else same function

    """
    while isinstance(func, functools.partial):
        func = func.func
    return func


def get_func_for_redirect(func_name: str, redirect_from: Callable) -> Optional[Callable]:
    """Get the function with name `func_name` from context of `redirect_from`.

    The module of `redirect_from` is searched for `func_name`

    Args:
        func_name: Name of the function to search
        redirect_from: Function from which we begin searching

    Returns:
        The function with `func_name` if found else None.

    """
    try:
        func = exec_and_return(func_name)
        return func
    except Exception:
        pass
    try:
        func = exec_and_return(".".join([redirect_from.__module__, func_name]))
        return func
    except Exception:
        pass
    try:
        func_class = getattr(sys.modules[redirect_from.__module__],
                             redirect_from.__qualname__.split(".")[0])
        func = getattr(func_class, func_name)
        return func
    except Exception:
        return None


def check_for_redirects(var: str, redirect_from: Callable) ->\
        Optional[Tuple[str, List[str]]]:
    """Check for indirections in the given `var`.

    Indirections can lead to jumps across documentations to avoid repitition.

    Args:
        var: Part of the docstring to process
        redirect_from: Current function from which it's extracted

    Returns:
        A schema variable or None if none found after redirects.

    """
    if re.match(ref_regex, var):
        var = ref_repl(var)
    if var.startswith("~"):
        var = var[1:]
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
    """Get an attribute `attr` from function `func_name` from context of `redirect_from`

    Like :func:`check_for_redirects` but instead of checking whether current
    function's docstring contains `schema` or not we check for attribute `attr`.
    CHECK: How are the two different and where are they used

    Args:
        func_name: Function name from which attribute will be extracted.
        attr: Name of the attribute to extract.
        redirect_from: Current function from which it's extracted

    Returns:
        The attribute from the docstring of the function.

    """
    func = get_func_for_redirect(func_name, redirect_from)
    if func is None:
        return None
    else:
        if func.__doc__ is None:
            return None
        else:
            doc = docstring.GoogleDocstring(func.__doc__)
            return getattr(doc, attr)


def generate_responses(func: Callable) -> Dict[int, Dict]:
    """Generate OpenAPI compliant responses from a given `func`.

    `func` would necessarily be a `flask` view function and should contain
    appropriate sections in its docstring.

    Args:
        func: The function for which to generate responses

    Returns:
        A dictionary containing the responses extracted from the docstring.

    """
    if func.__doc__ is None:
        return {}
    doc = docstring.GoogleDocstring(func.__doc__)
    responses = {}

    def response_subroutine(name, response_str):
        response_args = exec_and_return(response_str)
        # if isinstance(response_args, ResponseSchema):
        if response_args.mimetype == mt.text:
            content = response_args.schema()
        elif response_args.mimetype == mt.json:
            sf = response_args.schema_field
            if not hasattr(doc, "schemas") or doc.schemas is None:
                spec = check_for_redirects(sf, func)
                var = sf.split(":")[-1].strip()
                spec = get_schema_var(spec[1], var, func)
            else:
                var = sf.split(":")[-1].strip()
                spec = get_schema_var(doc.schemas[1], var, func)
            schema = spec.schema()
            if "description" in schema:
                schema.pop("description")
            content = response_args.schema(schema)
        responses[name] = content
        # else:
        #     responses[name] = gen_response(name, *response_args)
        #     if response_args[-2] == mt.json:
        #         if not hasattr(doc, "schemas") or doc.schemas is None:
        #             schema = check_for_redirects(response_args[-1], func)
        #             var = response_args[-1].split(":")[-1]
        #             schema = get_schema_var(schema[1], var)
        #         else:
        #             var = response_args[-1].split(":")[-1]
        #             schema = get_schema_var(doc.schemas[1], var)
        #         if schema is None:
        #             raise AttributeError(f"{responses[name]} should be in schemas")
        #         schema_dict = schema.schema()
        #         schema_dict.pop("title")
        #         code = [*responses[name].keys()][0]
        #         responses[name][code]['content']['application/json'] = schema_dict

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
    retval = {}
    for x in responses.values():
        retval.update(x)
    return retval


# def gen_response(name: str, code: int, desc: str, mimetype: mt,
#                  content, **kwargs) -> Dict[int, Dict[str, Any]]:
#     import ipdb; ipdb.set_trace()
#     if mimetype == mt.text:
#         content = {"properties": {"type": "string"}}
#     return {code: {"description": desc,
#                    "content": {mimetype: {"schema": content}}}}


def get_schema_var(schemas: List[str], var: str,
                   func: Optional[Callable] = None) -> BaseModel:
    """Extract and return a `pydantic.BaseModel` from docstring.

    Args:
        schemas: The lines of the schemas section of the docstring
        func: The function from which to extract the type.

    Returns:
        A `BaseModel` type, or `DefaultModel` if the variable is not found.

    """
    ldict: Dict[str, Any] = {}
    for i, s in enumerate(schemas):
        if re.match(ref_regex, s):
            indent = [*filter(None, re.split(r'(\W+)', s))][0]
            typename = s.strip().split(":", 1)[0]
            target, trailing = ref_repl(s).rsplit(".", 1)
            if target.startswith("~"):
                target = target[1:]
            try:
                if func is not None:
                    tfunc = get_func_for_redirect(target, func)
                    if isinstance(tfunc, property):
                        tfunc = tfunc.fget
                    if trailing.strip(" ").startswith("return") and\
                       "return" in tfunc.__annotations__:
                        schemas[i] = indent + typename + ": " +\
                            str(tfunc.__annotations__["return"])
            except Exception as e:
                print(f"Error {e} for {func} in get_schema_var")
                schemas[i] = indent + typename + ": " + "Optional[Any]"
    exec("\n".join(schemas), globals(), ldict)
    if var not in ldict:
        return DefaultModel
    else:
        return ldict[var]


def get_description(func: Callable) -> str:
    if func.__doc__ is None:
        return ""
    else:
        doc = docstring.GoogleDocstring(func.__doc__)
        if doc.lines():
            return doc.lines()[0]
        else:
            return ""


def get_request_params(lines: List[str]) -> List[Dict[str, Any]]:
    ldict: Dict[str, Any] = {}
    lines = ["    " + x for x in lines]
    exec("\n".join(["class Params(ParamsModel):", *lines]), globals(), ldict)
    params = ldict["Params"]
    schema = params.schema()
    retval = []
    for k, w in schema["properties"].items():
        temp = {}
        w.pop("title")
        temp["required"] = w.pop("required")
        temp["name"] = k
        temp["in"] = "query"
        temp["schema"] = w
        retval.append(temp)
    return retval


def get_request_body(lines: List[str]) -> List[Dict[str, Any]]:
    ldict: Dict[str, Any] = {}
    lines = ["    " + x for x in lines]
    exec("\n".join(["class Body(BaseModel):", *lines]), globals(), ldict)
    body = ldict["Body"]
    return body.schema()
    # retval = []
    # for k, w in schema["properties"].items():
    #     temp = {}
    #     w.pop("title")
    #     temp["required"] = w.pop("required")
    #     temp["name"] = k
    #     temp["in"] = "query"
    #     temp["schema"] = w
    #     retval.append(temp)
    # return retval


def get_requests(func: Callable, method: str) -> Dict:
    if func.__doc__ is None:
        return {}
    doc = docstring.GoogleDocstring(func.__doc__)
    if not hasattr(doc, "requests"):
        return {}
    else:
        lines = doc.requests[1]
        sections: Dict[str, Union[str, List[str]]] = {}
        for line in lines:
            if not line.startswith(" "):
                subsection, val = line.split(":", 1)
                if val.strip():
                    sections[subsection] = val.strip()
                else:
                    subsection = line.strip(":").strip(" ")
                    sections[subsection] = []
            if line.startswith(" "):
                sections[subsection].append(line.strip())
        return sections


def get_tags(func: Callable) -> List[str]:
    if func.__doc__ is None:
        return []
    doc = docstring.GoogleDocstring(func.__doc__)
    if not hasattr(doc, "tags"):
        return []
    else:
        tags = re.sub(r' +', '', doc.tags)
        tags = re.sub(r',+', ',', tags)
    if tags:
        return tags.split(",")
    else:
        return []


def get_specs_for_path(name: str, rule: werkzeug.routing.Rule,
                       method_func: Callable, method: str) ->\
                       Tuple[Dict[str, Any], Tuple[str, str]]:
    retval = {}
    # FIXME: Find a better way to generate operationId
    retval["operationId"] = name.split("/")[1] + "_" +\
        rule.endpoint.strip("_")
    request = get_requests(method_func, method)
    tags = get_tags(method_func)
    description = get_description(method_func)
    if description:
        retval["description"] = description
    if tags:
        retval["tags"] = tags
    if "params" in request:
        parameters = get_request_params(request["params"])
        retval["parameters"] = parameters
    if "body" in request:
        if method.lower() == "get":
            return {}, (rule.rule, "Request body cannot be in GET")
        elif method.lower() == "post":
            body = get_request_body(request["body"])
            body.pop("title")
            body.pop("description")
            if "content-type" in request:
                try:
                    content_type = exec_and_return(request["content-type"]).value
                except Exception as e:
                    try:
                        content_type = mt(request["content-type"]).value
                    except Exception as ex:
                        return {}, (rule.rule, f"{e, ex}")
            else:
                content_type = mt.json.value
            request_body = {"content": {content_type: {"schema": body}}}
            retval["requestBody"] = request_body
            retval['x-codegen-request-body-name'] = 'body'
        else:
            return {}, (rule.rule, "Only methods GET and POST are supported")
    try:
        responses = generate_responses(method_func)
        retval["responses"] = responses
        error = ()
    except Exception as e:
        retval = {}
        # if "local variable" in f"{e}":
        #     import ipdb; ipdb.set_trace()
        error = (rule.rule, f"{e}")
    # security = [{'petstore_auth': ['write:pets', 'read:pets']}]
    return retval, error


def make_paths(app: flask.Flask, excludes: List[str]) ->\
        Tuple[Dict, List[Tuple[str, str]], List[str]]:
    paths: Dict = {}
    errors: List[Tuple[str, str]] = []
    excluded: List[str] = []
    default_response = {200: {"content": {"application/json":
                                          {"schema": {"type": "object",
                                                      "content": {},
                                                      "nullable": True}}}}}
    for rule in app.url_map.iter_rules():
        name = rule.rule
        if any(re.match(e, name) for e in excludes):
            excluded.append(name)
            continue
        endpoint = app.view_functions[rule.endpoint]
        # name.split("/")[1] in {"trainer", "check_task"}:  # in {"/trainer/<port>/<endpoint>"}:
        paths[name] = {}
        for method in rule.methods:
            if method in ["GET", "POST"]:
                if hasattr(endpoint, "view_class"):
                    method_func = getattr(endpoint.view_class, method.lower())
                else:
                    if not {"GET", "POST"} - set(rule.methods):
                        print("Multiple methods not supported for rule {rule.rule}" +
                              "without MethodView", file=sys.stderr)
                        errors.append((rule.rule, "Multiple methods without methodview"))
                        continue
                    else:
                        method_func = resolve_partials(endpoint)
                spec, error = get_specs_for_path(name, rule,
                                                 method_func,
                                                 method.lower())
                if error:
                    paths[name][method.lower()] = default_response
                    errors.append(error)
                else:
                    paths[name][method.lower()] = spec
    return paths, errors, excluded


def openapi_spec(app: flask.Flask, excludes: List[str] = []) ->\
        Tuple[Dict[str, Union[str, Dict]], List[Tuple[str, str]], List[str]]:
    """Generate openAPI spec for a :mod:`flask` app.

    Args:
        app: The flask app for which the paths should be generated.
                The app should be live and running.
        exclude: A list of regexps to exlucde from spec generation
                Useful paths like /static/ etc.

    Returns:
        A tuple of `api_spec` and `errors` which occurred during the spec generation

    """
    paths, errors, excluded = make_paths(app, excludes)
    return {'openapi': '3.0.1',
            'info': {'title': 'DORC Server',
                     'description': 'API specification for Deep Learning ORChestrator',
                     'license': {'name': 'MIT'},
                     'version': '1.0.0'},
            "paths": paths}, errors, excluded


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
