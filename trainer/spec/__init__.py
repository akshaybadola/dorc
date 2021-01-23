import typing
from typing import Union, List, Callable, Dict, Tuple, Optional, Any, Type
import pathlib
import werkzeug
import functools
import flask
import re
import sys
import pydantic
import ipaddress

from .. import daemon
from .. import trainer
from .. import interfaces

from . import docstring
from .schemas import ResponseSchema, MimeTypes, MimeTypes as mt,\
    FlaskTypes as ft, SwaggerTypes as st
from .models import BaseModel, ParamsModel, DefaultModel


try:
    from types import NoneType
except Exception:
    NoneType = type(None)


def recurse_dict_with_pop(jdict: Dict[str, Any],
                          p_pred: Optional[Callable[[str, Any], bool]],
                          r_pred: Callable[[str, str], bool],
                          repl: Callable[[str], str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    to_pop = []
    popped: Dict[str, Any] = {}
    for k, v in jdict.items():
        # if k == key and v == sub:
        if p_pred is not None and p_pred(k, v):
            to_pop.append(k)
        if r_pred(k, v):          # k == "$ref", v.startswith("#/definitions/")
            jdict[k] = repl(v)  # v.replace("#/definitions/", "#/components/schemas")
        if isinstance(v, dict):
            jdict[k], _popped = recurse_dict_with_pop(v, p_pred, r_pred, repl)
            popped.update(**_popped)
    for p in to_pop:
        popped.update(jdict.pop(p))
    return jdict, popped


def recurse_dict(jdict: Dict[str, Any],
                 pred: Callable[[str, str], bool],
                 repl: Callable[[str], str]) -> Dict[str, Any]:
    if not (isinstance(jdict, dict) or isinstance(jdict, list)):
        return jdict
    if isinstance(jdict, dict):
        for k, v in jdict.items():
            # if k == key and v == sub:
            if pred(k, v):         # k == "$ref", v.startswith("#/definitions/")
                jdict[k] = repl(v)  # v.replace("#/definitions/", "#/components/schemas")
            if isinstance(v, dict):
                jdict[k] = recurse_dict(v, pred, repl)
            if isinstance(v, list):
                for i, item in enumerate(v):
                    v[i] = recurse_dict(item, pred, repl)
    elif isinstance(jdict, list):
        for i, item in enumerate(jdict):
            jdict[i] = recurse_dict(item, pred, repl)
    return jdict


file_content = {'content': {'multipart/form-data':
                            {'schema':
                             {'properties':
                              {'additionalMetadata':
                               {'type': 'string',
                                'description': 'Additional data to pass to server'},
                               'file': {'type': 'string',
                                        'description': 'file to upload',
                                        'format': 'binary'}}}}}}


ref_regex = re.compile(r'(.*)(:[a-zA-Z0-9]+[\-_+:.])`(.+?)`')
param_regex = re.compile(r' *([a-zA-Z]+[a-zA-Z0-9_]+?)( *: *)(.+)')
attr_regex = re.compile(r'(:[a-zA-Z0-9]+[\-_+:.])(`.+?`)( *: *)?([a-zA-Z]+[a-zA-Z0-9_]+?)?')


def ref_repl(x: str) -> str:
    """Replace any reference markup from docstring with empty string.
    Uses :attr:`ref_regex`

    Args:
        x: String on which to do replacement

    Returns:
        The replaced string

    """
    return re.sub(r'~?(.+),.*', r'\1', re.sub(ref_regex, r'\3', x))


def exec_and_return(exec_str: str,
                    modules: Optional[Dict[str, Any]] = None) -> Any:
    """Execute the exec_str with :meth:`exec` and return the value

    Args:
        exec_str: The string to execute

    Returns:
        The value in the `exec_str`

    """
    ldict: Dict[str, Any] = {}
    if modules is not None:
        exec("testvar = " + exec_str, {**modules, **globals()}, ldict)
    else:
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
        func = exec_and_return(".".join([redirect_from.__module__, func_name]), globals())
        return func
    except Exception:
        pass
    try:
        modname = redirect_from.__module__
        exec(f"import {modname}")
        if modname in sys.modules:
            func = exec_and_return(".".join([redirect_from.__module__, func_name]),
                                   {modname: sys.modules[modname]})
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
        Tuple[Optional[Callable], str]:
    """Check for indirections in the given `var`.

    This function checks for indirections in cases:
        a. the docstring doesn't have a schemas section
        b. The indirection is to another function, either a view function
           or a regular function

    The `var` would be part of the `Responses` section of the docstring, usually
    the part which specfies a schema variable.

    In case the indirection is to another view function, the schema variable
    must be specified as there can be multiple schemas present in any given
    docstring. Otherwise, the schema is inferred from the return annotations of
    the function.

    Args:
        var: Part of the docstring to process
        redirect_from: Current function from which it's extracted

    Returns:
        A schema variable or None if none found after redirects.

    """
    if re.match(ref_regex, var):
        var = ref_repl(var)
    if len(var.split(":")) > 1:
        func_name, attr = [x.strip() for x in var.split(":")]
    else:
        func_name, attr = var, "return"
    return get_func_for_redirect(func_name, redirect_from), attr


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


def check_indirection(response_str: str) -> Optional[Tuple[int, str]]:
    """Check for indirections in `response_str`.

    This function checks for indirections in case the response_str is of type:
        ResponseSchema(200, "Some description", <indirection>, <indirection>)

    <indirection> in this case is a directive of type `:func:mod.some_func`

    In this case, both the return type and the schema are given by the latter
    function (or property) and are unkonwn to the view function. The response
    schema is determined by the annotations of that function.

    Args:
        response_str: A string which possibly evaluates to :class:`ResponseSchema`

    Returns:
        A schema variable or None if none found after redirects.

    """
    match = re.match(r'.*\((.+)\).*', response_str)
    retval = None
    if match:
        inner = [x.strip() for x in match.groups()[0].split(",")]
        type_field = inner[-2]
        schema_field = inner[-1]
        if re.match(ref_regex, type_field) and re.match(ref_regex, schema_field):
            retval = int(inner[0]), inner[1].replace("'", "").replace('"', "")
    return retval


def infer_from_annotations(func: Callable) ->\
        Tuple[MimeTypes, Union[str, BaseModel, Dict[str, str]]]:
    """Infer an OpenAPI response schema from function annotations.

    The annotations are converted to a `BaseModel` and schema is extracted from it.

    Args:
        func: The function whose annotations are to be inferred.

    Returns:
        A `BaseModel` generated from the annotation's return value

    """
    annot = func.__annotations__
    if 'return' not in annot:
        raise AttributeError(f"return not in annotations for {func}")
    ldict: Dict[str, Any] = {}
    # TODO: Parse and add example
    if annot["return"] == str:
        return mt.text, ""
    if annot["return"] in st:
        return mt.text, st[annot["return"]]
    if isinstance(annot["return"], type) and issubclass(annot["return"], pydantic.BaseModel):
        return mt.json, annot["return"]
    elif type(annot["return"]) in typing.__dict__.values():
        class Annot(BaseModel):
            default: annot["return"]  # type: ignore
        return mt.json, Annot
    else:
        annot_ret = str(annot["return"])
        # NOTE: Substitute property with Callable. property is not json
        #       serializable Doesn't make a difference though. pydantic exports
        #       it as: {"object": {"properties": {}}}
        annot_ret = re.sub(r'([ \[\],]+?.*?)(property)(.*?[ \[\],]+?)', r"\1Callable\3", annot_ret)
        lines = ["    " + "default: " + annot_ret]
        exec("\n".join(["class Annot(BaseModel):", *lines]), globals(), ldict)
        return mt.json, ldict["Annot"]


def get_schema_var(schemas: List[str], var: str,
                   func: Optional[Callable] = None) -> Type[BaseModel]:
    """Extract and return a `pydantic.BaseModel` from docstring.

    Args:
        schemas: The lines of the schemas section of the docstring
        func: The function from which to extract the type.

    Returns:
        A `BaseModel` type, or `DefaultModel` if the variable is not found.

    """
    ldict: Dict[str, Any] = {}
    tfunc = None
    for i, s in enumerate(schemas):
        if re.match(ref_regex, s):
            indent = [*filter(None, re.split(r'(\W+)', s))][0]
            typename = s.strip().split(":", 1)[0]
            target, trailing = ref_repl(s).rsplit(".", 1)
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
        raise AttributeError(f"{var} not in docstring Schemas for {(tfunc or func)}")
    else:
        return ldict[var]


def generate_responses(func: Callable, rulename: str, redirect: str) -> Dict[int, Dict]:
    """Generate OpenAPI compliant responses from a given `func`.

    `func` would necessarily be a `flask` view function and should contain
    appropriate sections in its docstring.

    What we would normally be looking for is `Requests`, `Responses` and `Maps`.
    In case, the Request or Response is processed or sent by another function,
    it can be pointed to as a sphinx directive, like \"See `:directive:`\".

    Args:
        func: The function for which to generate responses

    Returns:
        A dictionary containing the responses extracted from the docstring.

    """
    if func.__doc__ is None:
        return {}
    doc = docstring.GoogleDocstring(func.__doc__)
    responses = {}

    # if "config_file" in rulename:
    #     import ipdb; ipdb.set_trace()
    def remove_description(schema):
        if "title" in schema:
            schema["title"] = "default"
        if "description" in schema:
            schema.pop("description")
        return schema

    def response_subroutine(name, response_str):
        inner_two = check_indirection(response_str)
        if redirect and inner_two:
            redir_func = get_func_for_redirect(redirect.lstrip("~"), func)
            if isinstance(redir_func, property):
                redir_func = redir_func.fget
            mtt, ret = infer_from_annotations(redir_func)
            if mtt == mt.text:
                if ret:
                    response = ResponseSchema(*inner_two, mtt, spec=ret)
                else:
                    response = ResponseSchema(*inner_two, mtt, ret)
            else:
                schema = remove_description(ret.schema())
                response = ResponseSchema(*inner_two, mtt, spec=schema)
            content = response.schema()
        else:
            response = exec_and_return(response_str)
            if response.mimetype == mt.text:
                content = response.schema()
            elif response.mimetype in {mt.json, mt.binary}:
                sf = response.schema_field
                # Basically there are two cases
                # 1. we redirect to another view function
                # 2. we redirect to a regular function or method
                if not hasattr(doc, "schemas") or doc.schemas is None:
                    # FIXME: Error is here
                    #        check_for_redirects is called if above condition is true
                    redir_func, attr = check_for_redirects(sf, func)
                    if not redir_func:
                        raise AttributeError("Dead end for redirect")
                    elif attr == "return":
                        if isinstance(redir_func, property):
                            redir_func = redir_func.fget
                        mtt, ret = infer_from_annotations(redir_func)
                        if mtt == mt.text:
                            response = ResponseSchema(*inner_two, mtt, ret)
                        elif inner_two:
                            schema = remove_description(ret.schema())
                            response = ResponseSchema(*inner_two, mtt, spec=schema)
                        else:
                            schema = remove_description(ret.schema())
                            response.spec = schema
                        content = response.schema()
                    else:
                        import ipdb; ipdb.set_trace()
                        spec = get_schema_var(spec[1], var, func)
                else:
                    var = sf.split(":")[-1].strip()
                    spec = get_schema_var(doc.schemas[1], var, func)
                    schema = remove_description(spec.schema())
                    content = response.schema(schema)
        responses[name] = content

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


def join_subsection(lines: List[str]) -> List[str]:
    """Join indented subsection to a single line.

    Args:
        lines: List of lines to (possibly) join


    For example:

    The following line will be parsed as two lines:

        bleh: Union[List[Dict], None, str, int, bool,
                    Dict[List[str]], List[List[str]]]

    It will be joined to:

        bleh: Union[List[Dict], None, str, int, bool, Dict[List[str]], List[List[str]]]

    """
    _lines = []
    prev = ""
    for line in lines:
        if not re.match(param_regex, line):
            prev += line
        else:
            if prev:
                _lines.append(prev)
            prev = line
    _lines.append(prev)
    return _lines


def get_request_body(lines: List[str],
                     current_func: Optional[Callable] = None) -> Dict[str, Any]:
    ldict: Dict[str, Any] = {}
    lines = join_subsection(lines)
    _lines = lines.copy()
    if current_func is not None:
        for i, line in enumerate(lines):
            matches = attr_regex.findall(line)
            if matches:
                for match in matches:
                    attr_str = "".join(match)
                    redir_func, attr = check_for_redirects(attr_str, current_func)
                    if redir_func:
                        if attr == "return":
                            mtt, ret = infer_from_annotations(redir_func)
                            # Although mypy gives a type error here, because of
                            # attr_regex match infer_from_annotations should
                            # always return some annotation or raise error
                            attr_name = line.split(":")[0].strip() + "_" + ret.__name__
                            exec(f"{attr_name} = ret")
                            exec(f'lines[i] = re.sub(attr_regex, "{attr_name}", lines[i], count=1)')
                        elif not redir_func.__doc__:
                            raise AttributeError(f"{redir_func} has no docstring")
                        else:
                            doc = docstring.GoogleDocstring(redir_func.__doc__)
                            if not hasattr(doc, "schemas"):
                                raise AttributeError(f"No schema in doc for {redir_func}")
                            else:
                                exec(f'{attr} = get_schema_var(doc.schemas[1], attr)')
                                exec(f'lines[i] = re.sub(attr_regex, "{attr}", lines[i], count=1)')
                    else:
                        raise ValueError(f"Error parsing for {attr_str} and {current_func}")
    lines = ["    " + x for x in lines]
    exec("\n".join(["class Body(BaseModel):", *lines]), {**globals(), **locals()}, ldict)
    body = ldict["Body"]
    return body.schema()


def get_requests(func: Callable, method: str) -> Dict:
    if func.__doc__ is None:
        return {}
    doc = docstring.GoogleDocstring(func.__doc__)
    if not hasattr(doc, "requests"):
        return {}
    else:
        lines = doc.requests[1]
        sections: Dict[str, Union[List[str], str]] = {}
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


def get_opId(name: str, func: Callable, params: List[str], method: str) -> str:
    mod = func.__qualname__.split(".")[0]
    name = name.split("/")[1]
    return mod + "__" + name + "__" + "_".join(params) + method.upper()


def get_params_in_path(name: str) -> List[Dict[str, Any]]:
    params_in_path = re.findall(r"\<(.+?)\>", name)
    params = []
    if params_in_path:
        for p in params_in_path:
            p_type = "string"
            splits = p.split(":")
            if len(splits) > 1:
                p = splits[1]
                p_type = ft[splits[0]] if splits[0] in ft else "string"
            param = {"in": "path",
                     "name": p,
                     "required": True,
                     "schema": {"type": p_type}}
            if p_type == "uuid":
                param["schema"] = {"type": "string", "format": "uuid"}
            params.append(param)
    return params


def check_function_redirect(docstr: Optional[str], rulename: str) -> Tuple[str, str]:
    var = ""
    rest = ""
    if docstr:
        doc = docstring.GoogleDocstring(docstr)
        if hasattr(doc, "map"):
            rule, rest = [x.strip() for x in doc.map.split(":", 1)]
            rule_ = rule.split("/")
            rest = ref_repl(rest)
            rulename_ = rulename.split("/")
            for x, y in zip(rule_, rulename_):
                if re.match(r'\<.+\>', x):
                    rest = rest.replace(x, y)
                    var = x[1:-1]
    return var, rest


def get_specs_for_path(name: str, rule: werkzeug.routing.Rule,
                       method_func: Callable, method: str) ->\
                       Tuple[Dict[str, Any], Tuple[str, str]]: # NOQA
    retval: Dict[str, Any] = {}
    # FIXME: Find a better way to generate operationId
    request = get_requests(method_func, method)
    tags = get_tags(method_func)
    description = get_description(method_func)
    var, redirect = check_function_redirect(method_func.__doc__, name)
    if description:
        if redirect:
            last = redirect.split(".")[-1]
            description = re.sub(f"`{var}`", f"`{last}`", description)
        retval["description"] = description
    if tags:
        retval["tags"] = tags
    parameters: List[Dict[str, Any]] = get_params_in_path(name)
    # TODO: Fix opId in case there's indirection
    #       /props/devices has currently FlaskInterface__props__GET
    #       instead of FlaskInterface__props_device__GET or something
    #       It can also be getTrainerPropsDevice based on some rules
    retval["operationId"] = get_opId(name,
                                     method_func,
                                     [x["name"] for x in parameters],
                                     method)
    # if "/props/" in name:
    #     import ipdb; ipdb.set_trace()
    if "params" in request:
        parameters.extend(get_request_params(request["params"]))
    if parameters:
        retval["parameters"] = parameters
    if "body" in request:
        if method.lower() == "get":
            return {}, (rule.rule, "Request body cannot be in GET")
        elif method.lower() == "post":
            body = get_request_body(request["body"], method_func)
            # NOTE: Hack because for some reason, the title and description are of
            # :class:`BaseModel` instead of the docstring
            body.pop("title", None)
            body.pop("description", None)
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
        responses = generate_responses(method_func, name, redirect)
        retval["responses"] = responses
        error = ()
    except Exception as e:
        retval = {}
        # if "local variable" in f"{e}":
        #     import ipdb; ipdb.set_trace()
        error = (str(rule.rule), f"{e}")
    return retval, error


def make_paths(app: flask.Flask, excludes: List[str]) ->\
        Tuple[Dict, List[Tuple[str, str]], List[str]]:
    """Generate OpenAPI `paths` component for a :class:`~flask.Flask` app

    Args:
        app: :class:`~flask.Flask` app
        excludes: List of regexps to exclude.
                  The regexp is matched against the rule name

    Return:
        A tuple of generated paths, errors and excluded rules.

    Paths returned are a dictionary of rule name and the schema for it. Errors
    similarly are a tuple of rule name and the error that occurred. Excluded
    rules are returned as a :class:`list`.

    """
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
        newname = name
        params_in_path = re.findall(r"\<(.+?)\>", name)
        for param in params_in_path:
            repl = param.split(":")[1] if len(param.split(":")) > 1 else param
            newname = newname.replace(f"<{param}>", "{" + repl + "}")
        paths[newname] = {}
        for method in rule.methods:
            if method in ["GET", "POST"]:
                if hasattr(endpoint, "view_class"):
                    method_func = getattr(endpoint.view_class, method.lower())
                else:
                    if not {"GET", "POST"} - set(rule.methods):
                        print(f"Multiple methods not supported for rule {rule.rule} " +
                              "without MethodView", file=sys.stderr)
                        errors.append((rule.rule, "Multiple methods without methodview"))
                        continue
                    else:
                        method_func = resolve_partials(endpoint)
                spec, error = get_specs_for_path(name, rule,
                                                 method_func,
                                                 method.lower())
                if error:
                    paths[newname][method.lower()] = default_response
                    errors.append(error)
                else:
                    paths[newname][method.lower()] = spec
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

    def pred(a, b):
        if a == "$ref" and b.startswith("#/definitions/"):
            return True
        elif a == "type" and b == "type":
            return True
        else:
            return False

    def repl(v):
        if v.startswith("#/definitions/"):
            return v.replace("#/definitions/", "#/components/schemas/")
        elif v == "type":
            return "string"

    paths = recurse_dict(paths, pred, repl)

    def pop_pred(x, y):
        return x == "definitions" and isinstance(y, dict)

    def pop_if(pred, jdict):
        to_pop = []
        popped = {}
        for k, v in jdict.items():
            if pred(k, v):
                to_pop.append(k)
            if isinstance(v, dict):
                popped.update(pop_if(pred, v))
        for p in to_pop:
            popped.update(jdict.pop(p))
        return popped

    definitions = pop_if(pop_pred, paths)
    return {'openapi': '3.0.1',
            'info': {'title': 'DORC Server',
                     'description': 'API specification for Deep Learning ORChestrator',
                     'license': {'name': 'MIT'},
                     'version': '1.0.0'},
            "paths": paths,
            "components": {"schemas": definitions}}, errors, excluded
