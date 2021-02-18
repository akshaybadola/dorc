import typing
from typing import Union, List, Callable, Dict, Tuple, Optional, Any, Type, Iterable
import werkzeug
import functools
import flask
import re
import sys
import pathlib
import pydantic
from pydantic import BaseConfig
from pydantic.fields import ModelField
import ipaddress
import traceback

from ..util import exec_and_return

from . import docstring
from .schemas import ResponseSchema, MimeTypes, MimeTypes as mt,\
    FlaskTypes as ft, SwaggerTypes as st
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


ref_regex = re.compile(r'(.*)(:[a-zA-Z0-9]+[\-_+:.])`(.+?)`')
param_regex = re.compile(r' *([a-zA-Z]+[a-zA-Z0-9_]+?)( *: *)(.+)')
# attr_regex = re.compile(r'(:[a-zA-Z0-9]+[\-_+:.])(`.+?`)( *: *)?([a-zA-Z]+[a-zA-Z0-9_]+?)?')
attr_regex = re.compile(r'(:[a-zA-Z0-9]+[\-_+:.])(`.+?`)( *: *)?((<_>)?[a-zA-Z]+[a-zA-Z0-9_]+?)?')
global_modules: Dict[str, Any] = {}


def ref_repl(x: str) -> str:
    """Replace any reference markup from docstring with empty string.
    Uses :attr:`ref_regex`

    Args:
        x: String on which to do replacement

    Returns:
        The replaced string

    """
    return re.sub(r'~?(.+),?.*', r'\1', re.sub(ref_regex, r'\3', x))


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
    global global_modules
    try:
        func = exec_and_return(func_name, {**global_modules, **globals()})
        return func
    except Exception:
        pass
    try:
        func = exec_and_return(".".join([redirect_from.__module__, func_name]),
                               {**global_modules, **globals()})
        return func
    except Exception:
        pass
    try:
        modname = redirect_from.__module__
        exec(f"import {modname}")
        if modname in sys.modules:
            func = exec_and_return(".".join([redirect_from.__module__, func_name]),
                                   {modname: sys.modules[modname], **global_modules, **globals()})
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
    """Check for indirections in the given :code:`var`.

    This function checks for indirections in cases:

        a. the docstring doesn't have a schemas section
        b. The indirection is to another function, either a view function
           or a regular function

    The :code:`var` would be part of the :code:`Responses` section of the
    docstring, usually the part which specfies a schema variable.

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
    """Get an attribute :code:`attr` from function :code:`func_name` from context of
    :code:`redirect_from`

    Like :func:`check_for_redirects` but instead of checking whether current
    function's docstring contains :code:`schema` or not we check for attribute :code:`attr`.
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
    """Check for indirections in :code:`response_str`.

    This function checks for indirections in case the response_str is of the form:

        `ResponseSchema(200, "Some description", <indirection>, <indirection>)`

    `<indirection>` in this case is a directive of type `:func:mod.some_func`

    In this case, both the return type and the schema are given by the latter
    function (or property) and are unkonwn to the view function. The response
    schema is determined by the annotations of that function.

    Args:
        response_str: A string which possibly evaluates to :class:`ResponseSchema`

    Returns:
        A tuple of schema variable and status code or None if nothing
        found after redirects.

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


def infer_request_from_annotations(func: Callable) -> Optional[Type[BaseModel]]:
    """Infer an OpenAPI request body from function annotations.

    The annotations are converted to a `BaseModel` and schema is extracted from
    it.

    Args:
        func: The function whose annotations are to be inferred.

    Returns:
        A `BaseModel` generated from the annotation's return value

    """
    annot = {x: y for x, y in func.__annotations__.items() if x != "return"}
    if annot:
        class Annot(BaseModel):
            pass
        for x, y in annot.items():
            Annot.__fields__[x] = ModelField(name=x, type_=y, class_validators={},
                                             model_config=BaseConfig,
                                             required=True)
        return Annot
    else:
        return None


def infer_response_from_annotations(func: Callable) ->\
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
        raise AttributeError(f"'return' not in annotations for {func}")
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
        exec("\n".join(["class Annot(BaseModel):", *lines]), {**global_modules, **globals()}, ldict)
        return mt.json, ldict["Annot"]


def get_symbols_in_expr(schema_expr: List[str]) -> Dict[str, Any]:
    ldict: Dict[str, Any] = {}
    exec("\n".join(schema_expr), {**global_modules, **globals()}, ldict)
    return ldict


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
            # try:
            if func is not None:
                tfunc = get_func_for_redirect(target, func)
                if isinstance(tfunc, property):
                    tfunc = tfunc.fget
                if trailing.strip(" ").startswith("return") and\
                   "return" in tfunc.__annotations__:
                    schemas[i] = indent + typename + ": " +\
                        str(tfunc.__annotations__["return"])
            # except Exception as e:
            #     print(f"Error {e} for {func} in get_schema_var")
            #     schemas[i] = indent + typename + ": " + "Optional[Any]"
    exec("\n".join(schemas), {**global_modules, **globals()}, ldict)
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
        rulename: Name of the :class:`~werkzeug.routing.Rule`
        redirect: If there's a `redirect` to another function present.

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
        # if "title" in schema:
        #     schema["title"] = "default"
        # if "description" in schema:
        #     schema.pop("description")
        return schema

    def response_subroutine(name, response_str):
        inner_two = check_indirection(response_str)
        if redirect and inner_two:
            redir_func = get_func_for_redirect(redirect.lstrip("~"), func)
            if isinstance(redir_func, property):
                redir_func = redir_func.fget
            mtt, ret = infer_response_from_annotations(redir_func)
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
            response = exec_and_return(response_str, {**global_modules, **globals()})
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
                    elif isinstance(redir_func, type) and issubclass(redir_func, BaseModel):
                        content = remove_description(ret.schema())
                    elif attr == "return":
                        if isinstance(redir_func, property):
                            redir_func = redir_func.fget
                        mtt, ret = infer_response_from_annotations(redir_func)
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
            response_str = exec_and_return(response_str, {**global_modules, **globals()})
            response_subroutine(name, response_str)
        else:
            # print(name, func.__qualname__)
            response_subroutine(name, response_str)
    retval = {}
    for x in responses.values():
        retval.update(x)
    return retval


def schema_to_query_params(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert schema to `query` parameters list compatible with OpenAPI.

    Args:
        schema: A schema dictionary

    Returns:
        A List of parameter dictionaries.

    """
    retval = []
    for k, w in schema["properties"].items():
        temp = {}
        w.pop("title")
        if "required" in w:
            temp["required"] = w.pop("required")
        temp["name"] = k
        temp["in"] = "query"
        temp["schema"] = w
        retval.append(temp)
    return retval


def get_request_params(lines: List[str]) -> List[Dict[str, Any]]:
    """Return the paramters of an HTTP request if they exist.

    Args:
        lines: Lines of a function's docstring

    Returns:
        A List of schema dictionaries.

    """
    ldict: Dict[str, Any] = {}
    lines = ["    " + x for x in lines]
    exec("\n".join(["class Params(ParamsModel):", *lines]), {**global_modules, **globals()}, ldict)
    params = ldict["Params"]
    schema = params.schema()
    return schema_to_query_params(schema)


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
                     current_func: Optional[Callable] = None,
                     redir_func: Optional[Callable] = None) -> Dict[str, Any]:
    """Get the `Body` component of the request from the request lines.

    Args:
        lines: Lines of the request section of the docstring
        current_func: The current function which is being processed

    `lines` are joined according to indent and schemas are extracted.
    `current_func` is used to determine redirects.

    Returns:
        A :class:`dict` of schema.

    """
    ldict: Dict[str, Any] = {}
    lines = join_subsection(lines)
    _lines = lines.copy()
    if current_func is not None:
        for i, line in enumerate(lines):
            matches = attr_regex.findall(line)
            if matches:
                for match in matches:
                    match = [*match]
                    if "<_>" in match:
                        match.remove("<_>")
                    attr_str = "".join(match)
                    if not redir_func:
                        redir_func, attr = check_for_redirects(attr_str, current_func)
                    else:
                        _, attr = check_for_redirects(attr_str, current_func)
                    if redir_func:
                        if attr == "return":
                            mtt, ret = infer_response_from_annotations(redir_func)
                            # Although mypy gives a type error here, because of
                            # attr_regex match infer_response_from_annotations should
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
                            elif "<_>" in attr:
                                symbol_dict = get_symbols_in_expr(doc.schemas[1])
                                for k, v in symbol_dict.items():
                                    if re.match(r"[a-zA-Z]+" + attr.replace("<_>", ".*"), k):
                                        return v.schema()
                                raise TypeError(f"Could not infer from symbol_dict {symbol_dict}\n" +
                                                f"Schema: {doc.schemas[1]}")
                            else:
                                exec(f'{attr} = get_schema_var(doc.schemas[1], attr)')
                                exec(f'lines[i] = re.sub(attr_regex, "{attr}", lines[i], count=1)')
                    else:
                        raise ValueError(f"Error parsing for {attr_str} and {current_func}")
    lines = ["    " + x for x in lines]
    exec("\n".join(["class Body(BaseModel):", *lines]),
         {**global_modules, **globals(), **locals()}, ldict)
    body = ldict["Body"]
    return body.schema()


def get_requests(func: Callable, method: str, redirect: str) -> Dict:
    """Get the request section of a function's docstring

    Args:
        func: The function whose docstring will be processed.
        method: GET or POST

    Returns:
        A :class:`dict` of request with subsections as keys

    The subsections can be `content-type`, `parameters` or `file` etc.

    """
    # if redirect:
    #     import ipdb; ipdb.set_trace()
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
    """Get the `tags` section of a function's docstring

    Args:
        func: The function whose docstring will be processed.

    Returns:
        A :class:`list` of tags.
    """
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


def capitalize(x):
    return x[0].upper() + x[1:]


def capital_case(x):
    return x.title().replace("_", "")


def camel_case(x):
    x = capital_case(x)
    return x[0].lower() + x[1:]


def template_subroutine(x, components, prefix):
    if isinstance(x, str):
        x = [*filter(None, re.split(r'(%[a-zA-Z])', x))]
    for c, v in components.items():
        try:
            i = [_.lower() for _ in x].index(c)
            if x[i] == "%p":
                x[i] = x[i].replace(x[i], prefix.join(v))
                return x[i]
            elif x[i] == "%P":
                x[i] = x[i].replace(x[i], prefix.join([_ and capitalize(_) for _ in v]))
                return x[i]
            elif x[i] == "%h":
                x[i] = x[i].replace(x[i], v.lower())
            elif x[i] == "%H":
                x[i] = x[i].replace(x[i], v.upper())
            elif x[i] == c.upper():
                x[i] = v and x[i].replace(x[i], capitalize(v))
            else:
                x[i] = x[i].replace(x[i], v)
        except ValueError:
            pass
    return prefix.join(filter(None, x))


def get_opId(name: str, func: Callable,
             redir_func: Optional[Callable],
             params: List[str], method: str,
             template: str = "[__%C%f%r%n]_[_%p]_%H",
             aliases: Dict[str, str] = {}) -> str:
    """Generate a unique OpenAPI `operationId` for the function

    Args:
        name: name of an HTTP (flask usually) endpoint
        func: The function whose docstring will be processed.
        opid_template: Template for generation of OpenAPI operationId
                       Default template is `[__%C%f%r%n]_[_%p]_%H`, where:

                       - [_%x] represents "_".join(x) and [__%x] == "__".join(x) etc.
                       - %M is the (capitalized) module name
                       - %C is the (capitalized) class name
                       - %f is the function name
                       - %r is the redirected function name
                       - %n is the endpoint's basename
                       - %p are the parameters to the endpoint
                       - %H is the (uppercase) name of the method (GET,POST)

                      See https://swagger.io/docs/specification/paths-and-operations/
                      For details on operationId
        aliases: A list of aliases for module names

    The capitalized version of these (e.g, %R) indicates to capitalize that
    token.  %H will upcase the entire word (GET).  [%r] means to join the list
    with empty string "".  If only %p is given, the params will be joined with
    an empty string.  Nested braces [] are not allowed.  The final opId is
    converted to camelCase or CapitalCase depending on whether to capitalize the
    first token or not.

    If some attribute is not available for generation of opId, it's value is not
    considered.

    Examples:
        >>> get_opId(name, some_func, None, ["task_id"], "GET", "[__%M,%r,%n]_[_%p]_%H")
        >>> get_opId(name, some_func, None, ["task_id"], "GET", "[%M,%r,%n][%P]_%H")
        >>> get_opId(name, some_func, None, ["task_id"], "GET", "[%M,%r][%P]_%H")
        >>> get_opId(name, some_func, None, ["task_id"], "GET", "%M%R%p%H")

    Returns:
        An operation id

    """
    if not template:
        raise ValueError("Template cannot be empty")
    mod = func.__module__
    fname = func.__name__
    if len(func.__qualname__.split(".")) > 1:
        clsname = func.__qualname__.split(".")[0]
    else:
        clsname = ""
    if clsname == fname and "%f" in template:
        clsname = ""
    redir: str = redir_func.__qualname__.split(".")[-1] if redir_func else ""
    if redir == fname and "%f" in template:
        redir = ""
    name = name.split("/")[-1]
    if fname == name and "%n" in template:
        fname = ""
    reg = re.compile(r'\[.+?\]')
    if "[[" in "".join([x for x in template if x in {"[", "]"}]):
        raise ValueError(f"Nested braces are not allowed for template: {template}")
    matches = reg.findall(template)
    components = {"%m": camel_case(mod), "%c": clsname, "%f": fname,
                  "%r": redir, "%n": name, "%p": params,
                  "%h": method}
    if matches:
        for x in matches:
            x = x.strip('[]')
            prefix_match = re.match(r'^(.*?)%', x)
            if prefix_match:
                prefix = prefix_match.groups()[0]
                x = x.lstrip(prefix)
                x = template_subroutine(x, components, prefix)
                template = re.sub(reg, x, template, 1)
    template = template_subroutine(template, components, "")
    return template


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
                       method_func: Callable, method: str,
                       gen_opid: bool,
                       opid_template: str, aliases: Dict[str, str]) ->\
                       Tuple[Dict[str, Any], Tuple]: # NOQA
    errors: List[str] = []
    retval: Dict[str, Any] = {}
    # FIXME: Find a better way to generate operationId
    var, redirect = check_function_redirect(method_func.__doc__, name)
    request = get_requests(method_func, method, redirect)
    tags = get_tags(method_func)
    # if name == "/sessions":
    #     import ipdb; ipdb.set_trace()
    # if "archive_session" in name:
    #     import ipdb; ipdb.set_trace()
    parameters: List[Dict[str, Any]] = get_params_in_path(name)
    if redirect:
        redir_func = get_func_for_redirect(redirect, method_func)
        if redir_func is None:
            raise ValueError(f"Redirect {redirect} not found for {method_func}")
        else:
            try:
                doc = docstring.GoogleDocstring(redir_func.__doc__ or "")
            except Exception as e:
                errors.append(f"e")
                raise ValueError(f"Error parsing docstring for {redir_func}, {e}")
            description = getattr(doc, "description", "")
            tags.extend(get_tags(redir_func))
            if isinstance(redir_func, property):
                redir_func = redir_func.fget
            if "request" in redir_func.__annotations__ and\
               issubclass(redir_func.__annotations__['request'], flask.Request):
                errors.append(f"Flask request cannot be inferred {name}, {redir_func}")
                annot = None
            else:
                annot = infer_request_from_annotations(redir_func)
            if annot:
                redir_schema: Optional[Dict[str, Any]] = annot.schema()
            else:
                redir_schema = None
    else:
        redir_func = None
        try:
            doc = docstring.GoogleDocstring(method_func.__doc__ or "")
        except Exception as e:
            errors.append(f"{e}" + "\n" + traceback.format_exc())
            raise ValueError(f"Error parsing docstring for {method_func}, {e}")
        description = getattr(doc, "description", "")
    retval["description"] = description
    if gen_opid:
        retval["operationId"] = get_opId(name,
                                         method_func,
                                         redir_func,
                                         [x["name"] for x in parameters],
                                         method,
                                         opid_template,
                                         aliases)
    if tags:
        retval["tags"] = tags
    # TODO: Fix opId in case there's indirection
    #       /props/devices has currently FlaskInterface__props__GET
    #       instead of FlaskInterface__props_device__GET or something
    #       It can also be getTrainerPropsDevice based on some rules
    if "params" in request:
        parameters.extend(get_request_params(request["params"]))
    if redirect and redir_schema and method.lower() == "get":
        parameters.extend(schema_to_query_params(redir_schema))
    if redirect and redir_schema and method.lower() == "post":
        # FIXME: form
        request_body = {"content": {"application/json": {"schema": redir_schema}}}
        retval["requestBody"] = request_body
    if parameters:
        retval["parameters"] = parameters
    if "body" in request:
        if method.lower() == "get":
            return {}, (rule.rule, "Request body cannot be in GET")
        elif method.lower() == "post":
            body = get_request_body(request["body"], method_func, redir_func)
            # NOTE: Hack because for some reason, the title and description are of
            # :class:`BaseModel` instead of the docstring
            body.pop("title", None)
            body.pop("description", None)
            if "content-type" in request:
                try:
                    content_type = exec_and_return(request["content-type"],
                                                   {**global_modules, **globals()}).value
                except Exception as e:
                    try:
                        content_type = mt(request["content-type"]).value
                    except Exception as ex:
                        errors.append(f"{e, ex}" + "\n" + traceback.format_exc())
                        raise ValueError(f"Could not parse request body for {name}, {method}. " +
                                         f"Error {e, ex}")
                        # return {}, (rule.rule, f"{e, ex}")
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
        # raise ValueError(f"Could not parse response for {name}, {method}. " +
        #                  f"Error {e}")
        error = (str(rule.rule), f"{e}" + "\n" + traceback.format_exc() + "\n".join(errors))  # type: ignore
    return retval, error


def make_paths(app: flask.Flask, excludes: List[str],
               gen_opid: bool, opid_template: str,
               aliases: Dict[str, str]) ->\
        Tuple[Dict, List[Tuple[str, str]], List[str]]:
    """Generate OpenAPI `paths` component for a :class:`~flask.Flask` app

    Args:
        app: :class:`~flask.Flask` app
        excludes: List of regexps to exclude.
                  The regexp is matched against the rule name
        gen_opid: Whether to generate operationId or not
        opid_template: See :func:`openapi_spec`
        aliases: See :func:`openapi_spec`

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
                                                 method.lower(),
                                                 gen_opid,
                                                 opid_template,
                                                 aliases)
                if error:
                    paths[newname][method.lower()] = default_response
                    errors.append(error)
                else:
                    if not spec["responses"]:
                        errors.append((rule.rule, "Got empty response spec"))
                    paths[newname][method.lower()] = spec
    return paths, errors, excluded
