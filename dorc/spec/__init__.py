from typing import Union, List, Callable, Dict, Tuple, Optional, Any, Type, Iterable
import flask

from ..util import recurse_dict, pop_if
from .parser import make_paths, global_modules


def _init(modules):
    global global_modules
    global_modules.update(modules)


def extract_definitions(paths: Dict[str, Any], missing: List[type]) -> Dict[str, Any]:
    """Extract `definitions` from generated openAPI spec for :mod:`flask` app endpoints.

    Args:
        paths: The paths extracted from the docstrings.
        missing: Additional :class:`BaseModel` which are not converted to
                 :code:`/definitions` by :mod:`pydantic`.

    Returns:
        A dictionary of :code:`$ref`s which are stored under key `definitions`.

    :mod:`pydantic` on generating schema stores the :code:`$ref`s as :code:`definitions`
    which OpenAPI somehow doesn't like. We simply replace that with
    :code:`components/schema`

    """
    def pred(a, b):
        if a == "$ref" and b.startswith("#/definitions/"):
            return True
        elif a == "type" and b == "type":
            return True
        else:
            return False

    def repl(k, v):
        if v.startswith("#/definitions/"):
            return v.replace("#/definitions/", "#/components/schemas/")
        elif v == "type":
            return "string"

    paths = recurse_dict(paths, pred, repl)

    def pop_pred(x, y):
        return x == "definitions" and isinstance(y, dict)
    definitions = pop_if(paths, pop_pred)
    for x in missing:
        if x.__name__ not in definitions:
            definitions[x.__name__] = x.schema()
    return definitions


def fix_redundancies(paths: Dict[str, Any], definitions: Dict[str, Any]):
    """Lookup matching titles in specifications and replace with :code:`$ref`

    Args:
        paths: paths extracted from docs
        definitions: OpenaAPI model definitions

    Return:
        Fixed paths

    """
    def pred(k, v):
        if k == "schema" and "title" in v and v["title"] in definitions and\
           v == definitions[v["title"]]:
            return True
        else:
            return False

    def repl(k, v):
        ref = v["title"]
        return {'$ref': f'#/components/schemas/{ref}'}
    paths = recurse_dict(paths, pred, repl, True)
    return paths


def fix_yaml_references(spec: str) -> str:
    """Fix yaml generated references.

    Args:
        spec: Yaml dump of OpenAPI spec generated from pydantic

    For some reason, some refs aren't included by default by pydantic and yaml
    then inserts :code:`&id001` etc references in there. This replaces those with
    :code:`$ref`.

    Return:
        Fixed spec dump.

    """
    splits = spec.split("\n")
    refs = {}
    for i, x in enumerate(splits):
        if "&id" in x:
            ref, _id = [_.strip() for _ in x.split(": ")]
            refs[_id[1:]] = ref.rstrip(":")
            splits[i] = x.split(": ")[0] + ":"
    for i, x in enumerate(splits):
        if "*id" in x:
            name, _id = [_.strip() for _ in x.split(": ")]
            ref = refs[_id[1:]]
            splits[i] = x.split(":")[0] + ":\n" + x.split(":")[0].replace(name, "  ") +\
                f"$ref: '#/components/schemas/{ref}'"
    return "\n".join(splits)


def openapi_spec(app: flask.Flask, excludes: List[str] = [],
                 gen_opid: bool = False,
                 opid_template: str = "",
                 modules: Dict[str, Any] = {},
                 missing: List[type] = [],
                 aliases: Dict[str, str] = {}) ->\
        Tuple[Dict[str, Union[str, Dict]], List[Tuple[str, str]], List[str]]:
    """Generate openAPI spec for a :class:`flask.app.Flask` app.

    Args:
        app: The flask app for which the paths should be generated.
                The app should be live and running.
        exclude: A list of regexps to exlucde from spec generation
                Useful paths like /static/ etc.
        opid_template: Template for generation of OpenAPI operationID.
                       See :func:`get_opId` for details on its meaning.
        modules: A list of modules to for :func:`exec` commands.
                 `exec` will be run with :func:`globals` and those extra modules
                 which are given.
        missing: A list of missing :class:`BaseModel` types
                 For some reason, some types aren't extracted correctly.
                 In such cases such types can be passed as additional arguments.
        aliases: A list of aliases for module names

    Returns:
        A 3-tuple of `api_spec`, `errors` which occurred during the spec
        generation and a :class:`list` of rules that were excluded.

    See Also:
        :func:`fix_yaml_references`

    """
    global global_modules
    global_modules.update(modules)
    paths, errors, excluded = make_paths(app, excludes, gen_opid, opid_template, aliases)
    definitions = extract_definitions(paths, missing)
    fix_redundancies(paths, definitions)
    return {'openapi': '3.0.1',
            'info': {'title': 'DORC Server',
                     'description': 'API specification for Deep Learning ORChestrator',
                     'license': {'name': 'MIT'},
                     'version': '1.0.0'},
            "paths": paths,
            "components": {"schemas": definitions}}, errors, excluded
