import inspect
import pydantic
from typing import List, Union, Dict, Any, Optional, Callable
from types import ModuleType
import sys
import json
import copy
from functools import partial

from ..mods import load_module_exports, load_symbols
from ..util import exec_and_return, recurse_multi_dict, identity, serialize_defaults
from ..trainer import config as trainer_config


class TranslationLayer:
    """Translation layer to convert strings into python objects.

    Required for objects which can't be serialized to JSON, like functions,
    modules, classes etc.

    We'll define a simple mechanism to translate such objects from/JSON.
    """
    def __init__(self, jdict: Union[Dict, str], extra_opts: Dict[str, Any] = {}):
        if isinstance(jdict, dict):
            self.jdict = jdict.copy()
        else:
            self.jdict = json.loads(jdict)
        if "load_modules" in self.jdict:
            self.modules: Dict[str, ModuleType] = {}
            mods = self.jdict.pop("load_modules")
            self.modules = self.repl_module(mods["module"])
        else:
            self.modules = {}
        self.patched = None
        self.extra_opts = extra_opts

    def from_json(self):
        if self.patched is None:
            temp_dict = recurse_multi_dict(self.jdict, [self.pred_function,
                                                        self.pred_module,
                                                        self.pred_expression],
                                           {"function": self.repl_function,
                                            "module": self.repl_module,
                                            "expression": self.repl_expression})
            temp_dict = recurse_multi_dict(temp_dict, [partial(self.pred_shift_up, "expr")],
                                           {"expr_shift_up": partial(self.repl_shift_up, "expr")})
            temp_dict = recurse_multi_dict(temp_dict, [partial(self.pred_shift_up, "function")],
                                           {"function_shift_up":
                                            partial(self.repl_shift_up, "function",
                                                    transform=self.function_initialize)})
            self.patched = self.patch_config(temp_dict)
        return copy.deepcopy(self.patched.copy())

    def patch_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        _model = config["model_params"]["net"].pop("model")
        config["model_params"]["net"]["model"] = _model["function"]
        config["model_params"]["net"]["params"] = _model["params"]
        update_func_params = trainer_config.UpdateFunctionsParams(
            **config["model_step_params"]["params"])
        update_functions = trainer_config.UpdateFunctions(
            function=config["model_step_params"]["function"],
            params=update_func_params)
        config["update_functions"] = update_functions
        config.update(self.extra_opts)
        return config

    def convert_annotations(self, annot: dict):
        return None

    def initialize_functions(self, jdict):
        shifted = recurse_multi_dict(jdict,
                                     [partial(self.pred_shift_up, "function")],
                                     {"function_shift_up":
                                      partial(self.repl_shift_up, "function",
                                              transform=self.function_initialize)})
        return recurse_multi_dict(shifted, [self.pred_function],
                                  {"function": self.function_initialize})

    def function_initialize(self, v: dict) -> Dict[str, Any]:
        v_keys = [*v.keys()]
        v["function"] = v["callable"]
        for k in v_keys:
            if k not in {"function", "params"}:
                v.pop(k)
        return v

    def get_callable_dict(self, obj):
        annotations = getattr(obj, "__annotations__", {})
        doc = getattr(obj, "__doc__", "")
        sig = inspect.signature(obj)
        params = {k: "EMPTY" if v.default == inspect._empty else v.default
                  for k, v in sig.parameters.items()}
        return {"annotations": annotations,
                "doc": doc,
                "params": params}

    def encoder(self, obj):
        if isinstance(obj, type):
            retval = {"module": {"name": obj.__name__,
                                 "path": obj.__module__ + obj.__qualname__}}
            if callable(obj):
                retval["module"].update(self.get_callable_dict(obj))
            return json.dumps(retval)
        elif callable(obj):
            retval = {"function": {"name": (getattr(obj, "__name__", None) or
                                            getattr(obj.__class__, "__name__")),
                                   "path": obj.__module__ + (getattr(obj, "__qualname__", None) or
                                                             getattr(obj.__class__, "__qualname__")),
                                   **self.get_callable_dict(obj)}}
            return json.dumps(retval)
        else:
            return serialize_defaults(obj)

    def to_json(self, obj):
        if isinstance(obj, pydantic.BaseModel):
            return obj.json(encoder=self.encoder)
        else:
            return self.encoder(obj)

    def pred_shift_up(self, key: str, k: str, v: Any) -> str:
        if isinstance(v, dict) and key in v:
            return key + "_shift_up"
        else:
            return ""

    def pred_shift_down(self, key: str, k: str, v: Any) -> str:
        if isinstance(v, dict) and key in v:
            return key + "_shift_down"
        else:
            return ""

    def pred_function(self, k: str, v: Any) -> str:
        if k == "function" and isinstance(v, dict):
            return "function"
        else:
            return ""

    def pred_module(self, k: str, v: Any) -> str:
        if k == "module" and isinstance(v, str):
            return "module"
        else:
            return ""

    # certain heuristics like "transform" is an expression
    def pred_expression(self, k: str, v: str) -> str:
        if k in {"expression", "expr"} and isinstance(v, str):
            return "expression"
        else:
            return ""

    def repl_shift_up(self, key: str, v: Dict[str, Any],
                      transform: Callable[[Dict], Any] = identity) -> Any:
        return transform(v[key])

    def repl_shift_down(self, key: str, v: Dict[str, Any],
                        transform: Callable[[Dict], Any] = identity) -> Any:
        return {key: transform(v)}

    def repl_function(self, v: Dict[str, Any]) -> Any:
        modules = self.modules
        if "name" not in v and "path" in v:
            v["name"] = v["path"].rsplit(".")[-1]
        if "path" in v:
            if "." in v["path"]:
                mod = v["path"].split(".")[0]
                if mod not in sys.modules:
                    exec(f"import {mod}")
                    modules = {mod: sys.modules[mod], **self.modules}
            func = exec_and_return(v["path"], modules)
            return {"callable": func, **v}
        elif "source" in v:
            status, message = load_symbols(v["source"], [v["name"]], [], modules)
            if status:
                return {"callable": message[v["name"]], **v}
            else:
                raise ValueError(f"Could not load from {v}. Error {message}")
        else:
            raise AttributeError(f"None of path or source in {v}")

    def repl_module(self, v: Any) -> Any:
        if "path" in v:
            return exec_and_return(v["path"])
        elif "source" in v and "name" in v:
            status, message = load_symbols(v["source"], [v["name"]], [], self.modules)
        elif "source" in v and "names" in v:
            status, message = load_symbols(v["source"], v["names"], [], self.modules)
        else:
            if "source" in v:
                status, message = load_module_exports(v["source"], [], self.modules)
            else:
                status, message = load_module_exports(v, [], self.modules)
        if status:
            if "module_exports" in message:
                return message["module_exports"]
            else:
                return message
        else:
            raise ValueError(f"Could not load from {v}. Error {message}")

    def repl_expression(self, v: Any) -> Any:
        return exec_and_return(v, self.modules)
