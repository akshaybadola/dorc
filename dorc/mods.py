from typing import Callable, Iterable, Tuple, Dict, Any, List, Union, Optional
import os
import io
import sys
import uuid
from types import FunctionType
import shutil
import magic
import zipfile
import traceback

# imports needed for typing
import pathlib


def eval_python_exprs(module_str: Union[str, bytes],
                      checks: Iterable[Callable[[str], bool]],
                      modules: Dict[str, Any],
                      return_keys: List[str]) -> Tuple[bool, Union[str, Dict[str, Any]]]:
    """Evaluate a string or bytes as python expressions.

    Args:
        module_str: The string or bytes to evaluate.
        checks: An Iterable of functions which checks the loaded module. Can be empty.
        modules: A dictionary of additional modules to load while evaluting the string
        return_keys: The keys of the objects to return
    Returns:
        A tuple of status, and message if fails, or dictionary with
        the symbols and values.

    """
    try:
        ldict: Dict[str, Any] = {}
        flag = True
        if isinstance(module_str, bytes):
            module_str = module_str.decode("utf-8")
        for check_p in checks:
            if not check_p(module_str):
                flag = False
                break
        if not flag:
            return False, f"Module check failed {check_p}"
        else:
            print(f"Executing the string")
            exec(module_str, {**globals(), **modules}, ldict)
            retval = {}
            if all(k in ldict for k in return_keys):
                for k in return_keys:
                    retval[k] = ldict[k]
                return True, retval
            else:
                return False, f"Keys {set(return_keys) - set(ldict.keys())} not in module"
    except ImportError as e:
        return False, f"Could not import module_exports from given file. Error {e}" +\
            "\n" + traceback.format_exc()
    except Exception as e:
        return False, f"Some weird error occured while importing. Error {e}" +\
            "\n" + traceback.format_exc()


def load_module_exports(module_str: Union[str, bytes],
                        checks: Iterable[Callable[[str], bool]] = [],
                        modules: Dict[str, Any] = {}) ->\
        Tuple[bool, Union[str, Dict[str, Any]]]:
    return_keys = ["module_exports"]
    return eval_python_exprs(module_str, checks, modules, return_keys)


def load_symbols(expr_str: Union[str, bytes],
                 symbol_names: List[str],
                 checks: Iterable[Callable[[str], bool]] = [],
                 modules: Dict[str, Any] = {}) ->\
        Tuple[bool, Union[str, Dict[str, Any]]]:
    return_keys = symbol_names
    return eval_python_exprs(expr_str, checks, modules, return_keys)


def _print(x):
    print(x)
    return x


class Modules:
    """Dynamic loading and unloading of python modules.

    If two modules with the same name are imported, the previous one is
    overwritten. A global list of modules is maintained.

    The modules are listed as file/def where def is any variable definition:
    function, class or global variable. They're simply imported as file.var.

    Note:
        A bit of the code is old in the sense that :mod:`importlib` has changed to
        allow loading python modules without a file.

    Args:
        mods_dir: Path where modules are located.
                  Only top level directory is scanned
        loggers: Optional loggers

    """
    def __init__(self, mods_dir: str,
                 loggers: Dict[str, Callable[[str], str]] = {}):
        self._mods_dir = mods_dir
        if loggers:
            self._logd = loggers["logd"]
            self._loge = loggers["loge"]
            self._logi = loggers["logi"]
            self._logw = loggers["logw"]
        else:
            self._logd = _print
            self._loge = _print
            self._logi = _print
            self._logw = _print
        if not os.path.exists(mods_dir):
            self._logd(f"Creating directory {self._mods_dir}")
            os.mkdir(mods_dir)

    @classmethod
    def _check_file_magic(cls, _file: bytes, test_str: str) -> bool:
        """Determine the file type from `libmagic`

        Args:
            _file: Bytes of the file
            test_str: String to test for in the magic output
        Returns:
            True or False according to whether string is in magic output

        """
        if hasattr(magic, "from_buffer"):
            test = test_str in magic.from_buffer(_file).lower()
        elif hasattr(magic, "detect_from_content"):
            test = test_str in magic.detect_from_content(_file).name.lower()
        return test

    @classmethod
    def _load_python_file(cls, module: bytes, checks: Iterable[Callable[[str], bool]],
                          write_path: Union[str, pathlib.Path], exec_cmd: str,
                          return_key: str, env_str: str = "") ->\
            Tuple[bool, List[str], Optional[Dict]]:
        """Load a python file containing a module.

        Args:
            module: A stream of bytes. It's written on the disk and module loaded.
            checks: An Iterable of functions which checks the loaded module.
            write_path: Path to which the file is written
            exec_cmd: Command to execute for loading
            return_key: The key of the object to return
        Returns:
            A tuple of status, and message if fails or dictionary with the module.

        """
        msgs: List[str] = []
        msgs.append("Detected python file")
        with open(write_path, "w") as f:
            if env_str:
                f.write(env_str)
            f.write(module.decode())
            msgs.append(f"Written to {write_path}")
        try:
            ldict: Dict[str, Any] = {}
            msgs.append("Checking functions")
            flag = True
            for check_p in checks:
                if not check_p(str(write_path)):
                    flag = False
                    break
            if not flag:
                msgs.append(f"Module check failed {check_p}")
                return False, msgs, None
            else:
                msgs.append(f"Executing {exec_cmd}")
                exec(exec_cmd, globals(), ldict)
                return True, msgs, ldict[return_key]
        except ImportError as e:
            msgs.append(f"Could not import module_exports from given file. Error {e}"
                        + "\n" + traceback.format_exc())
            return False, msgs, None
        except Exception as e:
            msgs.append(f"Some weird error occured while importing. Error {e}"
                        + "\n" + traceback.format_exc())
            return False, msgs, None

    @classmethod
    def _load_zip_file(cls, module: bytes, checks: Iterable[Callable[[str], bool]],
                       write_path: Union[str, pathlib.Path], exec_cmd: str,
                       return_key: str, env_str: str = "") ->\
            Tuple[bool, List[str], Optional[Dict]]:
        """Load a zip file containing a module.

        The file should contain an __init__.py at the top level and that should
        contain all the definitions which should be available.  The zip file is
        extracted to a folder and loaded as a directory.

        Args:
            module: A stream of bytes. It's written on the disk and module loaded.
            checks: An Iterable of functions which checks the loaded module.
            write_path: Path to which the file is written
            exec_cmd: Command to execute for loading
            return_key: The key of the object to return
        Returns:
            A tuple of status, and message if fails or dictionary with the module.

        """
        zf = zipfile.ZipFile(io.BytesIO(module))
        # make sure that __init__.py is at the root of tmp_dir
        msgs: List[str] = []
        if not any(["__init__.py" in x.split("/")[0] for x in zf.namelist()]):
            msgs.append(f"zip file must have __init__.py at the top level")
            return False, msgs, None
        else:
            # zf.extractall(os.path.join(cls._mods_dir, write_path))
            msgs.append(f"Extracting to {write_path}")
            zf.extractall(write_path)
            if env_str:
                with open(os.path.join(write_path, "__init__.py"), "+") as f:
                    f_str = f.read()
                    f.write(env_str)
                    f.write(f_str)
            try:
                ldict: Dict[str, Any] = {}
                exec(exec_cmd, globals(), ldict)
                return True, msgs, ldict[return_key]
            except ImportError as e:
                msgs.append(f"Could not import {return_key} from given file. Error {e}"
                            + "\n" + traceback.format_exc())
                return False, msgs, None
            except Exception as e:
                msgs.append(f"Some weird error occured while importing {return_key}. Error {e}"
                            + "\n" + traceback.format_exc())
                return False, msgs, None

    @classmethod
    def _add_module(cls, mods_dir: str, module: bytes, name: Optional[str],
                    checks: Iterable[Callable[[str], bool]], overwrite: bool = False) ->\
            Tuple[bool, List[str], Optional[Dict]]:
        """Add an arbitrary module to some directory from a given stream of bytes

        One of either :code:`name` or :code:`module_exports` must be defined in
        the module. e.g, in case no :code:`name` is given::

            class Cls
                # some stuff here

            def some_stuff():
                # Do something here

            module_exports = {"abc": CLS, "some_stuff": some_stuff}

        The above code snippet exports :code:`CLS` as :code:`abc` and
        :code:`some_stuff` as :code:`some_stuff`, while if :code:`name` is
        given, then all the symbols are returend as is.

        If :code:`name` is not given then a random file (or dir) name is
        generated and the module is imported from there.

        Args:
            mods_dir: Directory where the module file will be written
            module: A stream of bytes. It should contain the module file.
                    The file can be either a python file in text format or a zipped
                    module. If it's a zipped module then :code:`__init__.py` must be at the top
                    level in the zip file.
            name: name of the module
            checks: Checks to conduct on the file.
            overwrite: Whether to overwrite the module in case name is given.

        Returns:
            A tuple of status, list of messages and module if successful or :code:`None`.

        """
        msgs: List[str] = []
        if not os.path.abspath(mods_dir) in sys.path:
            msgs.append(f"Modules path {mods_dir} was not in sys.path")
            sys.path.append(os.path.abspath(mods_dir))
        try:
            # file can be zip or text
            test = cls._check_file_magic(module, "python")
            if not name:
                name = "tmp_" + str(uuid.uuid4()).replace("-", "_")
                exec_cmd = f"from {name} import module_exports"
                return_key = "module_exports"
            else:
                exec_cmd = f"import {name}"
                return_key = name
            if test:
                py_file = os.path.join(mods_dir, name + ".py")
                if os.path.exists(py_file) and not overwrite:
                    return False, [f"Module with name {name} already exists. " +
                                   "Use overwrite=True to overwrite."], None
                else:
                    return cls._load_python_file(module, checks, py_file, exec_cmd, return_key)
            elif zipfile.is_zipfile(io.BytesIO(module)):
                mod_path = os.path.join(mods_dir, name)
                if os.path.exists(mod_path):
                    if not overwrite:
                        return False, [f"Module with name {name} already exists. " +
                                       "Use overwrite=True to overwrite."], None
                    else:
                        shutil.rmtree(mod_path)
                os.mkdir(mod_path)
                # exec_cmd = f"from {name} import module_exports"
                return cls._load_zip_file(module, checks, name, exec_cmd, return_key)
            else:
                msgs.append(f"Given file neither python nor zip.")
                return False, msgs, None
        except Exception as e:
            msgs.append(f"Error occured while reading file {e}"
                        + "\n" + traceback.format_exc())
            return False, msgs, None

    @classmethod
    def module_symbols(cls, mod_name: str, preds: List[Callable[[Any], bool]] = []) ->\
            Dict[str, Any]:
        """Read all the symbols from a given module in the module directory `mods_dir`.

        Args:
            mod_name: Module name
            preds: Return symbols for which all preds return True
        Returns:
            A :class:`dict` of symbol names and variables

        """
        exec_cmd = f"import {mod_name}"
        ldict: Dict[str, Any] = {}
        exec(exec_cmd, globals(), ldict)
        symbols = {x: y for x, y in ldict[mod_name].__dict__.items()
                   if not x.startswith("__") and all(p(y) for p in preds)}
        del ldict
        return symbols

    @classmethod
    def module_functions(cls, mod_name: str) -> Dict[str, Callable]:
        """Read all the symbols from a given module in the module directory `mods_dir`.

        Args:
            mod_name: Module name
        Returns:
            A :class:`dict` of function names and functions

        """
        return cls.module_symbols(mod_name, [lambda x: isinstance(x, FunctionType)])

    @classmethod
    def module_classes(cls, mod_name: str) -> Dict[str, type]:
        """Read all the symbols from a given module in the module directory `mods_dir`.

        Args:
            mod_name: Module name
        Returns:
            A :class:`dict` of class names and classes

        """
        return cls.module_symbols(mod_name, [lambda x: isinstance(x, type)])

    @classmethod
    def read_modules_from_dir(cls, mods_dir,
                              excludes: Iterable[Callable[[str], bool]] = [],
                              preds: Iterable[Callable[[str], bool]] = []) -> Dict[str, Any]:
        """Read all the symbols from available modules in the module directory `mods_dir`.

        Args:
            mods_dir: Module directory
            excludes: Exclude the given symbols from the return value
            preds: Return symbols for which all preds return True
        Returns:
            A dictionary of module and symbol names

        """
        modules_dict = {}
        if mods_dir not in sys.path:
            sys.path.append(str(mods_dir))
        py_mods = [x[:-3] for x in os.listdir(mods_dir) if x.endswith(".py")]
        dir_mods = [x for x in os.listdir(mods_dir) if not x.endswith(".py")
                    and os.path.isdir(os.path.join(mods_dir, x))]
        mods = [*py_mods, *dir_mods]
        # NOTE: Time taken may depend on load time of individual modules
        for m in mods:
            if not any(e(m) for e in excludes):
                modules_dict[m] = [*cls.module_symbols(m, preds).keys()]
        sys.path.remove(str(mods_dir))
        return modules_dict

    @property
    def mods_dir(self):
        return self._mods_dir

    @property
    def available_modules(self):
        return self.read_modules_from_dir(self.mods_dir,
                                          excludes=[lambda x: x.startswith(".")])

    @property
    def available_functions(self):
        return self.read_modules_from_dir(self.mods_dir,
                                          excludes=[lambda x: x.startswith(".")],
                                          preds=[lambda x: isinstance(x, FunctionType)])

    @property
    def available_classes(self):
        return self.read_modules_from_dir(self.mods_dir,
                                          excludes=[lambda x: x.startswith(".")],
                                          preds=[lambda x: isinstance(x, type)])

    @classmethod
    def add_config(cls, config_dir: Union[str, pathlib.Path], module: bytes,
                   env: Optional[str] = None, env_str: str = ""):
        """Load a :class:`Trainer` configuration

        Args:
            config_dir: The directory where the config resides.
                        It's appended to the :attr:`sys.path`
            module: Config file bytes
            env: Additional commands to prepend to the commad for loading the config
        Returns:
            A tuple of status, and message if fails or the config

        """
        test_py_file = cls._check_file_magic(module, "python")
        return_key = "config"
        checks: List[Callable] = []
        if config_dir not in sys.path:
            sys.path.append(str(config_dir))
        tmp_name = "session_config"
        if env:
            exec_cmd = env + "\n" + f"from session_config import config"
        else:
            exec_cmd = f"from session_config import config"
        if test_py_file:
            tmp_file = os.path.join(os.path.abspath(config_dir), tmp_name + ".py")
            status, msgs, retval = cls._load_python_file(module, checks, tmp_file,
                                                         exec_cmd, return_key,
                                                         env_str=env_str)
        else:
            tmp_path = os.path.join(os.path.abspath(config_dir), tmp_name)
            status, msgs, retval = cls._load_zip_file(module, checks, tmp_path,
                                                      exec_cmd, return_key,
                                                      env_str=env_str)
        if not status:
            return status, "\n".join(msgs)
        else:
            return status, retval

    # def add_module(self, module: bytes, name: Optional[str],
    #                checks: Iterable[Callable[[str], bool]]) ->\
    #         Tuple[bool, Union[str, Dict]]:
    #     """Add an arbitrary module from a given stream of bytes

    #     ``module_exports`` must be defined in the module. e.g.::

    #         class Cls
    #             # some stuff here

    #         def some_stuff():
    #             # Do something here

    #         module_exports = {"abc": CLS, "some_stuff": some_stuff}

    #     The above code exports ``CLS`` as ``abc`` and ``some_stuff`` as ``some_stuff``.

    #     Args:
    #         request: http request forwarded from daemon. Must contain the module file
    #                  The file can be either a python file in text format or a zipped
    #                  module. If it's a zipped module then ``__init__.py`` must be at the top
    #                  level in the zip file.
    #         name: name of the module
    #         checks: Checks to conduct on the file.
    #     Returns:
    #         A tuple of status, and message if fails or dictionary with the module.

    #     """
    #     if not os.path.abspath(self._mods_dir) in sys.path:
    #         self._logd(f"Modules path {self._mods_dir} was not in sys.path")
    #         sys.path.append(os.path.abspath(self._mods_dir))
    #     try:
    #         # file can be zip or text
    #         test = self._check_file_magic(module, "python")
    #         return_key = "module_exports"
    #         if test:
    #             if not name:
    #                 name = "tmp_" + str(uuid.uuid4()).replace("-", "_")
    #             tmp_file = os.path.join(self._mods_dir, name + ".py")
    #             exec_cmd = f"from {name} import module_exports"
    #             status, msgs, retval = self._load_python_file(module, checks, tmp_file,
    #                                                           exec_cmd, return_key)
    #             for msg in msgs:
    #                 self._logd(msg)
    #             if not status:
    #                 return status, msgs[-1]
    #             else:
    #                 return status, retval
    #         elif zipfile.is_zipfile(io.BytesIO(module)):
    #             tmp_dir = self.make_temp_directory(self._mods_dir)
    #             exec_cmd = f"from {tmp_dir} import module_exports"
    #             status, msgs, retval = self._load_zip_file(module, checks, tmp_dir,
    #                                                        exec_cmd, return_key)
    #             for msg in msgs:
    #                 self._logd(msg)
    #             if not status:
    #                 return status, msgs[-1]
    #             else:
    #                 return status, retval
    #         else:
    #             return False, self._logd(f"Given file neither python nor zip.")
    #     except Exception as e:
    #         return False, self._logd(f"Error occured while reading file {e}"
    #                                  + "\n" + traceback.format_exc())

    def add_module(self, module: bytes, name: Optional[str],
                   checks: Iterable[Callable[[str], bool]]) ->\
            Tuple[bool, Union[str, Dict]]:
        """Add an arbitrary module from a given stream of bytes to :attr:`mods_dir`

        See :meth:`_add_module` for details

        """
        status, msgs, retval = self._add_module(self._mods_dir, module, name, checks)
        for msg in msgs:
            self._logd(msg)
        if not status:
            return status, msgs[-1]
        else:
            return status, retval

    def add_named_module(self, mods_dir: Union[str, pathlib.Path], module: bytes,
                         module_name: str, overwrite: bool = False,
                         checks: List[Callable[..., bool]] = []) ->\
            Tuple[bool, Union[str, Dict[str, List[str]]]]:
        """Add an named module from a given stream of bytes to :attr:`mods_dir`

        See :meth:`_add_module` for details

        """
        status, msgs, retval = self._add_module(self._mods_dir, module, module_name,
                                                checks, overwrite)
        for msg in msgs:
            self._logd(msg)
        if not status:
            return status, msgs[-1]
        else:
            symbol_names = [x for x in retval.__dict__ if not x.startswith("__")]
            return True, {module_name: symbol_names}

    def add_user_funcs(self, module: bytes, user_funcs: Dict[str, Callable]) ->\
            Tuple[bool, str]:
        """Add a user function from a given python or module as a zip file.

        The request content is treated as a module and call :meth:`add_module`
        with additional checks.  Prospective rules for adding a user func:

        Args:
            request: http request forwarded from the daemon
            user_funcs: :attr:`trainer.Trainer._user_funcs` existing user functions
        Returns:
            A tuple of status, and message if fails or dictionary with the functions

        """
        # NOTE: In the python way, we can't really restrain whatever is in
        #       there, we can only warn and try to avoid it.
        #       Checks can then be like:
        #       1. "inspect." not in file
        #       2. "self." not in file
        checks: List[Callable[[str], bool]] = []
        addm_retval = self.add_module(module, None, checks)
        status: bool = addm_retval[0]
        response: Union[str, Dict] = addm_retval[1]
        if status:              # dependent type, not caught by mypy
            module_exports: Dict[str, Any] = response
            if "functions" not in module_exports:
                return False, self._logw("No functions in data")
            else:
                statuses = []
                for f in module_exports["functions"]:
                    # Arbitrary user func can be anything actually as long as it's a callable
                    if "name" in f.keys() and "function" in f.keys():
                        statuses.append((True, f["name"]))
                        user_funcs[f["name"]] = f["function"]
                    else:
                        statuses.append((False, f["name"]))
                if all(x[0] for x in statuses):
                    return True, self._logd("All functions added successfully")
                else:
                    retval = [str(x[1]) + " added, " if x[0] else " failed, " for x in statuses]
                    return False, self._logd(f"{retval}")
        else:
            return False, "Failed to add module"


    # # FIXME: This is redundant with just a name added
    # def add_named_module(self, mods_dir: Union[str, pathlib.Path], module: bytes,
    #                      module_name: str, overwrite: bool = False) ->\
    #         Tuple[bool, Union[str, Dict[str, List[str]]]]:
    #     test_py_file = self._check_file_magic(module, "python")
    #     return_key = module_name
    #     checks: List[Callable] = []
    #     if mods_dir not in sys.path:
    #         sys.path.append(str(mods_dir))
    #     if test_py_file:
    #         tmp_name = module_name
    #         tmp_file = os.path.join(os.path.abspath(mods_dir), tmp_name + ".py")
    #         if os.path.exists(tmp_file) and not overwrite:
    #             return False, f"Module with name {module_name} already exists. " +\
    #                 "Use overwrite=True to overwrite."
    #         exec_cmd = f"import {module_name}"
    #         mod = self._load_python_file(module, checks, tmp_file, exec_cmd, return_key)
    #     else:
    #         tmp_dir = module_name
    #         tmp_path = os.path.join(os.path.abspath(mods_dir), tmp_dir)
    #         if os.path.exists(tmp_path):
    #             if not overwrite:
    #                 return False, f"Module with name {module_name} already exists. " +\
    #                     "Use overwrite=True to overwrite."
    #             else:
    #                 shutil.rmtree(tmp_path)
    #         os.mkdir(tmp_path)
    #         exec_cmd = f"import {module_name}"
    #         mod = self._load_zip_file(module, checks, tmp_path, exec_cmd, return_key)
    #     sys.path.remove(str(mods_dir))
    #     if mod[0]:
    #         symbol_names = [x for x in mod[1].__dict__ if not x.startswith("__")]
    #         return True, {module_name: symbol_names}
    #     else:
    #         return mod
