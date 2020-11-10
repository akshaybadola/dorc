from typing import Callable, Iterable, Tuple, Dict, Any, List, Union
import os
import io
import sys
import uuid
import magic
import zipfile
import traceback

# imports needed for typing
import flask
import pathlib


class Modules:
    """Dynamic loading and unloading of python modules.

    If two modules with the same name are imported, the previous one is
    overwritten. A global list of modules is maintained.

    The modules are listed as file/def where def is any variable definition:
    function, class or global variable. They're simply imported as file.var.

    A bit of the code is old in the sense that :mod:`importlib` has changed to
    allow loading python modules without a file.

    """
    def __init__(self, mods_dir: str, logd: Callable[[str], str],
                 loge: Callable[[str], str], logi: Callable[[str], str],
                 logw: Callable[[str], str]):
        self._mods_dir = mods_dir
        self._logd = logd
        self._loge = loge
        self._logi = logi
        self._logw = logw
        if not os.path.exists(mods_dir):
            self._logd(f"Creating directory {self._mods_dir}")
            os.mkdir(mods_dir)

    def _check_file_magic(self, _file: bytes, test_str: str) -> bool:
        """Determine the file type from `libmagic`

        Args:
            _file: Bytes of the file
            test_str: String to test for in the magic output
        Returns:
            True or False according to whether string is in magic output

        """
        if hasattr(magic, "from_buffer"):
            self._logd("from_buffer in magic")
            test = test_str in magic.from_buffer(_file).lower()
        elif hasattr(magic, "detect_from_content"):
            self._logd("detect_from_content in magic")
            test = test_str in magic.detect_from_content(_file).name.lower()
        return test

    def make_temp_directory(self) -> str:
        dirname = "tmp_" + str(uuid.uuid4()).replace("-", "_")
        os.mkdir(os.path.join(self._mods_dir, dirname))
        return dirname

    def _load_python_file(self, module_file: bytes, checks: Iterable[Callable[[str], bool]],
                          write_path: Union[str, pathlib.Path], exec_cmd: str, return_key: str) ->\
                          Tuple[bool, Union[str, Dict]]:
        """Load a python file containing a module.

        Args:
            module_file: A stream of bytes. It's written on the disk and module loaded.
            checks: An Iterable of functions which checks the loaded module.
            write_path: Path to which the file is written
            exec_cmd: Command to execute for loading
            return_key: The key of the object to return
        Returns:
            A tuple of status, and message if fails or dictionary with the module.

        """
        self._logd("Detected python file")
        with open(write_path, "w") as f:
            print(self._logd(f"Written to {write_path}"))
            f.write(module_file.decode())
        try:
            ldict: Dict[str, Any] = {}
            self._logd("Checking functions")
            flag = True
            for check_p in checks:
                if not check_p(write_path):
                    flag = False
                    break
            if not flag:
                return False, self._logd(f"Module check failed {check_p}")
            else:
                self._logd(f"Executing {exec_cmd}")
                exec(exec_cmd, globals(), ldict)
                return True, ldict[return_key]
        except ImportError as e:
            return False, self._logd(f"Could not import module_exports from given file. Error {e}"
                                     + "\n" + traceback.format_exc())
        except Exception as e:
            return False, self._logd(f"Some weird error occured while importing. Error {e}"
                                     + "\n" + traceback.format_exc())

    def _load_zip_file(self, module_file: bytes, checks: Iterable[Callable[[str], bool]],
                       write_path: Union[str, pathlib.Path], exec_cmd: str, return_key: str) ->\
                       Tuple[bool, Union[str, Dict]]:
        """Load a zip file containing a module.

        The file should contain an __init__.py at the top level and that should
        contain all the definitions which should be available.  The zip file is
        extracted to a folder and loaded as a directory.

        Args:
            module_file: A stream of bytes. It's written on the disk and module loaded.
            checks: An Iterable of functions which checks the loaded module.
            write_path: Path to which the file is written
            exec_cmd: Command to execute for loading
            return_key: The key of the object to return
        Returns:
            A tuple of status, and message if fails or dictionary with the module.

        """
        zf = zipfile.ZipFile(io.BytesIO(module_file))
        # make sure that __init__.py is at the root of tmp_dir
        print("FILE NAME list", zf.namelist())
        if not any(["__init__.py" in x.split("/")[0] for x in zf.namelist()]):
            return False, self._logd(f"zip file must have __init__.py at the top level")
        else:
            # zf.extractall(os.path.join(self._mods_dir, write_path))
            print(self._logd(f"Extracting to {write_path}"))
            zf.extractall(write_path)
            try:
                ldict: Dict[str, Any] = {}
                print(self._logd(f"Executing {exec_cmd}"))
                exec(exec_cmd, globals(), ldict)
                return True, ldict[return_key]
            except ImportError as e:
                return False, self._logd(f"Could not import {return_key} from given file. Error {e}"
                                         + "\n" + traceback.format_exc())
            except Exception as e:
                return False, self._logd(f"Some weird error occured while importing {return_key}. Error {e}"
                                         + "\n" + traceback.format_exc())

    def add_module(self, request: flask.Request, checks: Iterable[Callable[[str], bool]]) ->\
            Tuple[bool, Union[str, Dict]]:
        """Add an arbitrary module from a given python or module as a zip file. File
        must be present in request and is read as ``request.files["file"]``

        ``module_exports`` must be defined in the module. e.g.::

            class Cls
                # some stuff here

            def some_stuff():
                # Do something here

            module_exports = {"abc": CLS, "some_stuff": some_stuff}

        The above code exports ``CLS`` as ``abc`` and ``some_stuff`` as ``some_stuff``.

        Args:
            request: http request forwarded from daemon. Must contain the module file
                     The file can be either a python file in text format or a zipped
                     module. If it's a zipped module then ``__init__.py`` must be at the top
                     level in the zip file.
            checks: Checks to conduct on the file.
        Returns:
            A tuple of status, and message if fails or dictionary with the module.

        """
        if not os.path.abspath(self._mods_dir) in sys.path:
            self._logd(f"Modules path {self._mods_dir} was not in sys.path")
            sys.path.append(os.path.abspath(self._mods_dir))
        try:
            # file can be zip or text
            module_file: bytes = request.files["file"].read()
            test = self._check_file_magic(module_file, "python")
            return_key = "module_exports"
            if test:
                tmp_name = "tmp_" + str(uuid.uuid4()).replace("-", "_")
                tmp_file = os.path.join(self._mods_dir, tmp_name + ".py")
                exec_cmd = f"from {tmp_name} import module_exports"
                return self._load_python_file(module_file, checks, tmp_file, exec_cmd, return_key)
            elif zipfile.is_zipfile(io.BytesIO(module_file)):
                tmp_dir = self.make_temp_directory()
                exec_cmd = f"from {tmp_dir} import module_exports"
                return self._load_zip_file(module_file, checks, tmp_dir, exec_cmd, return_key)
            else:
                return False, self._logd(f"Given file neither python nor zip.")
        except Exception as e:
            return False, self._logd(f"Error occured while reading file {e}"
                                     + "\n" + traceback.format_exc())

    def add_user_funcs(self, request: flask.Request, user_funcs) -> Tuple[bool, str]:
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
        addm_retval = self.add_module(request, checks)
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

    def add_config(self, config_dir: Union[str, pathlib.Path], module_file: bytes,
                   env: None = None):
        """Load a :class:`Trainer` configuration

        Args:
            config_dir: The directory where the config resides. It's append to the :attr:`sys.path`
            module_file: Config file bytes
            env: Additional commands to prepend to the commad for loading the config
        Returns:
            A tuple of status, and message if fails or the config

        """
        test_py_file = self._check_file_magic(module_file, "python")
        return_key = "config"
        checks = []
        if config_dir not in sys.path:
            sys.path.append(config_dir)
        tmp_dir = tmp_name = "session_config"
        if env:
            exec_cmd = env + "\n" + f"from session_config import config"
        else:
            exec_cmd = f"from session_confing import config"
        if test_py_file:
            tmp_file = os.path.join(os.path.abspath(config_dir), tmp_name + ".py")
            return self._load_python_file(module_file, checks, tmp_file, exec_cmd, return_key)
        else:
            tmp_path = os.path.join(os.path.abspath(config_dir), tmp_dir)
            return self._load_zip_file(module_file, checks, tmp_path, exec_cmd, return_key)
        # NOTE: I'm sure this code is unreachable
        # sys.path.remove(config_dir)

    def add_named_module(self, mods_dir: Union[str, pathlib.Path], module_file: bytes,
                         module_name: str) -> Tuple[bool, str]:
        test_py_file = self._check_file_magic(module_file, "python")
        return_key = module_name
        checks = []
        if mods_dir not in sys.path:
            sys.path.append(mods_dir)
        if test_py_file:
            tmp_name = module_name
            tmp_file = os.path.join(os.path.abspath(mods_dir), tmp_name + ".py")
            exec_cmd = f"import {module_name}"
            mod = self._load_python_file(module_file, checks, tmp_file, exec_cmd, return_key)
        else:
            tmp_dir = module_name
            tmp_path = os.path.join(os.path.abspath(mods_dir), tmp_dir)
            os.mkdir(tmp_path)
            print("WRITE PATH", mods_dir, tmp_path, module_name)
            exec_cmd = f"import {module_name}"
            mod = self._load_zip_file(module_file, checks, tmp_path, exec_cmd, return_key)
        sys.path.remove(mods_dir)
        if mod[0]:
            symbol_names = [x for x in mod[1].__dict__ if not x.startswith("__")]
            return True, {module_name: symbol_names}
        else:
            return mod

    def _get_symbols_from_module(self, mod_name: str) -> Iterable:
        """Read all the symbols from a given module in the module directory `mods_dir`.

        Args:
            mod_name: Module name
        Returns:
            A list of symbol names

        """
        exec_cmd = f"import {mod_name}"
        ldict = {}
        exec(exec_cmd, globals(), ldict)
        symbol_names = [x for x in ldict[mod_name].__dict__
                        if not x.startswith("__")]
        del ldict
        return symbol_names

    def read_modules_from_dir(self, mods_dir: Union[pathlib.Path, str],
                              excludes: Iterable[Callable[[str], bool]]) -> Dict:
        """Read all the symbols from available modules in the module directory `mods_dir`.

        Args:
            mods_dir: Module directory
            excludes: Exclude the given symbols from the return value
        Returns:
            A dictionary of module and symbol names

        """
        modules_dict = {}
        if mods_dir not in sys.path:
            sys.path.append(mods_dir)
        py_mods = [x[:-3] for x in os.listdir(mods_dir) if x.endswith(".py")]
        dir_mods = [x for x in os.listdir(mods_dir) if not x.endswith(".py")
                    and os.path.isdir(os.path.join(mods_dir, x))]
        mods = [*py_mods, *dir_mods]
        # NOTE: Time taken may depend on load time of individual modules
        for m in mods:
            if not any(e(m) for e in excludes):
                modules_dict[m] = self._get_symbols_from_module(m)
        sys.path.remove(mods_dir)
        return modules_dict
