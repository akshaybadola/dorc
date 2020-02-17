import os
import io
import sys
import uuid
import magic
import zipfile


class Modules:
    def __init__(self, mods_dir, logd, loge, logi, logw):
        self._mods_dir = mods_dir
        self._logd = logd
        self._loge = loge
        self._logi = logi
        self._logw = logw
        if not os.path.exists(mods_dir):
            self._logd(f"Creating directory {self._mods_dir}")
            os.mkdir(mods_dir)

    def _check_file_magic(self, _file, test_str):
        if hasattr(magic, "from_buffer"):
            self._logd("from_buffer in magic")
            test = test_str in magic.from_buffer(_file).lower()
        elif hasattr(magic, "detect_from_content"):
            self._logd("detect_from_content in magic")
            test = test_str in magic.detect_from_content(_file).name.lower()
        return test

    def make_temp_directory(self):
        dirname = "tmp_" + str(uuid.uuid4()).replace("-", "_")
        os.mkdir(os.path.join(self._mods_dir, dirname))
        return dirname

    def _load_python_file(self, module_file, checks, write_path, exec_cmd, return_key):
        self._logd("Detected python file")
        with open(write_path, "w") as f:
            self._logd(f"Written to {write_path}")
            f.write(module_file.decode())
        try:
            ldict = {}
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
            return False, self._logd(f"Could not import module_exports from given file. Error {e}")
        except Exception as e:
            return False, self._logd(f"Some weird error occured while importing. Error {e}")

    def _load_zip_file(self, module_file, checks, write_path, exec_cmd, return_key):
        zf = zipfile.ZipFile(io.BytesIO(module_file))
        # make sure that __init__.py is at the root of tmp_dir
        if not any(["__init__.py" in x.split("/")[0] for x in zf.namelist()]):
            return False, self._logd(f"zip file must have __init__.py at the top level")
        else:
            # zf.extractall(os.path.join(self._mods_dir, write_path))
            zf.extractall(write_path)
            try:
                ldict = {}
                self._logd(f"Executing {exec_cmd}")
                exec(exec_cmd, globals(), ldict)
                return True, ldict[return_key]
            except ImportError as e:
                return False, self._logd(f"Could not import module_exports from given file. Error {e}")
            except Exception as e:
                return False, self._logd(f"Some weird error occured while importing. Error {e}")

    def add_module(self, request, checks):
        """Adds an arbitrary module from a given python or module as a zip file. File
        must be present in request and is read as ``request.files["file"]``

        The file can be either a python file in text format or a zipped
        module. If it's a zipped module then ``__init__.py`` must be at the top
        level in the zip file.

        exports must be defined in ``module_exports``. e.g.:

        .. code-block :: py

            class CLS
                # some stuff here

            def some_stuff():
                # Do something here

            module_exports = {"abc": CLS, "some_stuff": some_stuff}

        """
        if not os.path.abspath(self._mods_dir) in sys.path:
            self._logd(f"Modules path {self._mods_dir} was not in sys.path")
            sys.path.append(os.path.abspath(self._mods_dir))
        try:
            # file can be zip or text
            module_file = request.files["file"].read()
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
            return False, self._logd(f"Error occured while reading file {e}")

    def add_user_funcs(self, request, user_funcs):
        """Add a user function from a given python or module as a zip file. Delegates
        the requeste to :meth:`Trainer.add_module`

        Prospective rules for adding a user func

        1. `user_func` shouldn't be given access to the trainer instance itself as
        it may cause unwanted states

        2. `hooks` may be specific functions which can be given access to specific
        locals of particular functions

        3. `user_func` is the most generic function and has arbitrary call and
        return values

        4. In constrast `hooks` would be more restriced and while adding or
        removing a hook certain checks can be performed

        5. `user_funcs` maybe invoked in the middle of the program somewhere as
        defined, or can be invoked in any situation parallelly. While other
        functions may only execute in certain cricumstances

        :param request: :func:`flask.request`

        """
        # NOTE: In the python way, we can't really restrain whatever is in
        #       there, we can only warn and try to avoid it.
        #       Checks can then be like:
        #       1. "inspect." not in file
        #       2. "self." not in file
        checks = []
        status, response = self.add_module(request, checks)
        if status:
            module_exports = response
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

    def add_config(self, config_dir, module_file):
        test_py_file = self._check_file_magic(module_file, "python")
        return_key = "config"
        checks = []
        if config_dir not in sys.path:
            sys.path.append(config_dir)
        if test_py_file:
            tmp_name = "session_config"
            tmp_file = os.path.join(os.path.abspath(config_dir), tmp_name + ".py")
            exec_cmd = f"from {tmp_name} import config"
            return self._load_python_file(module_file, checks, tmp_file, exec_cmd, return_key)
        else:
            tmp_dir = "session_config"
            tmp_path = os.mkdir(os.path.join(os.path.abspath(config_dir), tmp_dir))
            exec_cmd = f"from {tmp_dir} import config"
            return self._load_zip_file(module_file, checks, tmp_path, exec_cmd, return_key)
        sys.path.remove(config_dir)
