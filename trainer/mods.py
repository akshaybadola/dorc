import os
import io
import sys
import uuid
import magic
import zipfile


class Modules:
    def __init__(self, mods_dir, logd, loge, logi, logw):
        self._mods_dir = mods_dir
        if not os.path.exists(mods_dir):
            os.mkdir(mods_dir)
        self._logd = logd
        self._loge = loge
        self._logi = logi
        self._logw = logw

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

    def add_module(self, request, checks):
        """Adds an arbitrary module from a given python or module as a zip file. File
        must be present in request and is read as ``request.files["file"]``

        The file can be either a python file in text format or a zipped
        module. If it's a zipped module then ``__init__.py`` must be at the top
        level in the zip file.

        exports must be defined in ``module_exports``. e.g.:

        .. code-block :: py

            class ABC
                # some stuff here

            def some_stuff():
                # Do something here

            module_exports = {"abc": ABC, "some_stuff"}

        """
        # TODO: Should be a configurable paramter self.modules_dir
        if not os.path.exists("trainer_modules"):
            self._logd("Creating directory trainer_modules")
            os.mkdir("trainer_modules")
        if not os.path.abspath("trainer_modules") in sys.path:
            self._logd("Modules path was not in sys")
            sys.path.append(os.path.abspath("trainer_modules"))
        try:
            # file can be zip or text
            model_file = request.files["file"].read()
            test = self._check_file_magic(model_file, "python")
            if test:
                self._logd("Detected python file")
                tmp_name = "tmp_" + str(uuid.uuid4()).replace("-", "_")
                tmp_file = os.path.join("trainer_modules", tmp_name + ".py")
                with open(tmp_file, "w") as f:
                    self._logd(f"Written to {tmp_file}")
                    f.write(model_file.decode())
                try:
                    ldict = {}
                    self._logd("Checking functions")
                    flag = True
                    for check_p in checks:
                        if not check_p(tmp_file):
                            flag = False
                            break
                    if not flag:
                        return False, self._logd(f"Module check failed {check_p}")
                    else:
                        self._logd(f"Executing 'from {tmp_name} import module_exports'")
                        exec(f"from {tmp_name} import module_exports", globals(), ldict)
                        return True, ldict["module_exports"]
                except ImportError as e:
                    return False, self._logd(f"Could not import module_exports from given file. Error {e}")
                except Exception as e:
                    return False, self._logd(f"Some weird error occured while importing. Error {e}")
            elif zipfile.is_zipfile(io.BytesIO(model_file)):
                zf = zipfile.ZipFile(io.BytesIO(model_file))
                # make sure that __init__.py is at the root of tmp_dir
                if not any(["__init__.py" in x.split("/")[0] for x in zf.namelist()]):
                    return False, self._logd(f"zip file must have __init__.py at the top level")
                else:
                    tmp_dir = self.make_temp_directory()
                    zf.extractall(os.path.join("trainer_modules", tmp_dir))
                    try:
                        ldict = {}
                        exec(f"from {tmp_dir} import module_exports", globals(), ldict)
                        return True, ldict["module_exports"]
                    except ImportError as e:
                        return False, self._logd(f"Could not import module_exports from given file. Error {e}")
                    except Exception as e:
                        return False, self._logd(f"Some weird error occured while importing. Error {e}")
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
