import os
import shutil
import unittest
import zipfile
import sys
sys.path.append("../")
from trainer.mods import Modules


class ModulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod_string_a = """
class A:
    def __init__(self):
        pass


def func(a, b):
    return a + b

module_exports = {"cls": A, "f": func}
"""
        cls.mod_string_b = """
class A:
    def __init__(self):
        pass

def func(a, b):
    return a + b
"""
        cls.mod_string_c = """
class B:
    def __init__(self):
        pass

def func(a, b):
    return a + b
"""
        cls.mod_string_i = """
from .mod_b import A, func as f
from .mod_c import B, func as g

module_exports = {"A": A, "B": B, "f": f, "g": g}
"""
        cls.mods_dir = ".test_mods_dir"
        if os.path.exists(cls.mods_dir):
            shutil.rmtree(cls.mods_dir)
        os.mkdir(cls.mods_dir)
        with open(os.path.join(cls.mods_dir, "mod_a.py"), "w") as f:
            f.write(cls.mod_string_a)
        with open(os.path.join(cls.mods_dir, "mod_b.py"), "w") as f:
            f.write(cls.mod_string_b)
        with open(os.path.join(cls.mods_dir, "mod_c.py"), "w") as f:
            f.write(cls.mod_string_c)
        with open(os.path.join(cls.mods_dir, "__init__.py"), "w") as f:
            f.write(cls.mod_string_i)
        cls.fnames = {"a": os.path.join(cls.mods_dir, "mod_a.py"),
                      "b": os.path.join(cls.mods_dir, "mod_b.py"),
                      "c": os.path.join(cls.mods_dir, "mod_c.py"),
                      "i": os.path.join(cls.mods_dir, "__init__.py")}

        def print_func(x):
            print(x)
        cls._modules = Modules(os.path.join(cls.mods_dir, "meh_modules"),
                               print_func, print_func, print_func, print_func)

    def test_load_py_file(self):
        # TODO: check file loading and errors
        # 1. file not python file
        # 2. file python file but loading error
        with open(self.fnames["a"], "rb") as f:
            meh = f.read()
        status, result = self._modules._load_python_file(meh, [], "_meh_a.py",
                                                         "from _meh_a import module_exports",
                                                         "module_exports")
        self.assertTrue(status)
        self.assertTrue("cls" in result)
        self.assertTrue("f" in result)
        os.remove("_meh_a.py")

    def test_load_zip_file(self):
        # TODO: check zip file loading and errors
        # 1. non zip file given
        # 2. wrong directory structure
        # 3. files inside zip not python
        with zipfile.ZipFile("bleh.zip", "w") as f:
            for x in {"b", "c", "i"}:
                fname = self.fnames[x]
                f.write(fname, arcname=os.path.basename(fname))
        with open("bleh.zip", "rb") as f:
            meh = f.read()
        status, result = self._modules._load_zip_file(meh, [], "_meh_dir",
                                                      "from _meh_dir import module_exports",
                                                      "module_exports")

    def test_add_config(self):
        with open("_setup.py", "rb") as f:
            meh = f.read()
        status, result = self._modules._load_python_file(meh, [], "_meh.py",
                                                         "from _meh import config",
                                                         "config")
        self.assertTrue(status)
        self.assertTrue("optimizer" in result)
        self.assertTrue("criteria" in result)

    def test_add_module(self):
        # add_py_file, zip_file, check for failures
        pass

    def test_add_named_module(self):
        for letter in self.fnames:
            if letter != "i":
                with open(self.fnames[letter], "rb") as f:
                    file_bytes = f.read()
                    result = self._modules.add_named_module(self.mods_dir,
                                                            file_bytes, "mod_" + letter)
                    self.assertTrue(result[0])
                    print(result[1])

    def test_get_symbols_from_module(self):
        mod_string = """
import types

class A:
    def __init__(self):
        pass


def func(a, b):
    return a + b

_test_var = {"cls": A, "f": func}
__invisible_var = {"Meow": "meow"}
sname = types.SimpleNamespace

module_exports = {"cls": A, "f": func}
"""
        with open(os.path.join(self.mods_dir, "test_symbols.py"), "w") as f:
            f.write(mod_string)
        if self.mods_dir not in sys.path:
            sys.path.append(self.mods_dir)
        symbol_names = self._modules._get_symbols_from_module("test_symbols")
        valid_names = ["A", "func", "_test_var", "sname", "module_exports"]
        invalid_names = ["__invisible_var", "__spec__", "__file__", "__dict__"]
        for v in valid_names:
            with self.subTest(i=v):
                self.assertTrue(v in symbol_names)
        for v in invalid_names:
            with self.subTest(i=v):
                self.assertFalse(v in symbol_names)

    def test_read_modules_from_dir(self):
        mod_string = """
import types

class A:
    def __init__(self):
        pass


def func(a, b):
    return a + b

_test_var = {"cls": A, "f": func}
__invisible_var = {"Meow": "meow"}
sname = types.SimpleNamespace

module_exports = {"cls": A, "f": func}
"""
        mods_dir = os.path.join(self.mods_dir, "mods_dir")
        if not os.path.exists(mods_dir):
            os.mkdir(os.path.join(self.mods_dir, "mods_dir"))
        with open(os.path.join(mods_dir, "test_symbols.py"), "w") as f:
            f.write(mod_string)
        if os.path.exists(os.path.join(mods_dir, "temp_mod")):
            shutil.rmtree(os.path.join(mods_dir, "temp_mod"))
        os.mkdir(os.path.join(mods_dir, "temp_mod"))
        for fname in self.fnames.values():
            shutil.copy(fname, os.path.join(mods_dir, "temp_mod"))
        modules = self._modules.read_modules_from_dir(mods_dir)
        print(modules)
        self.assertTrue("temp_mod" in modules)
        self.assertTrue("test_symbols" in modules)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.mods_dir):
            shutil.rmtree(cls.mods_dir)

    
if __name__ == '__main__':
    unittest.main()
