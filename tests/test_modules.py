import os
import unittest
import zipfile
import sys
sys.path.append("../")
from trainer.mods import Modules


class ModulesTest(unittest.TestCase):
    def setUp(self):
        self.mod_string_a = """
class A:
    def __init__(self):
        pass


def func(a, b):
    return a + b

module_exports = {"cls": A, "f": func}
"""
        self.mod_string_b = """
class A:
    def __init__(self):
        pass

def func(a, b):
    return a + b
"""
        self.mod_string_c = """
class B:
    def __init__(self):
        pass

def func(a, b):
    return a + b
"""
        self.mod_string_i = """
from .mod_b import A, func as f
from .mod_c import B, func as g

module_exports = {"A": A, "B": B, "f": f, "g": g}
"""
        with open("/tmp/mod_a.py", "w") as f:
            f.write(self.mod_string_a)
        with open("/tmp/mod_b.py", "w") as f:
            f.write(self.mod_string_b)
        with open("/tmp/mod_c.py", "w") as f:
            f.write(self.mod_string_c)
        with open("/tmp/__init__.py", "w") as f:
            f.write(self.mod_string_i)
        self.fnames = {"a": "/tmp/mod_a.py", "b": "/tmp/mod_b.py", "c": "/tmp/mod_c.py",
                       "i": "/tmp/__init__.py"}

        def print_func(x):
            print(x)
        self._modules = Modules("/tmp/meh_modules", print_func, print_func,
                                print_func, print_func)

    def test_load_py_file(self):
        # TODO: check file loading and errors
        # 1. file not python file
        # 2. file python file but loading error
        with open(self.fnames["a"], "rb") as f:
            meh = f.read()
        status, result = self._modules._load_python_file(meh, [], "_meh.py",
                                                         "from _meh import module_exports",
                                                         "module_exports")
        self.assertTrue(status)
        self.assertTrue("cls" in result)
        self.assertTrue("f" in result)
        os.remove("_meh.py")

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


if __name__ == '__main__':
    unittest.main()
