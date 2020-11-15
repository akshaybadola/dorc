import os
from setuptools import setup
from Cython.Build import cythonize

os.environ["CFLAGS"] = "-fno-var-tracking-assignments"
setup(
    ext_modules=cythonize('trainer.py', compiler_directives={"language_level": "3"})
)
