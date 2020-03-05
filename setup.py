from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# extensions = (
#     Extension('daemon', [
#         # 'trainer/__init__.py',
#         # "trainer/check.py",
#         # "trainer/_checks.py",
#         # "trainer/checks.py",
#         # "trainer/components.py",
#         "trainer/daemon.py",
#         # "trainer/device.py",
#         # "trainer/epoch.py",
#         # "trainer/functions.py",
#         # "trainer/helpers.py",
#         # "trainer/__init__.py",
#         # "trainer/interfaces.py",
#         # "trainer/_log.py",
#         # "trainer/mods.py",
#         # "trainer/overrides.py",
#         # "trainer/__pycache__",
#         # "trainer/simple_trainer.py",
#         # "trainer/task.py",
#         # "trainer/trainer.py",
#         # "trainer/util.py",
#         # "trainer/version.py"
#     ])
# )

# setup(
#     name='daemon',
#     ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"})
# )

setup(
    ext_modules=cythonize('trainer.py', compiler_directives={"language_level": "3"})
)
