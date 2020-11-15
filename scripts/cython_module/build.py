import os
import shlex
import shutil
from subprocess import run


file_list = ["version.py",
             "auth.py",
             "_log.py",
             "mods.py",
             "util.py",
             "device.py",
             "task.py",
             "epoch.py",
             "overrides.py",
             "model.py",
             "_checks.py",
             "helpers.py",
             "trainer.py",
             "interfaces.py",
             "daemon.py"]


def main(files_dir):
    if os.path.exists("build"):
        print("Cleaning build directory")
        shutil.rmtree("build")
    str_list = []
    print("Generating single trainer.py")
    for fname in file_list:
        with open(os.path.join(files_dir, fname)) as f:
            line = f.readline()
            while line:
                if line.startswith("import .") or line.startswith("from ."):
                    if "(" in line:
                        while ")" not in line:
                            line = f.readline()
                else:
                    # NOTE: An attempt to strip out pdb/ipdb
                    # if "pdb" or "ipdb" in line:
                    #     str_list.append("# " + line)
                    # else:
                    #     str_list.append(line)
                    str_list.append(line)
                line = f.readline()
    os.mkdir("build")
    with open("build/trainer.py", "w") as f:
        f.writelines(str_list)
    shutil.copy("setup.py", "build")
    # NOTE: switch to build dir
    os.chdir("build")
    cmd = "python setup.py build_ext --inplace"
    run(shlex.split(cmd), env=os.environ, cwd=os.curdir)


def copy_tests(tests_dir):
    os.chdir(tests_dir)
    files = [x for x in os.listdir(tests_dir) if (x.startswith("test_")
                                                  and x.endswith(".py")
                                                  and "check" not in x) or
             "_setup" in x]
    os.mkdir("build/tests")
    for f in files:
        shutil.copy(f"tests/{f}", f"build/tests/{f}")


if __name__ == '__main__':
    files_dir = os.path.abspath("../../trainer")
    tests_dir = os.path.abspath("../../tests")
    main(files_dir)
    copy_tests(tests_dir)
