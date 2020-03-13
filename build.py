import os
import shlex
import sys
import shutil
from subprocess import run


file_list = ["version.py",
             "_log.py",
             "mods.py",
             "util.py",
             "device.py",
             "task.py",
             "epoch.py",
             "overrides.py",
             "components.py",
             "_checks.py",
             "helpers.py",
             "trainer.py",
             "interfaces.py",
             "daemon.py"]


def clobberize(entry_file):
    "Give only one file and it recursively clobbers it up, only existing and relative modules"
    pass


def main():
    # if len(sys.argv) < 2:
    #     print("Need file to coalesce")
    # else:
    #     clobberize(sys.argv[1])
    if os.path.exists("build"):
        print("Cleaning build directory")
        shutil.rmtree("build")
    files_dir = "trainer"
    str_list = []
    for fname in file_list:
        with open(os.path.join(files_dir, fname)) as f:
            line = f.readline()
            while line:
                if line.startswith("import .") or line.startswith("from ."):
                    if "(" in line:
                        while ")" not in line:
                            line = f.readline()
                else:
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


def copy_tests():
    if "build" in os.path.abspath(os.curdir):
        os.chdir("..")
    files = [x for x in os.listdir("tests") if (x.startswith("test_")
                                                and x.endswith(".py")
                                                and "check" not in x) or
             "_setup" in x]
    os.mkdir("build/tests")
    for f in files:
        shutil.copy(f"tests/{f}", f"build/tests/{f}")


if __name__ == '__main__':
    main()
    copy_tests()
