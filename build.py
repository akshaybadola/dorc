import os
import shutil


def clobberize(entry_file):
    "Give only one file and it recursively clobbers it up, only existing and relative modules"
    pass


def main():
    # if len(sys.argv) < 2:
    #     print("Need file to coalesce")
    # else:
    #     clobberize(sys.argv[1])
    if os.path.exists("build"):
        print("Cleaning")
        shutil.rmtree("build")
    os.mkdir("build")
    shutil.copy("clobberize", "build/")

    # switch to build dir
    os.chdir("build")


if __name__ == '__main__':
    main()
