import os
import argparse
import shlex
from subprocess import Popen, PIPE, run

from build import main as build


def _exists(host, filename, switch):
    p = Popen(f"ssh {host}  [[ {switch} {filename} ]] ; echo $?",
              shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return "0" in out.decode("utf-8").split("\n")[0] and not len(err.decode("utf-8"))


def file_exists(host, filename):
    return _exists(host, filename, "-f")


def dir_exists(host, filename):
    return _exists(host, filename, "-d")


def create_dir(host, dirname):
    p = Popen(shlex.split(f"ssh {host} mkdir -p {dirname}"),
              stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return not len(err.decode("utf-8"))


def exec_cmd(host, cmd):
    p = Popen(f"ssh {host} {cmd}", shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return out, err


def deploy(host, _init=True):
    if _init:
        init(host)
    run(f"scp build/trainer.cpython-37m-x86_64-linux-gnu.so {host}:~/trainer/", shell=True)
    run(f"scp trainer/autoloads.py {host}:~/trainer/", shell=True)
    run(f"scp dist/* {host}:~/trainer/dist/", shell=True)


def update_venv(host):
    if not dir_exists(host, "~/trainer/env"):
        create_dir(host, "~/trainer/env")
        exec_cmd(host, "python3.7 -m virtualenv ~/trainer/env")
    run(f"scp deploy_scripts/requirements.txt {host}:~/trainer/", shell=True)
    run(f"scp deploy_scripts/if_run.py {host}:~/trainer/", shell=True)
    run(f"scp deploy_scripts/run.py {host}:~/trainer/", shell=True)
    run(f"scp ~/bin/myauth {host}:~/", shell=True)
    exec_cmd(host, "python3 ~/myauth; rm ~/myauth")
    exec_cmd(host, "source ~/trainer/env/bin/activate; pip install -r ~/trainer/requirements.txt")


def init(host):
    if not dir_exists(host, "~/trainer"):
        create_dir(host, "~/trainer")
        create_dir(host, "~/trainer/dist")
    update_venv(host)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=lambda x: x.split(","),
                        default="mc15pc15@10.5.0.96,prototype@10.5.1.93,user@10.5.0.92")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    curdir = os.path.abspath(os.curdir)
    if args.update:
        build()
    os.chdir(curdir)
    for host in args.hosts:
        deploy(host)


if __name__ == '__main__':
    main()
