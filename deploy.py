import os
import re
import time
import multiprocessing as mp
import argparse
import shutil
import shlex
from subprocess import Popen, PIPE, run

from build import main as build


def _exists(host, filename, switch):
    p = Popen(f"ssh {host}  '[[ {switch} {filename} ]] ; echo $?'",
              shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return "0" in out.decode("utf-8").split("\n")[0] and not len(err.decode("utf-8"))


def file_exists(host, filename):
    return _exists(host, filename, "-f")


def dir_exists(host, filename):
    return _exists(host, filename, "-d")


def create_dir(host, dirname):
    p = Popen(shlex.split(f"ssh {host} 'mkdir -p {dirname}'"),
              stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return not len(err.decode("utf-8"))


def exec_cmd(host, cmd):
    p = Popen(f"ssh {host} '{cmd}'", shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return out, err


def deploy(host, _init=False, copy_js=False, clean_env=False, _restart=False):
    print(f"Deploying to host {host}")
    if _init or not dir_exists(host, "~/trainer"):
        init(host, clean_env)
    run(f"scp build/trainer.cpython-37m-x86_64-linux-gnu.so {host}:~/trainer/", shell=True)
    run(f"scp trainer/autoloads.py {host}:~/trainer/", shell=True)
    run(f"scp deploy_scripts/if_run.py {host}:~/trainer/", shell=True)
    run(f"scp deploy_scripts/run.py {host}:~/trainer/", shell=True)
    if _init or copy_js:
        run(f"scp dist/* {host}:~/trainer/dist/", shell=True)
    exec_cmd(host, f"echo '{host}' > ~/trainer/daemon_name")
    if _restart:
        restart(host)


def update_venv(host, delete=False):
    if delete:
        shutil.rmtree("~/trainer/env")
    if not dir_exists(host, "~/trainer/env"):
        create_dir(host, "~/trainer/env")
        print(exec_cmd(host, "python3.7 -m virtualenv -p python3.7 ~/trainer/env"))
    run(f"scp deploy_scripts/requirements.txt {host}:~/trainer/", shell=True)
    run(f"scp ~/bin/myauth {host}:~/", shell=True)
    print(exec_cmd(host, "python3 ~/myauth; rm ~/myauth"))
    print(exec_cmd(host, "source ~/trainer/env/bin/activate; pip install -r ~/trainer/requirements.txt"))


def init(host, clean_env=False):
    if dir_exists(host, "~/trainer"):
        print("WARNING. Directory already exists. It must be removed manually. Exiting")
        return
    create_dir(host, "~/trainer")
    create_dir(host, "~/trainer/dist")
    update_venv(host, clean_env)
    dstr = """#! /bin/bash

source ~/new_env/bin/activate
cd ~/trainer
python run.py
"""
    with open("._tmp.sh", "w") as f:
        f.write(dstr)
    run(f"scp ._tmp.sh {host}:~/.daemon.sh", shell=True)
    exec_cmd(host, "chmod 755 ~/.daemon.sh")


def restart(host):
    print(f"Restarting {host}")
    exec_cmd(host, "killall ssh")
    time.sleep(1)
    out, err = exec_cmd(host, "ps -ef | grep -i \"python run.py\"")
    splits = [re.split(" +", x) for x in out.decode("utf-8").split("\n")]
    for s in splits:
        if len(s) > 1 and s[1]:
            print(f"Killing {' '.join(s)}")
            print(exec_cmd(host, f"kill -s SIGTERM {s[1]}"))
    p = mp.Process(target=exec_cmd, args=[host, f"nohup ~/.daemon.sh >> ~/nohup.out"])
    p.start()
    time.sleep(1)
    p.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=lambda x: x.split(","),
                        default="mc15pc15@10.5.0.96,prototype@10.5.1.93,user@10.5.0.92")
    parser.add_argument("--clean-env", action="store_true")
    parser.add_argument("--copy-js", action="store_true")
    parser.add_argument("--force-init", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--restart-only", action="store_true")
    args = parser.parse_args()
    curdir = os.path.abspath(os.curdir)
    if args.compile:
        build()
    os.chdir(curdir)
    for host in args.hosts:
        if args.restart_only:
            restart(host)
        else:
            deploy(host, args.force_init, args.copy_js, args.clean_env, args.restart)


if __name__ == '__main__':
    main()
